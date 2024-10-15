import numpy  as np
import pandas as pd
from .histogram_utils import *
from .data_utils      import calculate_track_distances
from .add_extreme_utils import add_ext_label

def add_binclass(mchits, mcpart, sig_creator = 'conv'):
    '''
    Adds binary class to each hit depending on if its considered as signal or background.
    The two classes are 0 - background, 1 - signal
    For double scape data, we should use sig_creator = 'conv' (i.e. 208Tl data), since
    this is its creator process.
    For 0nubb data, we should use sig_creator = 'none', since this is its creator
    process.

    Args:
        mchits: DATAFRAME
    Contains the MC hits information of every event in a file.
        mcpart: DATAFRAME
    Contains the MC particles information for every event in a file.
        sig_creator: STR
    If 'conv', signal will be the double scape data.
    If 'none', signal will be the neutrinoless decay data.
    RETURNS:
        mchits_binclass: DATAFRAME
    The mchits df with a new column containing the binclass.
    '''

    # select only those particles that actually left any hit in the active
    hits_part = pd.merge(mchits, mcpart, on = ['event_id', 'particle_id'])
    hits_part = hits_part[mcpart.columns].drop_duplicates()

    # Clean particles just to have e+ and e-
    hits_part = hits_part[np.isin(hits_part.particle_name, ['e+', 'e-'])]
    
    # Create a selector of signal/background
    selector = lambda x: 1 if int(sum(x == sig_creator)) == 2 else 0

    # With this selector we have:
    # * doublescape: its signal_creator is 'conv', so the events with 2 created conv particles would be
    #                signal (events with a e+e-), and for the rest (no conv, can be whatever) background
    # * 0nubb: its signal_creator is 'none', so the events with 2 created none particles would be signal
    #          (that is, events with 0nubb), and for the rest (208Tl, 214Bi... have just one none particle,
    #          the nuclei itself; the 1eroi have also just one none particle, the e-) background

    class_label = hits_part.groupby('event_id').creator_proc.apply(selector).astype(int)
    class_label.name = 'binclass'
    mchits_binclass  = pd.merge(mchits, class_label, on = 'event_id')

    return mchits_binclass

def add_segclass(mchits, mcpart, sig_creator = 'conv', delta_loss = None, delta_e = None, label_dict={'rest':1, 'track':2, 'blob':3}):
    '''
    Add segmentation class to each hit in the file, after being filled with the binclass.
    The classes are 1 - other, 2 - track, 3 - blob.

    Track class is assigned in the following way, depending on the kind of track:
        * Double scape (208Tl): hits of the  e- and e+ that come from a gamma conversion process
        (defined like that in Geant4)
        * Neutrinoless (0nubb): hits of both e- coming from the process 'none', so called in the
        MC simulation

        * Background: hits of the most energetic e- from a compton scattering proccess / photoelectric
        effect

    The blob class is chosen for the last hits of the tracks that fulfill a specific condition
    given either by delta_loss argument of the function or delta_e (explained in the Args part).

    Finally, other class is assigned to every hit that is not in the main tracks.

    It also computes the distance between the hits of the tracks (we take advantage of the tracks info
    extraction being done here to perform this calculation)

    Args:
        mchits: DATAFRAME
    Contains the hits information plus the binclass. It is the output of the add_binclass() function.
        mcpart: DATAFRAME
    Contains the particle information.
        sig_creator: STR
    If 'conv', signal will be the double scape data.
    If 'none', signal will be the neutrinoless decay data.
        delta_loss: FLOAT
    Energy loss threshold in percentage (respect to the total track energy) for the last hits of a
    track to become blob class.
        delta_e: FLOAT
    Energy threshold for the last hits of a track to become blob class.
        label_dict: DICTIONARY
    Has the correspondence for the class names.
    RETURN:
        hits_label: DATAFRAME
    Contains the hits information with event_id, coordinates, energy, segclass, binclass
    and hits distances of the tracks (also creator and particle to get the traces and do the bragg peak study)
    '''

    # Join hits and particles, so each hit has particle information
    # Also, only work with particles that left a hit in the detector
    hits_part = pd.merge(mchits, mcpart, on = ['event_id', 'particle_id'])
    del mcpart

    # Group to have the total energy for each particle (to select main track in bkg)
    per_part_info = hits_part.groupby(['event_id',
                                       'particle_id',
                                       'particle_name',
                                       'binclass',
                                       'creator_proc']).agg({'energy':[('track_ener', sum)]})
    per_part_info.columns = per_part_info.columns.get_level_values(1)
    per_part_info.reset_index(inplace=True)

    # Select signal
    tracks_sig = per_part_info[(per_part_info.binclass == 1) &\
                               (per_part_info.particle_name.isin(['e+', 'e-']) &\
                                (per_part_info.creator_proc == sig_creator))]

    # Select background
    tracks_bkg = per_part_info[(per_part_info.binclass == 0) &\
                               (per_part_info.particle_name == 'e-') &\
                                (per_part_info.creator_proc.isin(['compt', 'phot', 'none']))]
    del per_part_info

    tracks_bkg = tracks_bkg.loc[tracks_bkg.groupby('event_id').track_ener.idxmax()] # select the most energetic e-

    # We add here track extreme label, and join to hits
    track_ext = add_ext_label(mchits, tracks_sig, tracks_bkg)
    hits_part = hits_part.merge(track_ext.drop('track_ener', axis = 1), how='outer').fillna(0)

    # Join all tracks and add track label to them
    tracks_info = pd.concat([tracks_bkg, tracks_sig]).sort_values('event_id')
    tracks_info = tracks_info.assign(segclass = label_dict['track'])
    del tracks_sig, tracks_bkg

    # Join to hits
    hits_part  = hits_part.reset_index()
    hits_label = hits_part.merge(tracks_info[['event_id', 'particle_id', 'track_ener', 'segclass']],
                                 how='outer', on=['event_id', 'particle_id'])
    del hits_part

    # Add ohter label to the leftover hits
    hits_label.segclass = hits_label.segclass.fillna(label_dict['rest'])

    # Sort hits in descendent order to do cummulative sum
    hits_label = hits_label.sort_values(['event_id', 'particle_id', 'hit_id'], ascending=[True, True, False])
    hits_label = hits_label.assign(cumenergy = hits_label.groupby(['event_id', 'particle_id']).energy.cumsum())

    # Create the % of lost energy
    hits_label = hits_label.assign(lost_ener = (hits_label.cumenergy / hits_label.track_ener).fillna(0))

    if delta_e is not None:
        # Choose by absolute energy loss
        blob_mask = (hits_label.cumenergy < delta_e)

    if delta_loss is not None:
        # Choose by relative energy loss
        blob_mask = (hits_label.lost_ener < delta_loss)

    if delta_e == None and delta_loss == None:
        raise ValueError('Neither delta_e nor delta_loss has been given a value to define the blobs')

    # For those selected in the previous step, assign blob label
    hits_label.loc[(hits_label.segclass==label_dict['track'])& blob_mask, 'segclass'] = label_dict['blob']
    del blob_mask

    #Cojo la informacion de las trazas que tienen blobs etiquetados y miro cuales de esas quedaron sin ningun blob
    blob_labelled_tracks = hits_label[hits_label.segclass == label_dict['blob']][['event_id', 'particle_id']].drop_duplicates()
    missing_blob_mask = tracks_info[['event_id', 'particle_id']].merge(blob_labelled_tracks, how='left', indicator=True)._merge == 'left_only'
    blobless_tracks = tracks_info[missing_blob_mask.values]
    del blob_labelled_tracks, missing_blob_mask

    #Localizo los hits de esas trazas (que suelen ser muy pocos por cada traza) y los etiqueto todos como blob, así no queda ninguna traza sin blob
    missing_hits_mask = hits_label[['event_id', 'particle_id']].merge(blobless_tracks, how='left', indicator=True)._merge == 'both'
    hits_label.loc[(hits_label.segclass==label_dict['track'])& missing_hits_mask.values, 'segclass'] = label_dict['blob']
    del blobless_tracks, missing_hits_mask

    #Calculo la distancia entre hits de las trazas y lo añado al df de información que tenía
    hits_label_dist = calculate_track_distances(tracks_info, hits_label)
    del tracks_info, hits_label

    #Escojo solo la información que me interesa
    hits_label_dist = hits_label_dist[['event_id', 'x', 'y', 'z', 'hit_id', 'particle_id',  'energy', 'segclass', 'binclass', 'ext', 'dist_hits', 'cumdist', 'particle_name', 'creator_proc']].reset_index(drop=True)

    return hits_label_dist

def add_hits_labels_MC(mchits, mcpart, sig_creator = 'conv', blob_ener_loss_th = None, blob_ener_th = None):
    '''
    Add binclass and segclass to the raw MC hits dataframe.

    Args:
        mchits: DATAFRAME
    Contains the MC hits information of every event in a file.

        mcpart: DATAFRAME
    Contains the MC particles information for every event in a file.

        sig_creator: STR
    If 'conv', signal will be the double scape data.
    If 'none', signal will be the neutrinoless decay data.

        blob_ener_loss_th: FLOAT
    Percentage of energy with respect to the full track energy for the last hits of
    a track to become blob class.

        blob_ener_th: FLOAT
    Energy threshold for the last hits of a track to become blob class.

    RETURNS:
        hits_clf_seg: DATAFRAME
    The mchits df with the binclass and segclass.

    '''
    hits_clf = add_binclass(mchits, mcpart, sig_creator = sig_creator)
    hits_clf_seg = add_segclass(hits_clf, mcpart, sig_creator = sig_creator,
                                delta_loss = blob_ener_loss_th, delta_e = blob_ener_th)
    return hits_clf_seg

def voxel_labelling_MC(labelled_hits, bins, coords = ['x', 'y', 'z'], id_name = 'event_id', label_name = 'segclass'):
        '''
        This function takes labelled Monte Carlo hits and voxelizes them.

        In a voxel with several different hits, the function will label the voxel as the kind of hit that layed more energy,
        regardless of the number of hits. For a 8 hit voxel:
        - Hit A kind: 3 hits with energies 2, 2, 4    ---> total = 8
        - Hit B kind: 1 hit  with energy   6          ---> total = 6
        - Hit C kind: 4 hits with energies 1, 1, 2, 1 ---> total = 5
        This voxel will be labelled as kind A.

        Args:
            labelled_hits: DATAFRAME
        Output from the add_hits_labels_MC function.

            bins: list of arrays
        Contains the bins for the 3 dimensions.

            coords: list
        Name of the coords in the labelled_hits DF.

        RETURN:
            voxel_ener: DATAFRAME
        '''
        bname = ['xbin', 'ybin', 'zbin']
        # Voxelize
        for i in range(3): labelled_hits[bname[i]] = pd.cut(labelled_hits[coords[i]], bins[i], labels = np.arange(0, len(bins[i])-1), right = False).astype('int') #adding this to check if it matches old approach
        
        # Get energy and nhits for each voxel
        voxel_ener = labelled_hits.groupby([id_name] + bname + ['binclass']).agg(energy=('energy', 'sum'), nhits = ('energy', 'count')).reset_index()

        # Get energy for each voxel and segclass, and pick the most energetic segclass for each voxel
        voxel_seg_ener = labelled_hits.groupby([id_name] + bname + [label_name]).agg({'energy': 'sum'}).reset_index()
        max_ener_seg = voxel_seg_ener.loc[voxel_seg_ener.groupby([id_name] + bname)['energy'].idxmax()].rename(columns = {'energy':'max_seg_ener'})

        # Add segclass info to voxel df, compute the ratio of energy this class deposits in that voxel (just informative)
        voxel_ener = voxel_ener.merge(max_ener_seg, how = 'left')
        voxel_ener['ratio'] = voxel_ener['max_seg_ener'] / voxel_ener['energy']
        voxel_ener = voxel_ener.drop('max_seg_ener', axis = 1)
        
        # Rename columns to be consistent with rest of the code
        for i in range(3): voxel_ener = voxel_ener.rename(columns={bname[i]:coords[i]})
        voxel_ener = voxel_ener.rename(columns = {'energy':'ener'})
        # Reorder columns
        voxel_ener = voxel_ener[[coords] + ['ener', 'ratio'] + [label_name] + ['nhits', 'binclass'] + [id_name]]

        return voxel_ener

# !!!!!!!!!!!!! DEPRECATED !!!!!!!!!!!!!!!!! Not completely, it is still used for the beersheba part to voxelize the hits from beersheba. But I think 
# I can get rid of it in an analogous form as I did for MC. Then there is still the part of neighbour labelling that it is done using this bins, but both parts I THINK are independent
# def voxel_labelling_MC(img, mccoors, mcenes, hits_id, small_b_mask, bins):
#     '''
#     This function creates a D-dimensional array that corresponds a voxelized space (we will call it histogram).
#     The bins of this histogram will take the value of the ID hits that deposit more energy within them.
#     So, this function takes mainly Monte Carlo hits with a defined segmentation class and voxelizes them.

#     i.e., in a voxel with several hits, the function will label the voxel as the kind of hit that layed more energy,
#     regardless of the number of hits. For a 8 hit voxel:
#      - Hit A kind: 3 hits with energies 2, 2, 4    ---> total = 8
#      - Hit B kind: 1 hit  with energy   6          ---> total = 6
#      - Hit C kind: 4 hits with energies 1, 1, 2, 1 ---> total = 5
#     This voxel will be labelled as kind A.

#     The IDs of the hits may be the kind of particle, or even the segmentation classes (track, blob, other...)
#     Also, gives an histogram of the energies of the hits for each voxel, using mcimg function.

#     Args:
#         img: NUMPYARRAY
#     Frame to contain the event.

#         mccoors: NUMPYARRAY
#     Coordinates of the particle hits. Having N hits, this sould be shaped as (N, D).

#         mcenes: NUMPYARRAY
#     Energies of the particle hits. Having N hits, this should be shaped as (N,).

#         hits_id: NUMPYARRAY
#     IDs for each hit. They define the kind of voxeles we will have. Having N hits, this should be shaped as (N,).

#         small_b_mask: NUMPYARRAY
#     Mask for the blob groups of hits with very little energy, so we assure them to appear in the final voxeling

#         bins: LIST OF ARRAYS
#     D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.

#     RETURN:
#         mc_hit_id: NUMPYARRAY
#     D-dimensional histogram with the IDs for the voxels.

#         mc_hit_ener: NUMPYARRAY
#     D-dimensional histogram with the energies of the voxels.

#         mc_hit_portion: NUMPYARRAY
#     D-dimensional histogram with the ratio of the energy of the most important type of particle (the type is
#     defined by its id) to the total energy, per voxel.

#         nhits_hist: NUMPYARRAY
#     D-dimensional histogram with the number of hits for each voxel.

#     '''

#     mc_hit_id      = np.zeros(img.shape)   #array 3d a llenar con los identificadores
#     mc_hit_portion = np.zeros(img.shape)   #array 3d a llenar con el porcentaje de energía de la total que se lleva la partícula más importante
#     unique_hits    = np.unique(hits_id)    #lista de identificadores de los hits (identificador puede ser tipo de particula, label, etc...)

#     mc_hit_ener     = mcimg(mccoors, mcenes, bins) #histograma de energías
#     small_b_hist, _ = np.histogramdd(mccoors, bins, weights = small_b_mask) #histograma con los hits de blobs pequeños
#     nhits_hist,   _ = np.histogramdd(mccoors, bins) #histograma con el numero de hits en cada voxel

#     #Bucle en los identificadores de los hits para hacer un histograma de energía por tipo de hit
#     histograms, nonzero = [], []        #lista de histogramas y de sus coordenadas no nulas

#     for hit_id in unique_hits:
#         hit_id_mask = hits_id == hit_id                           #mascara de cada tipo de hit
#         vox, _ = np.histogramdd(mccoors[hit_id_mask],
#                                 bins,
#                                 weights = mcenes[hit_id_mask]) #histograma de energia por tipo de hit
#         histograms.append(vox)                                 #lista de histogramas
#         nonzero.append(np.array(vox.nonzero()).T)              #lista con las coordenadas no nulas
#     del mccoors, mcenes, hits_id, small_b_mask

#     #Bucle recorriendo los voxeles no nulos para comparar el valor de cada histograma

#     for nz in nonzero: #recorre cada tipo de hit con voxeles no nulos particulares (los hits tipo 1 cayeron en ciertos voxeles, los tipo 2 en otros etc...)
#         for i in nz:   #aqui recorre cada coordenada de cada tipo de hit
#             nonzero_coors = tuple(i)

#             if mc_hit_id[nonzero_coors] != 0 and mc_hit_portion[nonzero_coors] != 0:
#                 continue        #si cierto voxel ya ha sido llenado (es decir, si su valor no es cero)
#                                 #pasamos de volver a hacerle cosas

#             #Bucle en los histogramas para ver cual tiene el valor más grande en cada voxel,
#             #revelándome así qué tipo de voxel es

#             vox_eners = [] #contenedor de los valores de voxel para todos los histogramas
#             for histo in histograms:
#                 vox_eners.append(histo[nonzero_coors])

#             vox_eners = np.array(vox_eners)
#             assert len(vox_eners) == len(unique_hits)

#             # Ahora debemos escoger la etiqueta del voxel;
#             # Primero miramos si es un blob pequeño, es decir que su posición en el histograma de small_b no sea cero
#             # Si eso se cumple, se asigna a selected_id la última posición (que se corresponde con la etiqueta blob)
#             if small_b_hist[nonzero_coors] != 0:
#                 selected_id = -1

#             # Si no, mira la posición del elemento mayor en vox_eners
#             else:
#                 selected_id = vox_eners.argmax()

#             selected_id = vox_eners.argmax()

#             mc_hit_id[nonzero_coors] = unique_hits[selected_id]   #toma dicha posición de la lista unique_hits y se la asigna a la posición correspondiente en el array vacío

#             max_ener   = vox_eners[selected_id]     #energía de la partícula más importante en un voxel
#             total_ener = mc_hit_ener[nonzero_coors] #la energía total contenida en ese voxel
#             mc_hit_portion[nonzero_coors] = max_ener / total_ener

#     return mc_hit_id, mc_hit_ener, mc_hit_portion, nhits_hist


def hit_data_cuts(hits, bins, Rmax = np.nan, coords = ['x', 'y', 'z'], identifier = 'event_id'):
    '''
    This function performs the fiducial and boundary cuts to the input hits.

    Args:
        hits: DATAFRAME
    Contains the hits information.

        bins: LIST OF ARRAYS
    Contains the desired bins in each coordinate.

        Rmax: NaN OR FLOAT
    Value to perform the fiducial cut of the hits. If NaN, the cut is not done.

        coords: LIST
    Title of the columns for the coordinates.

    RETURNS:
        event_cut: DATAFRAME
    The same dataframe with the cut performed, where the events that fell outside
    the fiducial volume are not considered.
    '''

    #Creo el boundary cut (elimina hits fuera del tamaño del detector deseado)
    binsX, binsY, binsZ = bins
    del bins
    boundary_cut = (hits[coords[0]]>=binsX.min()) & (hits[coords[0]]<=binsX.max())\
                 & (hits[coords[1]]>=binsY.min()) & (hits[coords[1]]<=binsY.max())\
                 & (hits[coords[2]]>=binsZ.min()) & (hits[coords[2]]<=binsZ.max())

    #Creo el fiducial cut (toma los hits dentro de cierto radio)
    if np.isnan(Rmax):
        fiducial_cut = pd.Series(np.ones(len(hits), dtype=bool)) #creates a mask with all trues
    else:
        fiducial_cut = (hits[coords[0]]**2+hits[coords[1]]**2)<Rmax**2

    #Finalmente escojo dichos hits
    #hits_cut = hits[boundary_cut & fiducial_cut].reset_index(drop = True)

    #Tomo los eventos que NO cumplieron los requisitos de los cortes
    del_evs = hits[~(boundary_cut & fiducial_cut)][identifier].unique()

    #Del DF original tomo solo los hits de los eventos que cayeron en el volumen
    #fiducial
    event_cut = hits[~np.isin(hits[identifier], del_evs)]

    return event_cut

# def add_small_blob_mask(labelled_hits, small_blob_th = 0.1):
#     '''
#     Takes the add_hits_labels_MC output and creates a mask that marks all the small blob hits to make sure
#     afterwards that they get representation in the voxelization.

#     Args:
#         labelled_hits: DATAFRAME
#     Output of the add_hits_label_MC function.

#         small_blob_th: FLOAT
#     Threshold for the energy of a group of blob hits to become marked.

#     RETURNS:
#         labelled_hits: DATAFRAME
#     The same as in the input, but with a new column called small_b with the mask.
#     '''

#     per_label_info = labelled_hits.groupby(['event_id',
#                                         'particle_id',
#                                         'segclass']).agg({'energy':[('group_ener', sum)]})
#     per_label_info.columns = per_label_info.columns.get_level_values(1)
#     per_label_info.reset_index(inplace=True)

#     sb_mask = ((per_label_info.group_ener < small_blob_th) & (per_label_info.segclass == 3)).values
#     per_label_info['small_b'] = sb_mask

#     labelled_hits = labelled_hits.merge(per_label_info, on = ['event_id', 'particle_id', 'segclass'])

#     return labelled_hits
