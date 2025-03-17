import pandas as pd
import numpy as np
import tables as tb
import sys
from collections import defaultdict

from utils.histogram_utils import bin_creator, container_creator, mcimg
from utils.data_utils      import histog_to_coord
from utils.labelling_utils import voxel_labelling_MC, hit_data_cuts

from invisible_cities.io   import dst_io as dio

#############################################################
### CAMBIAR NOMRBE A RECO_LABELLING_UTILS, Y EN TODO LO DEMAS
#############################################################

def voxelize_reco(file, bins, labelled_vox = pd.DataFrame(), group = 'RECO', table = 'Events', column_names = ['event', 'X', 'Y', 'Z', 'Ec']):
    '''
    Voxelizes any kind of reconstructed hits (sophronia, beersheba). In addition, you can already include the binary classification
    information of each event (taken from the labelled MC voxels).
    Args:
        file: STR
    Contains the directory of a file with reconstructed events (hits: x, y, z, E). It usually will also contain the MC info
    needed in the MC labelling.

        bins: LIST
    Contains the binning in the 3 dimensions.

        labelled_vox: DATAFRAME
    Contains the labelled MC information, output of the labelling_MC function.

    RETURNS:
        voxel_df: DATAFRAME
    Contains the voxelized reconstructed hits, with their energy and normalized.
    '''
    # label_name is here always empty because we don't have segmentation for the reco hits, we will propagate the label from 
    # the MC voxels...
    id_name, coords, ene_name, label_name = column_names[0], column_names[1:4], column_names[4], []
    reco_hits = dio.load_dst(file, group, table)

    # img  = container_creator(total_size, voxel_size)
    # bins = bin_creator(img, steps = voxel_size, x0 = start_bin)

    #reco_hits = hit_data_cuts(reco_hits, detector_bins, Rmax = Rmax, coords = ['X', 'Y', 'Z'], identifier = 'event')
    #I perform the cut on beersheba data depending on the events that were cut
    #for the MC because of the fiducial volume
    labelled_vox_events = labelled_vox['event_id'].unique()

    #Check if there is a mapping between MC and the beersheba/isaura info
    with tb.open_file(file, 'r') as h5in:
        exists_map = '/Run/eventMap' in h5in

    if exists_map:
        event_mapping = dio.load_dst(file, 'Run', 'eventMap')
        map_dict = dict(zip(event_mapping.evt_number, event_mapping.nexus_evt))
        reco_hits[id_name] = reco_hits[id_name].map(map_dict)
        del event_mapping, map_dict

    # Pick only events that are labelled in the MC voxel dataframe
    reco_hits = reco_hits[np.isin(reco_hits['event'], labelled_vox_events)]

    if labelled_vox.empty != True:
        binclass_df = labelled_vox[['event_id', 'binclass']].drop_duplicates()
        assert len(binclass_df) == len(labelled_vox_events) #just in case the labelling made a mistake and assigned 2 different binclass to the same event
        reco_hits = reco_hits.merge(binclass_df.rename(columns = {'event_id':id_name}))
    else:
        reco_hits['binclass'] = None
    
    del labelled_vox, labelled_vox_events

    ###### DEPRECATED, following the previous changes

    # voxel_df = pd.DataFrame()
    # for (event_id, event_hits), binnum in zip(reco_hits.groupby(id_name), binclass):
    #     xhits, yhits, zhits = event_hits[coords[0]], event_hits[coords[1]], event_hits[coords[2]]

    #     mccoors  = np.array([xhits, yhits, zhits]).T
    #     mcenes   = np.array(event_hits[ene_name])
    #     if simple == True:
    #         ener_histo    = mcimg(mccoors, mcenes, bins)
    #         nhits_hist, _ = np.histogramdd(mccoors, bins)
    #         voxel_df = voxel_df.append(histog_to_coord(event_id, None, ener_histo, None, nhits_hist, bins, binnum = binnum))
    #     else:
    #         labels   = np.array(event_hits['npeak']) + 1 #the addition is for the functions to recognize the label
    #         #as it is based in an histogram where the algorythm looks into the nonzero coords
    #         label_histo, ener_histo, ratio_histo, nhits_hist = voxel_labelling_MC(img,
    #                                                                               mccoors,
    #                                                                               mcenes,
    #                                                                               labels,
    #                                                                               bins)
    #         voxel_df = voxel_df.append(histog_to_coord(event_id,
    #                                                    label_histo,
    #                                                    ener_histo,
    #                                                    ratio_histo,
    #                                                    nhits_hist,
    #                                                    bins,
    #                                                    binnum = binnum,
    #                                                    id_name = 'npeak'))
    # voxel_df.reset_index()

    # voxel_df = voxel_df.rename(columns = {'ener':'beersh_ener'})

    # if simple != True:
    #     voxel_df = voxel_df.rename(columns = {'ratio':'npeak_ratio'})
    #     #Reset the npeak labels values:
    #     voxel_df.npeak = voxel_df.npeak - 1

    # #Reduce the voxels to 1-1 size
    # for coord, (size, start) in zip(['x', 'y', 'z'], zip(voxel_size, start_bin)):
    #     voxel_df[coord] = voxel_df[coord] - start
    #     voxel_df[coord] = voxel_df[coord] / size

    # #Make int all the necesary values
    # for colname in voxel_df.columns:
    #     voxel_df[colname] = pd.to_numeric(voxel_df[colname], downcast = 'integer')

    nan_hits = np.isnan(reco_hits[ene_name])
    if nan_hits.any():
        print('There are some NaN hits in the reconstructed data, we proceed to remove them')
        reco_hits = reco_hits[~nan_hits]

    voxel_df = voxel_labelling_MC(reco_hits, bins, id_name=id_name, coords=coords, ene_name=ene_name, label_name=label_name)
    voxel_df = voxel_df.rename(columns = {'ener':'reco_ener'})

    return voxel_df

def label_reco_event(mc_ev, reco_ev, neighbor_shifts, ghost_label = 0):
    # Initialize mc_labels dictionary for (x, y, z) coordinates, using mc_voxels just for initial labeling
    mc_labels = {(row.x, row.y, row.z) : row.segclass for i, row in mc_ev.iterrows()}

    # Initialize label and energy dictionaries for diff_voxels
    reco_labels = {(row.X, row.Y, row.Z) : None          for i, row in reco_ev.iterrows()} # values from reco_ev hard coded because come from voxelize_reco output
    reco_energy = {(row.X, row.Y, row.Z) : row.reco_ener for i, row in reco_ev.iterrows()}

    # Label overlapping MC voxels in diff_voxels
    L = set()  # Labeled group
    for (x, y, z) in mc_labels.keys():
        if (x, y, z) in reco_labels:
            reco_labels[(x, y, z)] = mc_labels[(x, y, z)]
            L.add((x, y, z))
            
    while L:
        new_L = set()  # Track newly labeled voxels

        # Process each voxel in the current L group
        for voxel in L:
            x, y, z = voxel
            # Check each neighboring voxel
            for dx, dy, dz in neighbor_shifts:
                neighbor_voxel = (x + dx, y + dy, z + dz)

                # Only consider this neighbor if it's in E (unlabeled in diff_voxels)
                if neighbor_voxel in reco_labels and reco_labels[neighbor_voxel] is None:
                    
                    # Collect energies of labeled neighbors by class using diff_voxels energy
                    class_energy_sums = defaultdict(float)
                    
                    # Look for labeled neighbors of this E voxel
                    for d_dx, d_dy, d_dz in neighbor_shifts:
                        neighbor_neighbor_voxel = (neighbor_voxel[0] + d_dx, neighbor_voxel[1] + d_dy, neighbor_voxel[2] + d_dz)
                        
                        # Check if the neighboring voxel is labeled in diff_voxels
                        if neighbor_neighbor_voxel in reco_labels and reco_labels[neighbor_neighbor_voxel] is not None:
                            label = reco_labels[neighbor_neighbor_voxel]
                            energy = reco_energy.get(neighbor_neighbor_voxel, 0)  # Energy from reco_voxels
                            class_energy_sums[label] += energy

                    # Assign the label with the highest cumulative energy
                    if class_energy_sums:
                        best_label = max(class_energy_sums, key=class_energy_sums.get)
                        reco_labels[neighbor_voxel] = best_label
                        new_L.add(neighbor_voxel)
        
        # Update L for the next iteration
        L = new_L  # Only process newly labeled voxels in the next iteration

    lab_reco_vox = pd.DataFrame([(x, y, z, label) if label is not None else (x, y, z, ghost_label) for (x, y, z), label in reco_labels.items()], columns = ['X', 'Y', 'Z', 'segclass'])
    assert len(lab_reco_vox) == len(reco_ev), 'Something didnt match in the reco labelling'

    return reco_ev.merge(lab_reco_vox, on = ['X', 'Y', 'Z']).rename(columns = {'event':'event_id', 'X':'x', 'Y':'y', 'Z':'z'})

# def relabel_outside_voxels(merged_voxels):
#     '''
#     Function to remove the leftover voxeles (those MC with no beersheba correspondance) and use them to label
#     their nearest(s) empty beersheba voxel(s). It does it per event, so will be used inside a bigger function
#     with a whole file grouped by event.
#     Args:
#         merged_voxels: DATAFRAME
#     This dataframe contains (per EVENT) the MC labelled voxels merged with the beersheba voxels.

#     RETURNS:
#         merged_voxels: DATAFRAME
#     It now contains JUST the beersheba voxels, some of them with new labels, benefiting from the outside MC voxels.
#     '''

#     out_df = merged_voxels[merged_voxels.reco_ener.isnull()]
#     #ins_df = merged_voxels[merged_voxels.reco_ener.notnull() & merged_voxels.ratio.notnull()]
#     emp_df = merged_voxels[merged_voxels.segclass.isnull()]

#     if out_df.empty:
#         #continue
#         return merged_voxels

#     else:
#         #Combinations of the outside voxels and the empty to compute distances
#         combinations = pd.merge(out_df,
#                                 emp_df[['event_id', 'x', 'y', 'z']],
#                                 on = ['event_id'])[['event_id',
#                                                     'x_x', 'y_x', 'z_x',
#                                                     'x_y', 'y_y', 'z_y',
#                                                     'segclass',
#                                                     'ratio']]
#         del out_df, emp_df
#         combinations['distances'] = np.linalg.norm(combinations[['x_x', 'y_x', 'z_x']].values
#                                               - combinations[['x_y', 'y_y', 'z_y']].values, axis=1)
#         #Selection of the nearest empty voxels
#         switching_voxels_df = pd.DataFrame()
#         for event_id, df in combinations.groupby(['event_id', 'x_x', 'y_x', 'z_x']):
#             min_dist = min(df.distances)
#             df = df[df.distances == min_dist].drop(['x_x', 'y_x', 'z_x', 'distances'], axis = 1)
#             df = df.rename(columns = {'x_y':'x', 'y_y':'y', 'z_y':'z'})
#             switching_voxels_df = switching_voxels_df.append(df)
#         del combinations, df

#         switching_voxels_df = switching_voxels_df.sort_values(['x', 'y', 'z'])
#         #From here, the function will be removing repeated voxels with conflicts
#         #Calculating the needed quantities to resolve those conflicts
#         switching_voxels_df[['x1',
#                              'y1',
#                              'z1',
#                              'segclass1',
#                              'ratio1']] = switching_voxels_df[['x',
#                                                                'y',
#                                                                'z',
#                                                                'segclass',
#                                                                'ratio']].shift(1)

#         switching_voxels_df['dist'] = np.linalg.norm(switching_voxels_df[['x', 'y', 'z']].values
#                                                   - switching_voxels_df[['x1', 'y1', 'z1']].values, axis=1)
#         switching_voxels_df['is_same_segclass'] = switching_voxels_df.segclass.values - switching_voxels_df.segclass1.values
#         switching_voxels_df['is_same_ratio'] = switching_voxels_df.ratio.values - switching_voxels_df.ratio1.values

#         #df of conflictive voxels (those which are repeated)
#         conflict_vox_df = switching_voxels_df[switching_voxels_df.dist == 0]
#         switching_voxels_idx = np.array(switching_voxels_df.index)

#         #Adding to an array the indexes of the voxels to remove
#         dropping_index = np.array([])

#         #Dropping those with repeated segclass
#         for i in conflict_vox_df[conflict_vox_df.is_same_segclass == 0].index:
#             dropping_index = np.append(dropping_index, switching_voxels_idx[int(np.where(switching_voxels_df.index == i)[0] - 1)])
#         #dropping_index = np.array(conflict_vox_df[conflict_vox_df.is_same_segclass == 0].index)

#         #Dropping one of the voxel with same ratio and different segclass (with some cirteria)
#         ratio0 = conflict_vox_df[conflict_vox_df.is_same_ratio == 0]
#         segclass_pos_index = np.array(ratio0[ratio0.is_same_segclass > 0].index)

#         for i in segclass_pos_index:
#             dropping_index = np.append(dropping_index, switching_voxels_idx[int(np.where(switching_voxels_idx == i)[0]) - 1])
#         dropping_index = np.append(dropping_index, np.array(ratio0[ratio0.is_same_segclass < 0].index))
#         #Dropping the voxel with less ratio fo two with different ratio and different segclass
#         last_vox = conflict_vox_df[(conflict_vox_df.is_same_segclass != 0)
#                                     & (conflict_vox_df.is_same_ratio != 0)]
#         ratio_pos_index = np.array(last_vox[last_vox.is_same_ratio > 0].index)
#         for i in ratio_pos_index:
#             dropping_index = np.append(dropping_index, switching_voxels_idx[int(np.where(switching_voxels_idx == i)[0]) - 1])
#         dropping_index = np.append(dropping_index, np.array(last_vox[last_vox.is_same_ratio < 0].index))

#         switching_voxels_df = switching_voxels_df[['event_id', 'x', 'y', 'z', 'segclass']].drop(dropping_index)
#         #Add to the merged df the new labelled voxels and dropping the outside voxels
#         merged_voxels = merged_voxels.merge(switching_voxels_df, on = ['event_id', 'x', 'y', 'z'], how = 'outer')
#         merged_voxels['segclass'] = merged_voxels['segclass_y'].fillna(merged_voxels['segclass_x'])
#         merged_voxels = merged_voxels.drop(['segclass_x', 'segclass_y'], axis = 1)

#         merged_voxels = merged_voxels.drop(np.array(merged_voxels[merged_voxels.reco_ener.isnull()].index))
#         merged_voxels = merged_voxels.drop(np.array(merged_voxels[merged_voxels[['x', 'y', 'z']].duplicated()].index))
#         return merged_voxels



# def merge_mc_reco_voxels(labelled_voxels_MC, beersh_voxels, relabel = True, fix_track_connection = False):
#     '''
#     Function that does the relabelling to a complete file, i.e., it goes through all the events in each file
#     applying the relabel_outside_voxels to each event

#     Args:
#         labelled_voxels_MC: DATAFRAME
#     This DF contains MC labelled voxels for a file, i.e. the output of labelling_MC function.

#         beersh_voxels: DATAFRAME
#     Contains the beersheba voxels for a file, i.e. the output of the voxelize_beersh function

#         relabel: BOOL
#     If True, the merge_mc_reco_voxels would try to include the external MC labelled voxels to some empty beersheba
#     voxels, so we can benefit from this information. Else, this info will be lost and we would stick only to the
#     true coincident voxels.

#         fix_track_connection: STR
#     Used to solve the beersheba track desconnection problem (temporary) by adding the MC track voxels.
#     If 'track', only track MC voxels will be added. If 'all', all the MC voxels are added.
#     Otherwise this won't be done.

#     RETURNS:
#         merged_voxels: DATAFRAME
#     Contains just the beersheba voxels, without the outside MC voxels, and with a relabelling done for those
#     voxels, for several events.
#     '''

#     #We are going to eliminate the events that don't appear in the beersheba voxels
#     #because its hits had 0 energy and the voxelization histogram remained empty
#     #There are other options, but for now I'll do this one
#     coinc_evs = beersh_voxels.merge(labelled_voxels_MC, on = 'event_id', how = 'outer', indicator = True)
#     null_beer_evs = coinc_evs[coinc_evs._merge != 'both'].event_id.unique()
#     labelled_voxels_MC = labelled_voxels_MC[~np.isin(labelled_voxels_MC.event_id, null_beer_evs)]

#     merged_voxels = beersh_voxels.merge(labelled_voxels_MC[['event_id',
#                                                             'x',
#                                                             'y',
#                                                             'z',
#                                                             'ener',
#                                                             'ratio',
#                                                             'binclass',
#                                                             'segclass', 
#                                                             'ext']],
#                                         on = ['event_id', 'x', 'y', 'z', 'binclass'],
#                                         how = 'outer')
#     del labelled_voxels_MC, beersh_voxels, coinc_evs, null_beer_evs

#     if fix_track_connection == 'all':
#         #this adds all the MC voxels that fell outside
#         merged_voxels['reco_ener'] = merged_voxels.reco_ener.fillna(0)

#     if fix_track_connection == 'track':
#         #just the track voxels that fell outside are added, and the rest will be relabelled
#         merged_voxels['reco_ener'] = np.where(merged_voxels.segclass == 2,
#                                                 merged_voxels.reco_ener.fillna(0),
#                                                 merged_voxels.reco_ener)

#     if relabel:
#         length = len(merged_voxels)
#         for event_id, df in merged_voxels.groupby('event_id'):
#             no_out_voxels_df = relabel_outside_voxels(df)
#             merged_voxels = merged_voxels.merge(no_out_voxels_df[['event_id', 'x', 'y', 'z', 'segclass']],
#                                                 on = ['event_id', 'x', 'y', 'z'], how = 'left')
#             merged_voxels['segclass'] = merged_voxels['segclass_x'].fillna(merged_voxels['segclass_y'])
#             merged_voxels = merged_voxels.drop(['segclass_x', 'segclass_y'], axis = 1)

#             #Comprobamos que la función de eliminar los voxeles no hace cosas raras al mergear (esto si tal mirar
#             #de poner luego en un test digo yo)
#             if len(merged_voxels) != length:
#                 print('En el evento {} aumentan {} voxeles en merged_voxels, cuando no deberían'.format(event_id, len(merged_voxels) - length))
#                 length = len(merged_voxels)

#             prueba = merged_voxels.drop(np.array(merged_voxels[merged_voxels.reco_ener.isnull()].index))
#             if prueba[prueba.event_id == event_id].reset_index(drop = True).equals(no_out_voxels_df.reset_index(drop = True)) != True:
#                 print('En el evento {} el reasignado de voxeles out no fue bien'.format(event_id))

#     merged_voxels = merged_voxels.drop(np.array(merged_voxels[merged_voxels.reco_ener.isnull()].index))
#     #we have to sort the voxels, otherwise the neighbour filling will fail!!
#     #if we don't use fix tracks, they would be already sorted, but I prefer adding it here just in case
#     merged_voxels = merged_voxels.sort_values(['event_id', 'x', 'y', 'z'])

#     return merged_voxels


# def scale_bins(bins, voxel_size, start_bin):
#     '''
#     Scales the detector bins to unitary bins. If the input are already unitary bins, it does nothing.

#     Args:
#         bins: LIST OF ARRAYS
#     Usually, detector bins to be normalized.

#         voxel_size: TUPLE
#     Size of the voxels in each dimension.

#         start_bin: TUPLE
#     Contains the first voxel position for each coordinate.

#     RETURNS:
#         bins: LIST OF ARRAYS
#     Returns the unitary bins as they are stored in the dataframes.
#     '''
#     size = [abs(b[1] - b[0]) for b in bins]
#     is_scaled = [True if s == 1 else False for s in size]
#     if not np.array(is_scaled).any():
#         bins = [(b - st) / s for b, (s, st) in zip(bins, zip(voxel_size, start_bin))]
#     return bins

# def assign_nlabels(label_dict = {'rest':1, 'track':2, 'blob':3},
#                    corresp_label_dict = {'nrest':'rest', 'ntrack':'track', 'nblob':'blob'}):
#     '''
#     Creates a dictionary that links the pure segclasses with their neighbour ones.

#     Args:
#         label_dict: DICT
#     Has the original class names with their corresponding number.

#         corresp_label_dict: DICT
#     Has the correspondance between the neighbour class names and the pure class names.

#     RETURNS:
#         nlabel_dict: DICT
#     Has the correspondance between pure seclass number and neighbour segclass number.
#     '''

#     nlabel_dict = {}
#     for nclass in corresp_label_dict.keys():
#         label = label_dict[corresp_label_dict[nclass]]
#         nlabel_dict[label] = label + len(label_dict) #le sumo el numero de etiquetas que hay... que obviamente son 3 para nosotros
#     return nlabel_dict


# def moves(ndim):
#     '''
#     Function that returns all the movements from one voxel to its neighbours (we consider a neighbour if a part of
#     the voxel touches another, i.e. for 3dim faces are first neighbours, edges are second and vertex are third, and
#     we consider them all).

#     Args:
#         ndim: INT
#     Number of dimensions of the movements.

#     RETURNS:
#         vs: LIST
#     Contains all the posible unitary movements to the neighbours.
#     '''
#     u0 = np.zeros(ndim)
#     def u1(idim):
#         ui1 = np.zeros(ndim)
#         ui1[idim] = 1
#         return ui1.astype(int)

#     vs = (u0, u1(0), -1 * u1(0))
#     for idim in range(1, ndim):
#         us = (u0, u1(idim), -1 * u1(idim))
#         vs = [(vi + ui).astype(int) for vi in vs for ui in us]
#     vs.pop(0)

#     return vs


# def count_neighbours(voxel_segclass, coords, bins):
#     '''
#     Counts the number of neighbours of each class a voxel has. The neighbour count is performed by moving all
#     the voxels a certain position. For each class, the function will make N movements to all the neighbours
#     (26 for 3 dimensional data), in each movement it performs an histogram to count. We add the histograms
#     corresponding to the same class in each step, and finally select the desired voxels (only those that exist
#     in the event, i.e. that have a beersheba energy associated).

#     Args:
#         voxel_segclass: PANDAS SERIES
#     Segclass column of the event dataframe to know which classes are going to participate in the counting.

#         coords: NUMPY ARRAY
#     Has the coordinates of the beersheba voxels. It's shaped as (N, d), with N the number of voxels and d the
#     dimensions.

#         bins: LIST OF ARRAYS
#     Normalized bins for each dimension.

#     RETURNS:
#         nbour_counts: LIST
#     Has the number of neighbour voxels per class one voxel has. It is a list of arrays, where each array
#     corresponds to a class (the first position is the class 1 - rest, etc).
#     '''

#     model_histo, _ = np.histogramdd(coords, bins)
#     #This is like creating a mask for the positions of the voxels, for the count
#     voxel_positions = model_histo.nonzero()
#     #Take the segclass values, deleting NaN
#     seg_unique = voxel_segclass.unique()
#     seg_unique = np.sort(seg_unique[~np.isnan(seg_unique)])
#     #List to append neighbour counts for each segclass
#     nbour_counts = []
#     #We need to use all the classes in the range avaliable, otherwise our algorithm will mix classes (because
#     #we use positions of the counts to recognize the class)
#     for seg in range(1, int(max(seg_unique)) + 1):
#         #Create an empty histogram for each class to fill
#         counts = np.zeros(model_histo.shape)
#         #Create the mask for the specific class
#         segclass_mask = voxel_segclass == seg
#         #Now we select the coords that have that specific class
#         selected_coor = coords[segclass_mask]
#         ndim = len(bins)
#         for move in moves(ndim):
#             #We use norm coordinates, so it is not necessary to add the step to move the coordinates
#             coors_next     = selected_coor + move
#             #We count how many of one specific segclass are in that move
#             counts_next, _ = np.histogramdd(coors_next, bins)
#             #We add to the empty histogram the counts, in each step will be adding for a new move...
#             counts         = counts + counts_next
#         #Finally we select the values of the beersheba coordinates (because when moving we filled non existing voxels)
#         nbour_counts.append(counts[voxel_positions])
#     return nbour_counts


# def fill_df_with_nbours_ordered(mc_beersh_event, nbour_counts, nlabel_dict):
#     '''
#     This function takes the neighbour counts scores for each empty voxel and assigns them their correspondent
#     neighbour class. If one voxel has no scores, it's ignored and remains empty (will be filled afterwards,
#     looping on this function). If two classes are tied, the function chooses the most important one.
#     (blob > track > rest).

#     Args:
#         mc_beersh_event: DATAFRAME
#     Contains one event from the merge_mc_reco_voxels function output.

#         nbour_counts: LIST
#     Output of the count_neighbours function, which has the number of neighbour voxels per class one voxel has.
#     It is a list of arrays, where each array corresponds to a class (the first position is the class 1 - rest, etc).
#     Each array has the counts of the number of neighbours for every beersheba voxel.

#         nlabel_dict: DICT
#     Contains the segclass correspondances between the main and the neighbour class.

#     RETURNS:
#         mc_beersh_event: DATAFRAME
#     The event's segclass column is filled with the desired classes. It has the same structure as the input one.
#     '''

#     #We take the segclass column
#     voxel_segclass = mc_beersh_event.segclass
#     #Know whick of the rows in the DF are to fill
#     null_mask = voxel_segclass.isnull()
#     #Score for each segclass (number of neighbors that a voxel has of each class)
#     class_scores = np.array(nbour_counts).T[null_mask]
#     assert len(class_scores) == sum(null_mask), 'Something failed in the count_neighbours function'

#     empty_positions = [i for i, score in enumerate(class_scores) if (score == np.zeros(len(score))).all()]
#     final_val = len(class_scores) - len(empty_positions) #to check things

#     #Deleting the ones empty
#     class_scores = np.delete(class_scores, empty_positions, axis = 0)
#     assert final_val == len(class_scores), 'Something failed with the empty count voxels'

#     #Index of the null segclass values (the ones with true in the mask ofcourse)
#     null_index = voxel_segclass[null_mask].index

#     #Indexes of the empty voxels that had no neighbour with class
#     empty_index = null_index[empty_positions]

#     #Changing the mask to ignore the ones empty
#     null_mask[empty_index] = ~null_mask[empty_index]
#     assert len(class_scores) == sum(null_mask), 'Something failed excluding empty values of the dataframe'

#     #Take the position of the most counted segclass; in case of ties, it chooses the most important (3>2>1)
#     class_values = [np.where(score == score.max())[0].max() + 1 for score in class_scores]

#     #Transforms the pure classes into neighbour classes; if it's already a neighbour it stays this way
#     nclass_values = [nlabel_dict[classv] if np.isin(classv, list(nlabel_dict.keys())) else classv for classv in class_values]

#     #DF with the new labelled voxels, with the same indexes as the original DF
#     nclass_df = pd.DataFrame({'segclass':nclass_values}, index = voxel_segclass[null_mask].index)

#     #Join DF and do cleaning; values are sorted to avoid weird labellings
#     mc_beersh_event = mc_beersh_event.merge(nclass_df, left_index=True, right_index=True, how = 'outer').sort_values(['event_id', 'x', 'y', 'z'])
#     mc_beersh_event['segclass'] = mc_beersh_event['segclass_y'].fillna(mc_beersh_event['segclass_x'])
#     mc_beersh_event = mc_beersh_event.drop(['segclass_x', 'segclass_y'], axis = 1)
#     assert sum(mc_beersh_event.segclass.isnull()) == len(empty_positions), 'Something failed excluding the empy voxels'

#     return mc_beersh_event, empty_index


# def label_neighbours_ordered(mc_beersh_event, bins, voxel_size, start_bin,  nlabel_dict, ghost_class = 7):
#     '''
#     Takes an event of beersheba primary labelled (only with the main segclass) and assigns neighbour
#     classes to the empty voxels. It uses the fill_df_with_nbours_ordered function, that particularly fills
#     the missing voxels based just on the order they appear in the neighbour algorythm.

#     Args:
#         mc_beersh_event: DATAFRAME
#     Contains one event from the merge_mc_reco_voxels function output.

#         bins: LIST OF ARRAYS
#     Detector bins, although they will be scaled to unity bins to work in this function.

#         voxel_size: TUPLE
#     Size in each dimension for the voxels.

#         nlabel_dict: DICT
#     Contains the segclass correspondances between the main and the neighbour class.

#     RETURNS
#         mc_beersh_event: DATAFRAME
#     The event's segclass column is filled with the desired classes. It has the same structure as the input one.
#     '''

#     coords = np.array(mc_beersh_event[['x', 'y', 'z']])
#     bins = scale_bins(bins, voxel_size, start_bin)

#     voxel_segclass = mc_beersh_event.segclass
#     empty_index = []
#     while sum(voxel_segclass.isnull()) != 0:
#         nbour_counts    = count_neighbours(voxel_segclass, coords, bins)
#         mc_beersh_event, empty_index_new = fill_df_with_nbours_ordered(mc_beersh_event, nbour_counts, nlabel_dict)

#         #If the previous empty voxels were the same as the current ones, we will label them as ghost class
#         #We check that the lists have the same lenght because if they don't, the comparison of values will
#         #return an error
#         if len(empty_index) == len(empty_index_new) and (empty_index == empty_index_new).all():
#             mc_beersh_event.segclass = mc_beersh_event.segclass.replace(to_replace = np.nan, value = ghost_class)

#         #Update the condition and the list of empty index
#         voxel_segclass  = mc_beersh_event.segclass
#         empty_index = empty_index_new

#     #Turn into an integer
#     mc_beersh_event.segclass = pd.to_numeric(mc_beersh_event.segclass, downcast = 'integer')
#     return mc_beersh_event
