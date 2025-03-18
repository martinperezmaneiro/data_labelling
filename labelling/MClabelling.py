import numpy  as np
import pandas as pd

from utils.labelling_utils import add_hits_labels_MC, voxel_labelling_MC, hit_data_cuts

from utils.add_extreme_utils import add_vox_ext_label

from invisible_cities.io   import dst_io as dio

def labelling_MC(directory, bins, sig_creator = 'conv', blob_ener_loss_th = None, blob_ener_th = None, Rmax = np.nan, evt_list = None):
    '''
    Performs hit labelling (binclass and segclass), voxelization of the hits (gives us the energy
    per voxel, adding up all the hits that fall inside a voxel) and voxel segclass labelling.

    Args:
        directory: STR
    Contains the directory of a file with several events with Monte Carlo information.

        bins: LIST
    Contains the binning in the 3 dimensions.

        sig_creator: STR
    If 'conv', signal will be the double scape data.
    If 'none', signal will be the neutrinoless decay data.

        blob_ener_loss_th: FLOAT
    Energy loss percentage of total track energy for the last hits that establishes a threshold for the blob class.

        blob_ener_th: FLOAT
    Energy threshold for the last hits of a track to become blob class.

        Rmax: NaN or FLOAT
    Value to perform the fiducial cut of the hits. If NaN, the cut is not done.

    RETURNS:
        voxelization_df: DATAFRAME
    It contains the positions, energies and labels for each voxel of each event in a single file.

    '''

    #Obtenemos la información de partíuclas y hits de un fichero en concreto
    mcpart = dio.load_dst(directory, 'MC', 'particles')
    mchits = dio.load_dst(directory, 'MC', 'hits')

    #Pick events from list
    if evt_list is not None:
        mcpart = mcpart[np.isin(mcpart.event_id, evt_list)]
        mchits = mchits[np.isin(mchits.event_id, evt_list)]

    #Seleccionamos los hits activos
    mchits = mchits[mchits.label == 'ACTIVE']

    #Etiquetamos los hits
    labelled_hits = add_hits_labels_MC(mchits, mcpart, sig_creator = sig_creator,
                                       blob_ener_loss_th = blob_ener_loss_th, blob_ener_th = blob_ener_th)
    
    #try to free some memory
    del mchits, mcpart

    #Hacemos los cortes en los hits
    labelled_hits = hit_data_cuts(labelled_hits, bins, Rmax = Rmax)

    # Vozelize with new function
    voxelization_df = voxel_labelling_MC(labelled_hits, bins)

    # Add extreme information to voxels using the hits
    voxelization_df = add_vox_ext_label(labelled_hits, voxelization_df, bins)

    # Make sure at least the extreme voxels have a blob label (this replaces the small blob mask for MC true hits)
    # Adding 3 as the label for a voxel with both extremes
    voxelization_df.loc[voxelization_df['extlabel'].isin([1, 3])    & (voxelization_df.binclass == 0), 'segclass'] = 3
    voxelization_df.loc[voxelization_df['extlabel'].isin([1, 2, 3]) & (voxelization_df.binclass == 1), 'segclass'] = 3

    # Reorder to match previous approach
    voxelization_df['segclass'] = voxelization_df['segclass'].astype(int)

    return voxelization_df, labelled_hits
