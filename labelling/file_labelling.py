import pandas as pd
import numpy  as np

from labelling.MClabelling        import labelling_MC
from labelling.beershebalabelling import labelling_beersheba

def label_file(directory, 
               total_size, 
               voxel_size, 
               start_bin, 
               label_neighbours_function,
               blob_ener_loss_th = None, 
               blob_ener_th = None, 
               simple = True, 
               relabel = True,
               binclass = True,
               segclass = True, 
               Rmax = np.nan):
    '''
    Function that performs the whole beersheba labelling. Starting from the MC hits, they are labelled in three
    classes (rest, track, blob) and voxelized with the labelling_MC function. Then, with the labelling_beersheba
    we voxelize the reconstructed hits. These voxels are merged with the MC voxels in order to match information.
    Some corrections are done to those MC voxels that fall outside the beersheba voxels. Once that is done, the
    algorithm labels the empty beersheba voxels as neighbours of one of the main classes. It is also created a new
    ghost class to label those disconnected voxels that arise from the beersheba reconstruction that don't have
    a MC origin, so they don't have main class neighbours to be labelled.
    
    Args: 
        directory: STR
    Contains the directory of a file with several events with Monte Carlo and beersheba hits information.
    
        total_size: TUPLE 
    Contains the max size of the detector.
    
        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate.
    
        start_bin: TUPLE
    Contains the first voxel position for each coordinate.
    
        label_neighbours_function: FUNCTION
    Selected function to perform the neighbour labelling (so I can easily change the method)
    
        blob_ener_loss_th: FLOAT
    Energy loss percentage of total track energy for the last hits that establishes a threshold for the blob class.
        
        blob_ener_th: FLOAT
    Energy threshold for the last hits of a track to become blob class.
        
        simple: BOOL
    If True, in the voxelization we will only use hits energy information. Else, the voxelization would include 
    the information of some feature (with its ratio), which in beersheba data it's just the npeak variable. I 
    don't really know if this is an important information.
    
        relabel: BOOL
    If True, the merge_MC_beersh_voxels would try to include the external MC labelled voxels to some empty beersheba
    voxels, so we can benefit from this information. Else, this info will be lost and we would stick only to the 
    true coincident voxels.
    
        binclass: BOOL
    If True, labelling_MC function will be passed. Otherwise, it will return empty dataframes.
    
        segclass: BOOL
    If True, and if binclass is also True (because we need MC labelled voxels information), labelling_beersheba
    will be passed. Otherwise, if False or if binclass False, will return an empty dataframe.
    
        Rmax: NaN or FLOAT
    Value to perform the fiducial cut of the hits. If NaN, the cut is not done.
    
    RETURNS:
        labelled_MC_voxels: DATAFRAME
    If the conditions are satisfied (binclass = True), this contains the labelled MC voxels for each event in 
    the file.
        
        labelled_MC_hits: DATAFRAME
    If the conditions are satisfied (binclass = True), this contains the labelled MC hits for each event in the
    file. We will use them to plot nicer images.
        
        labelled_beersheba: DATAFRAME
    If the conditions are satisfied (binclass and segclas = True), this contains the labelled beersheba voxels
    for each event in the file.
    '''
    
    #Just in case binclass and segclass are False, to return something
    labelled_MC_voxels, labelled_MC_hits, labelled_beersheba = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if binclass:
        labelled_MC_voxels, labelled_MC_hits = labelling_MC(directory, 
                                                            total_size, 
                                                            voxel_size, 
                                                            start_bin, 
                                                            blob_ener_loss_th = blob_ener_loss_th, 
                                                            blob_ener_th = blob_ener_th, 
                                                            Rmax = Rmax)
    
    if binclass and segclass:
        labelled_beersheba = labelling_beersheba(directory, 
                                                 total_size, 
                                                 voxel_size, 
                                                 start_bin, 
                                                 labelled_MC_voxels, 
                                                 label_neighbours_function, 
                                                 simple = simple, 
                                                 relabel = relabel, 
                                                 Rmax = Rmax)
    
        #Rename to match the names in the next_sparseconvnet functions
        labelled_beersheba = labelled_beersheba.rename(columns={'x': 'xbin', 
                                                                'y': 'ybin', 
                                                                'z': 'zbin', 
                                                                'beersh_ener': 'energy', 
                                                                'ener': 'MC_ener'})
    else:
        print('No labelling has been performed')
        
    return labelled_MC_voxels, labelled_MC_hits, labelled_beersheba
