import pandas as pd
import numpy  as np
import os

from invisible_cities.io          import dst_io as dio

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
               Rmax = np.nan,
               small_blob_th = 0.1):
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

        small_blob_th: FLOAT
    Threshold for the energy of a group of blob hits to be marked as a small blob.
    
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
                                                            Rmax = Rmax,
                                                            small_blob_th = small_blob_th)
    
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


def create_final_dataframes(label_file_dfs,
                            start_id,
                            directory,
                            destination_directory,
                            total_size,
                            voxel_size,
                            start_bin,
                            Rmax = np.nan,
                            blob_ener_loss_th = None,
                            blob_ener_th = None,
                            small_blob_th = None,
                            add_isaura_info = False): 
    '''
    This function takes the output of label_file function and prepares the data to be saved in a h5 file.
    It will return a dataframe with the bins information of the voxelization of the hits, a dataframe with
    the information of each event, and the dataframes with hits/voxels from the input will have in the output
    an identifier that maps this each hit/voxel with an event in the event info dataframe, so we can track back 
    to its origin data.
    
    Args:
        label_file_dfs: TUPLE OF DATAFRAMES
    It's directly the output of the label_file function, that contains three dataframes: the labelled MC voxels,
    the labelled MC hits and the labelled beersheba voxels.
    
        start_id: INT
    Number to do the mapping between event information and its hits/voxels. It's actualized for every input file
    that we are going to add to the output file in the script that performs the creation of the labelled file.
    
        directory: STR
    Directory of the current file labelled to add its information to the event information df.

        destination_directory: STR
    Directory where the final information is stored. This is created because when merging all the files to train
    the neural network, we will try to get the minimum vaulable information, and so we can keep track and link
    the big file with a file that contains more labelling information.
    
        total_size: TUPLE 
    Contains the max size of the detector.
    
        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate.
    
        start_bin: TUPLE
    Contains the first voxel position for each coordinate.

        Rmax: FLOAT
    Value for the fiducial cut.

        blob_ener_loss_th: FLOAT
    Threshold for the last hits of a track to become blob regarding the percentage of energy lost out of 
    its total energy.

        blob_ener_th: FLOAT
    Threshold for the last hits of a track to become blob regarding a fixed value of energy.

        small_blob_th: FLOAT
    Threshold for the energy of a group of blob hits to be marked as a small blob.

        add_isaura_info: BOOL
    If True, it means that we have the isaura files in an analogue path to the beersheba file we are 
    labelling (changing beersheba for isaura), and we are going to add another DataFrame with this information.
    
    RETURNS:
        labelled_MC_voxels: DATAFRAME
    If the conditions are satisfied (binclass = True, i.e. the dataframe is not empty), this contains the 
    labelled MC voxels for each event in the file, and it has been added a dataset_id that maps each voxel
    with the event information.
        
        labelled_MC_hits: DATAFRAME
    If the conditions are satisfied (binclass = True, i.e. the dataframe is not empty), this contains the 
    labelled MC hits for each event in the file, and it has been added a dataset_id that maps each hit
    with the event information. We will use them to plot nicer images.
        
        labelled_beersheba: DATAFRAME
    If the conditions are satisfied (binclass and segclas = True, i.e. the dataframe is not empty), this 
    contains the labelled beersheba voxels for each event in the file, and it has been added a dataset_id 
    that maps each voxel with the event information.
    
        eventInfo: DATAFRAME
    Contains the information for each event: its original and destination file directory, its event_id
    and a dataset_id that maps every hit/voxel with them.
    
        binsInfo: DATAFRAME
    Contains the voxelization information and the value of the fiducial cut.

        isauraInfo: DATAFRAME
    Contains the isaura tracks information with a dataset_id to relate it to all the other event info. If 
    add_isaura_info is False or the isaura directory is not correct, it returns an empty dataframe.
    '''
    
    labelled_MC_voxels, labelled_MC_hits, labelled_beersheba = label_file_dfs
    if labelled_MC_voxels.empty:
        raise Exception('DataFrames are empty, labelling has not been performed')
    else:
        eventInfo = labelled_MC_voxels[['event_id', 'binclass']].drop_duplicates().reset_index(drop=True)
        #Making sure all datasets have the same event_id type for merging all datasets for the net
        eventInfo['event_id'] = eventInfo['event_id'].astype(np.int32)
        dct_map = {eventInfo.iloc[i].event_id : i + start_id for i in range(len(eventInfo))}
        pathname, basename = os.path.split(directory)
        label_pathname, label_basename = os.path.split(destination_directory)
        eventInfo = eventInfo.assign(dataset_id = eventInfo.event_id.map(dct_map),
                                     pathname = pathname,
                                     basename = basename,
                                     label_pathname = label_pathname,
                                     label_basename = label_basename)
        
        labelled_MC_voxels = labelled_MC_voxels.assign(dataset_id = labelled_MC_voxels.event_id.map(dct_map))
        labelled_MC_hits   = labelled_MC_hits.assign(dataset_id   = labelled_MC_hits.event_id.map(dct_map))
        
        labelled_MC_voxels = labelled_MC_voxels.drop('event_id', axis=1)
        labelled_MC_hits   = labelled_MC_hits.drop('event_id', axis=1)
        
        if labelled_beersheba.empty:
            print('Beersheba labelling has not been performed')
        else:
            labelled_beersheba = labelled_beersheba.assign(dataset_id = labelled_beersheba.event_id.map(dct_map))
            labelled_beersheba = labelled_beersheba.drop('event_id', axis=1)

        if add_isaura_info:
            isauraInfo = get_isaura_info(directory, dct_map)
            isauraInfo = isauraInfo[['dataset_id'] + [col for col in isauraInfo.columns if col != 'dataset_id']]
        else:
            isauraInfo = pd.DataFrame()
            
    min_x, min_y, min_z       = start_bin[0], start_bin[1], start_bin[2]
    total_x, total_y, total_z = total_size[0], total_size[1], total_size[2]
    max_x, max_y, max_z       = min_x + total_x, min_y + total_y, min_z + total_z 
    size_x, size_y, size_z    = voxel_size[0], voxel_size[1], voxel_size[2]
    nbins_x, nbins_y, nbins_z = [(total + voxel) / voxel for total, voxel in zip(total_size, voxel_size)] 
    binsInfo = pd.Series({'min_x'   : min_x,
                          'total_x' : total_x, 
                          'size_x'  : size_x,
                          'max_x'   : max_x,
                          'nbins_x' : nbins_x,
                          'min_y'   : min_y,
                          'total_y' : total_y,
                          'size_y'  : size_y,
                          'max_y'   : max_y,
                          'nbins_y' : nbins_y,
                          'min_z'   : min_z,
                          'total_z' : total_z,
                          'size_z'  : size_z,
                          'max_z'   : max_z,
                          'nbins_z' : nbins_z,
                          'Rmax'    : Rmax,
                          'loss_th' : blob_ener_loss_th,
                          'ener_th' : blob_ener_th,
                          'sb_th'   : small_blob_th
                          }).to_frame().T
    
    return labelled_MC_voxels, labelled_MC_hits, labelled_beersheba, eventInfo, binsInfo, isauraInfo


def get_isaura_info(directory, dct_map):
    '''
    This function will get the isaura tracking info and add the corresponding dataset_id so we have
    it in the final file. If the isaura file does not exist, it returns an empty dataframe.
     
    Args:
        directory: STR
    Path to the beersheba file we are currently labelling/working with. Needs to have the same structure
    as the isaura path, but changing the names of the cities in order to work.

        dct_map: DICT
    Map of the event_id of a individual file to the dataset_id we have as a grouped file.

    RETURNS:
        isaura_info: DATAFRAME
    Contains the tracks info of the isaura output with the corresponding dataset_id for each track.
    '''
    
    #I change the directory name to the one that contains isauras
    isaura_path = directory.replace('beersheba', 'isaura')

    if os.path.isfile(isaura_path):
        #Loading the track info dataframe
        isaura_info = dio.load_dst(isaura_path, 'Tracking', 'Tracks')

        #Mapping the event number with the dataset_id
        isaura_info = isaura_info.assign(dataset_id = isaura_info.event.map(dct_map))

    #If the file does not exist, we create an empty DF
    else:
        isaura_info = pd.DataFrame()
        
    return isaura_info
