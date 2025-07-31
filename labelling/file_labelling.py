import pandas as pd
import numpy  as np
import tables as tb
import os

from invisible_cities.io          import dst_io as dio

from labelling.MClabelling        import labelling_MC
from labelling.recolabelling      import labelling_reco

def label_file(directory,
               bins,
               sig_creator       = 'conv',
               blob_ener_loss_th = None,
               blob_ener_th      = None,
               interpol_params   = {'interpolate':False},
               reco_group        = 'RECO',
               reco_table        = 'Events',
               reco_columns      = ['event', 'X', 'Y', 'Z', 'Ec'],
               mc_label          = True,
               reco_label        = True,
               Rmax              = np.nan,
               evt_list          = None, 
               ghost_label       = 0):
    '''
    Function that performs the whole reco labelling. Starting from the MC hits, they are labelled in three
    classes (rest, track, blob) and voxelized with the labelling_MC function. Then, with the labelling_reco
    we voxelize the reconstructed hits. These voxels are merged with the MC voxels in order to match information.
    Some corrections are done to those MC voxels that fall outside the reco voxels. Once that is done, the
    algorithm labels the empty reco voxels as neighbours of one of the main classes. It is also created a new
    ghost class to label those disconnected voxels that arise from the reconstruction that don't have
    a MC origin, so they don't have main class neighbours to be labelled.

    Args:
        directory: STR
    Contains the directory of a file with several events with Monte Carlo and reco hits information.

        bins: LIST
    Contains the binning in the 3 dimensions.

        label_neighbours_function: FUNCTION
    Selected function to perform the neighbour labelling (so I can easily change the method)

        sig_creator: STR
    If 'conv', signal will be the double scape data.
    If 'none', signal will be the neutrinoless decay data.

        blob_ener_loss_th: FLOAT
    Energy loss percentage of total track energy for the last hits that establishes a threshold for the blob class.

        blob_ener_th: FLOAT
    Energy threshold for the last hits of a track to become blob class.

        interpol_params: DCT
    Contains the parameters for interpolate the data in XY.
    
        reco_group: STR
    Group name of the hits to label.

        reco_table: STR
    Table name of the hits to label.

        mc_label: BOOL
    If True, labelling_MC function will be passed. Otherwise, it will return empty dataframes.

        reco_label: BOOL
    If True, and if mc_label is also True (because we need MC labelled voxels information), labelling_reco
    will be passed. Otherwise, if False or if mc_label False, will return an empty dataframe.

        Rmax: NaN or FLOAT
    Value to perform the fiducial cut of the hits. If NaN, the cut is not done.

        evt_list: LIST
    List of the events we want to be labelled.

        ghost_label: INT
    Label for the ghost class (spurious voxels that don't have any coincidence)

    RETURNS:
        labelled_MC_voxels: DATAFRAME
    If the conditions are satisfied (mc_label = True), this contains the labelled MC voxels for each event in
    the file.

        labelled_MC_hits: DATAFRAME
    If the conditions are satisfied (mc_label = True), this contains the labelled MC hits for each event in the
    file. We will use them to plot nicer images.

        labelled_reco_voxels: DATAFRAME
    If the conditions are satisfied (mc_label and segclas = True), this contains the labelled reco voxels
    for each event in the file.
    '''

    #Just in case mc_label and reco_label are False, to return something
    labelled_MC_voxels, labelled_MC_hits, labelled_reco_voxels = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if mc_label:
        labelled_MC_voxels, labelled_MC_hits = labelling_MC(directory,
                                                            bins,
                                                            sig_creator = sig_creator,
                                                            blob_ener_loss_th = blob_ener_loss_th,
                                                            blob_ener_th = blob_ener_th,
                                                            Rmax = Rmax,
                                                            evt_list = evt_list)
    else:
        print('No MC labelling has been performed')

    if mc_label and reco_label:
        labelled_reco_voxels = labelling_reco(directory,
                                              bins, 
                                              labelled_MC_voxels, 
                                              interpol_params,
                                              group = reco_group, 
                                              table = reco_table, 
                                              column_names = reco_columns, 
                                              ghost_label = ghost_label)

        #Rename to match the names in the next_sparseconvnet functions
        labelled_reco_voxels = labelled_reco_voxels.rename(columns={'x': 'xbin',
                                                                    'y': 'ybin',
                                                                    'z': 'zbin',
                                                                    'reco_ener': 'energy'})
                                                                    # 'ener': 'MC_ener'})
    else:
        print('No reco labelling has been performed')

    return labelled_MC_voxels, labelled_MC_hits, labelled_reco_voxels


def create_final_dataframes(label_file_dfs,
                            start_id,
                            directory,
                            destination_directory,
                            bin_info,
                            detector_db       = 'next100',
                            binning           = 'regular',
                            Rmax              = np.nan,
                            blob_ener_loss_th = None,
                            blob_ener_th = None,
                            max_distance = None,
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
    the labelled MC hits and the labelled reco voxels.

        start_id: INT
    Number to do the mapping between event information and its hits/voxels. It's actualized for every input file
    that we are going to add to the output file in the script that performs the creation of the labelled file.

        directory: STR
    Directory of the current file labelled to add its information to the event information df.

        destination_directory: STR
    Directory where the final information is stored. This is created because when merging all the files to train
    the neural network, we will try to get the minimum vaulable information, and so we can keep track and link
    the big file with a file that contains more labelling information.

        bin_info: DICT
    Contains the info of the bins.

        detector_db: STR
    Name of the detector database used for the binning (in the case of 'sipm' binning)

        binning: STR
    Binning type. Use 'regular' for directly use the specified bin values, use 'sipm' to assign a sensor based
    voxelization for (x, y).

        Rmax: FLOAT
    Value for the fiducial cut.

        blob_ener_loss_th: FLOAT
    Threshold for the last hits of a track to become blob regarding the percentage of energy lost out of
    its total energy.

        blob_ener_th: FLOAT
    Threshold for the last hits of a track to become blob regarding a fixed value of energy.

        max_distance: FLOAT
    Indicates the maximum distance between nodes to be connected for element counting.

        add_isaura_info: BOOL
    If True, it means that we have the isaura files in an analogue path to the reco file we are
    labelling (changing reco for isaura), and we are going to add another DataFrame with this information.

    RETURNS:
        labelled_MC_voxels: DATAFRAME
    If the conditions are satisfied (mc_label = True, i.e. the dataframe is not empty), this contains the
    labelled MC voxels for each event in the file, and it has been added a dataset_id that maps each voxel
    with the event information.

        labelled_MC_hits: DATAFRAME
    If the conditions are satisfied (mc_label = True, i.e. the dataframe is not empty), this contains the
    labelled MC hits for each event in the file, and it has been added a dataset_id that maps each hit
    with the event information. We will use them to plot nicer images.

        labelled_reco: DATAFRAME
    If the conditions are satisfied (mc_label and segclas = True, i.e. the dataframe is not empty), this
    contains the labelled reco voxels for each event in the file, and it has been added a dataset_id
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

    labelled_MC_voxels, labelled_MC_hits, labelled_reco = label_file_dfs
    del label_file_dfs
    if labelled_MC_voxels.empty:
        raise Exception('DataFrames are empty, labelling has not been performed')
    else:
        eventInfo = labelled_MC_voxels[['event_id', 'binclass']].drop_duplicates().reset_index(drop=True)
        #Making sure all datasets have the same event_id type for merging all datasets for the net
        eventInfo['event_id'] = eventInfo['event_id'].astype(np.int64)
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

        if labelled_reco.empty:
            #just so I don't get an error when writing an empty df
            labelled_reco = pd.DataFrame([], columns = ['dataset_id'])
            print('Reco labelling has not been performed')
        else:
            labelled_reco = labelled_reco.assign(dataset_id = labelled_reco.event_id.map(dct_map))
            labelled_reco = labelled_reco.drop('event_id', axis=1)

        if add_isaura_info:
            isauraInfo = get_isaura_info(directory, dct_map)
            isauraInfo = isauraInfo[['dataset_id'] + [col for col in isauraInfo.columns if col != 'dataset_id']]
        else:
            isauraInfo = pd.DataFrame()

    min_x, min_y, min_z       = bin_info['min']
    total_x, total_y, total_z = bin_info['total']
    max_x, max_y, max_z       = bin_info['max']
    size_x, size_y, size_z    = bin_info['size']
    nbins_x, nbins_y, nbins_z = bin_info['nbins']
    binsInfo = pd.DataFrame([{'min_x'   : min_x,
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
                          'detector_db' : detector_db,
                          'binning' : binning,
                          'Rmax'    : Rmax,
                          'loss_th' : 'None' if blob_ener_loss_th == None else blob_ener_loss_th,
                          'ener_th' : 'None' if blob_ener_th == None else blob_ener_th, # to fix that NoneType has no length
                        #   'sb_th'   : small_blob_th,
                          'max_dis' : 'None' if max_distance == None else max_distance
                          }]).infer_objects() #solves the problem of mixing str with numbers in a df to write it using IC functions

    #We add this apart bc otherwise all the elements in the df change to object
    #type, and then when writing on a file throws an error
    # binsInfo['fix_conn'] = fix_track_connection

    return labelled_MC_voxels, labelled_MC_hits, labelled_reco, eventInfo, binsInfo, isauraInfo


def get_isaura_info(directory, dct_map):
    '''
    This function will get the isaura tracking info and add the corresponding dataset_id so we have
    it in the final file. If the isaura file does not exist, it returns an empty dataframe.

    Args:
        directory: STR
    Path to the reco file we are currently labelling/working with. Needs to have the same structure
    as the isaura path, but changing the names of the cities in order to work.

        dct_map: DICT
    Map of the event_id of a individual file to the dataset_id we have as a grouped file.

    RETURNS:
        isaura_info: DATAFRAME
    Contains the tracks info of the isaura output with the corresponding dataset_id for each track.
    '''

    #I change the directory name to the one that contains isauras
    isaura_path = directory.replace('beersheba', 'isaura')
    isaura_path = directory.replace('sophronia', 'isaura')

    if os.path.isfile(isaura_path):
        #Loading the track info dataframe
        isaura_info = dio.load_dst(isaura_path, 'Tracking', 'Tracks')

        #Check if there is a mapping between MC and beersheba & sophronia/isaura info
        #Needed for 0nubb data as the MC and isaura files don't have the
        #same id, but there is a mapping in /Run/eventMap
        with tb.open_file(directory, 'r') as h5in:
            exists_map = '/Run/eventMap' in h5in

        if exists_map:
            event_mapping = dio.load_dst(directory, 'Run', 'eventMap')
            map_dict = dict(zip(event_mapping.evt_number, event_mapping.nexus_evt))
            isaura_info.event = isaura_info.event.map(map_dict)

        #Mapping the event number with the dataset_id
        isaura_info = isaura_info.assign(dataset_id = isaura_info.event.map(dct_map))

        #Drop the events that we cut when we made the fiducial current
        isaura_info.dropna(subset = ['dataset_id'], inplace = True)

    #If the file does not exist, we create an empty DF
    else:
        isaura_info = pd.DataFrame()

    return isaura_info
