import numpy as np

from labelling.file_labelling               import create_final_dataframes, label_file
from utils.beersheba_labelling_utils        import label_neighbours_ordered
from utils.grouping_utils                   import label_event_elements
from utils.statistics_utils.stats_functions import get_ev_list_stats

def event_label_stats(source_path, 
                      evt_list,
                      total_size = (1000, 1000, 1300), 
                      voxel_size = (5, 5, 4), 
                      start_bin = (-500, -500, 0), 
                      label_neighbours_name = 'ordered', 
                      data_type = '0nubb', 
                      blob_ener_loss_th = 0.25, 
                      blob_ener_th = None, 
                      simple  = True, 
                      relabel = True, 
                      fix_track_connection = 'none', 
                      binclass = True, 
                      segclass = True, 
                      Rmax = 500, 
                      max_distance = np.sqrt(3),
                      small_blob_th = 1.5, 
                      add_isaura_info = False):
    
    #source_path is the path to a beersheba file
    #evt_list gets the MC event numbers from a beersheba file
    
    neighbours_functions_mapping = {'ordered':label_neighbours_ordered}
    data_type_mapping = {'doublescape':'conv', '0nubb':'none'}
    
    label_file_dfs = label_file(source_path, 
                                total_size, 
                                voxel_size, 
                                start_bin,
                                neighbours_functions_mapping[label_neighbours_name],
                                sig_creator = data_type_mapping[data_type],
                                blob_ener_loss_th = blob_ener_loss_th,
                                blob_ener_th = blob_ener_th,
                                simple = simple,
                                relabel = relabel,
                                fix_track_connection = fix_track_connection,
                                binclass = binclass,
                                segclass = segclass,
                                Rmax = Rmax,
                                small_blob_th = small_blob_th, 
                                evt_list= evt_list)
    
    labelled_MC_voxels, labelled_MC_hits, labelled_beersheba, eventInfo, binsInfo, isauraInfo = create_final_dataframes(label_file_dfs,
                                                                                                                        0,
                                                                                                                        '',
                                                                                                                        '',
                                                                                                                        total_size,
                                                                                                                        voxel_size,
                                                                                                                        start_bin,
                                                                                                                        Rmax = Rmax,
                                                                                                                        blob_ener_loss_th = blob_ener_loss_th,
                                                                                                                        blob_ener_th = blob_ener_th,
                                                                                                                        small_blob_th = small_blob_th,
                                                                                                                        max_distance = max_distance,
                                                                                                                        fix_track_connection = fix_track_connection,
                                                                                                                        add_isaura_info = add_isaura_info)
    
    labelled_beersheba = label_event_elements(labelled_beersheba,
                                              max_distance,
                                              coords = ['xbin', 'ybin', 'zbin'],
                                              ene_label = 'energy')
    
    ev_list_stats, var_list = get_ev_list_stats(labelled_beersheba)
    
    topoInfo = eventInfo[['event_id', 'binclass', 'dataset_id']]
    var_dict = {i:np.where(np.isin(topoInfo.dataset_id, j), True, False) for i, j in zip(var_list, ev_list_stats)}
    topoInfo = topoInfo.assign(**var_dict)
    
    return topoInfo, (labelled_MC_voxels, labelled_MC_hits, labelled_beersheba, binsInfo)

