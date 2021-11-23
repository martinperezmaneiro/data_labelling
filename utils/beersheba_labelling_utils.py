import pandas as pd
import numpy as np

from utils.histogram_utils import bin_creator, container_creator, mcimg
from utils.data_utils      import histog_to_coord
from utils.labelling_utils import voxel_labelling_MC

from invisible_cities.io   import dst_io as dio


def voxelize_beersh(beersh_dir, total_size, voxel_size, start_bin, labelled_vox = pd.DataFrame(), simple = True):
    '''
    Voxelizes the beersheba reconstructed hits. In addition, you can already include the binary classification 
    information of each event (taken from the labelled MC voxels).
    Args:
        beersh_dir: STR
    Contains the directory of a file with beersheba reconstructed events. It usually will also contain the MC info
    needed in the MC labelling.
    
        total_size: TUPLE 
    Contains the max size of the detector.
    
        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate.
    
        start_bin: TUPLE
    Contains the first voxel position for each coordinate.
    
        labelled_vox: DATAFRAME
    Contains the labelled MC information, output of the labelling_MC function.
    
        simple: BOOL
    If true, the output will just have an energy voxelization. Else, it will take npeak as hit_id for the 
    voxelization, and also would compute ratio for this variable.
    
    RETURNS:
        voxel_df: DATAFRAME
    Contains the voxelized beersheba hits, with their energy and normalized.
    '''
    
    beersh_hits = dio.load_dst(beersh_dir, 'DECO', 'Events') 
    detector_frame = container_creator(total_size, voxel_size)
    detector_bins  = bin_creator(detector_frame, steps = voxel_size, x0 = start_bin)
    
    binclass = np.array([])
    if labelled_vox.empty != True:
        binclass = np.append(binclass, 
                             [np.array(event_vox['binclass'])[0] 
                              for event_id, event_vox in labelled_vox.groupby('event_id')])
    else:
        binclass = np.append(binclass, [None] * len(beersh_hits.event.unique()))
        
    voxel_df = pd.DataFrame()
    for (event_id, event_hits), binnum in zip(beersh_hits.groupby('event'), binclass):
        xhits, yhits, zhits = event_hits['X'], event_hits['Y'], event_hits['Z']
        
        mccoors  = np.array([xhits, yhits, zhits]).T 
        mcenes   = np.array(event_hits['E'])
        if simple == True:
            ener_histo = mcimg(mccoors, mcenes, detector_bins)
            voxel_df = voxel_df.append(histog_to_coord(event_id, None, ener_histo, None, detector_bins, binnum = binnum))
        else:
            labels   = np.array(event_hits['npeak']) + 1 #the addition is for the functions to recognize the label 
            #as it is based in an histogram where the algorythm looks into the nonzero coords
            label_histo, ener_histo, ratio_histo = voxel_labelling_MC(detector_frame, 
                                                                      mccoors, 
                                                                      mcenes, 
                                                                      labels, 
                                                                      detector_bins)
            voxel_df = voxel_df.append(histog_to_coord(event_id, 
                                                       label_histo, 
                                                       ener_histo, 
                                                       ratio_histo, 
                                                       detector_bins, 
                                                       binnum = binnum, 
                                                       id_name = 'npeak'))
    voxel_df.reset_index()
    
    voxel_df = voxel_df.rename(columns = {'ener':'beersh_ener'})
    
    if simple != True:
        voxel_df = voxel_df.rename(columns = {'ratio':'npeak_ratio'})
        #Reset the npeak labels values:
        voxel_df.npeak = voxel_df.npeak - 1
    
    #Reduce the voxels to 1-1 size
    for coord, size in zip(['x', 'y', 'z'], voxel_size):
        voxel_df[coord] = voxel_df[coord] / size
        
    #Make int all the necesary values
    for colname in voxel_df.columns:
        voxel_df[colname] = pd.to_numeric(voxel_df[colname], downcast = 'integer')
    return voxel_df


def label_leftover_voxels_per_event(merged_voxels):
    '''
    Function to remove the leftover voxeles (those MC with no beersheba correspondance) and use them to label
    their nearest(s) empty beersheba voxel(s). It does it per event, so will be used inside a bigger function 
    with a whole file grouped by event.
    Args:
        merged_voxels: DATAFRAME
    This dataframe contains (per EVENT) the MC labelled voxels merged with the beersheba voxels. 
    
    RETURNS:
        merged_voxels: DATAFRAME
    It now contains JUST the beersheba voxels, some of them with new labels, benefiting from the outside MC voxels.
    '''
    
    out_df = merged_voxels[merged_voxels.beersh_ener.isnull()]
    ins_df = merged_voxels[merged_voxels.beersh_ener.notnull() & merged_voxels.ratio.notnull()]
    emp_df = merged_voxels[merged_voxels.segclass.isnull()]
    
    if out_df.empty:
        #continue
        return merged_voxels
    
    else:
        #Combinations of the outside voxels and the empty to compute distances
        combinations = pd.merge(out_df, 
                                emp_df[['event_id', 'x', 'y', 'z']], 
                                on = ['event_id'])[['event_id', 
                                                    'x_x', 'y_x', 'z_x', 
                                                    'x_y', 'y_y', 'z_y', 
                                                    'segclass', 
                                                    'ratio']]
        combinations['distances'] = np.linalg.norm(combinations[['x_x', 'y_x', 'z_x']].values 
                                              - combinations[['x_y', 'y_y', 'z_y']].values, axis=1)
        #Selection of the nearest empty voxels
        switching_voxels_df = pd.DataFrame()
        for event_id, df in combinations.groupby(['event_id', 'x_x', 'y_x', 'z_x']):
            min_dist = min(df.distances)
            df = df[df.distances == min_dist].drop(['x_x', 'y_x', 'z_x', 'distances'], axis = 1)
            df = df.rename(columns = {'x_y':'x', 'y_y':'y', 'z_y':'z'})
            switching_voxels_df = switching_voxels_df.append(df)
    
        switching_voxels_df = switching_voxels_df.sort_values(['x', 'y', 'z'])
        #From here, the function will be removing repeated voxels with conflicts
        #Calculating the needed quantities to resolve those conflicts
        switching_voxels_df[['x1', 
                             'y1', 
                             'z1', 
                             'segclass1', 
                             'ratio1']] = switching_voxels_df[['x', 
                                                               'y', 
                                                               'z', 
                                                               'segclass', 
                                                               'ratio']].shift(1)
    
        switching_voxels_df['dist'] = np.linalg.norm(switching_voxels_df[['x', 'y', 'z']].values 
                                                  - switching_voxels_df[['x1', 'y1', 'z1']].values, axis=1)
        switching_voxels_df['is_same_segclass'] = switching_voxels_df.segclass.values - switching_voxels_df.segclass1.values
        switching_voxels_df['is_same_ratio'] = switching_voxels_df.ratio.values - switching_voxels_df.ratio1.values
    
        #df of conflictive voxels (those which are repeated)
        conflict_vox_df = switching_voxels_df[switching_voxels_df.dist == 0]
        switching_voxels_idx = np.array(switching_voxels_df.index)
        
        #Adding to an array the indexes of the voxels to remove
        dropping_index = np.array([])
        
        #Dropping those with repeated segclass
        for i in conflict_vox_df[conflict_vox_df.is_same_segclass == 0].index:
            dropping_index = np.append(dropping_index, switching_voxels_idx[int(np.where(switching_voxels_df.index == i)[0] - 1)])
        #dropping_index = np.array(conflict_vox_df[conflict_vox_df.is_same_segclass == 0].index)
        
        #Dropping one of the voxel with same ratio and different segclass (with some cirteria)
        ratio0 = conflict_vox_df[conflict_vox_df.is_same_ratio == 0]
        segclass_pos_index = np.array(ratio0[ratio0.is_same_segclass > 0].index)
        
        for i in segclass_pos_index:
            dropping_index = np.append(dropping_index, switching_voxels_idx[int(np.where(switching_voxels_idx == i)[0]) - 1])
        dropping_index = np.append(dropping_index, np.array(ratio0[ratio0.is_same_segclass < 0].index))
        #Dropping the voxel with less ratio fo two with different ratio and different segclass
        last_vox = conflict_vox_df[(conflict_vox_df.is_same_segclass != 0) 
                                    & (conflict_vox_df.is_same_ratio != 0)]
        ratio_pos_index = np.array(last_vox[last_vox.is_same_ratio > 0].index)
        for i in ratio_pos_index:
            dropping_index = np.append(dropping_index, switching_voxels_idx[int(np.where(switching_voxels_idx == i)[0]) - 1])
        dropping_index = np.append(dropping_index, np.array(last_vox[last_vox.is_same_ratio < 0].index))
    
        switching_voxels_df = switching_voxels_df[['event_id', 'x', 'y', 'z', 'segclass']].drop(dropping_index)
        #Add to the merged df the new labelled voxels and dropping the outside voxels
        merged_voxels = merged_voxels.merge(switching_voxels_df, on = ['event_id', 'x', 'y', 'z'], how = 'outer')
        merged_voxels['segclass'] = merged_voxels['segclass_y'].fillna(merged_voxels['segclass_x'])
        merged_voxels = merged_voxels.drop(['segclass_x', 'segclass_y'], axis = 1)
    
        merged_voxels = merged_voxels.drop(np.array(merged_voxels[merged_voxels.beersh_ener.isnull()].index))
        merged_voxels = merged_voxels.drop(np.array(merged_voxels[merged_voxels[['x', 'y', 'z']].duplicated()].index))
        return merged_voxels


def relabelling_outside_voxels(merged_voxels):
    '''
    Function that does the relabelling to a complete file, i.e., it goes through all the events in each file 
    applying the label_leftover_voxels_per_event to each event
    
    Args:
        merged_voxels: DATAFRAME
    This DF contains MC labelled voxels merged with the beersheba voxels, for several events.
    
    RETURNS:
        merged_voxels: DATAFRAME
    Contains just the beersheba voxels, without the outside MC voxels, and with a relabelling done for those
    voxels, for several events.
    '''
    length = len(merged_voxels)
    for event_id, df in merged_voxels.groupby('event_id'):
        no_out_voxels_df = label_leftover_voxels_per_event(df)
        merged_voxels = merged_voxels.merge(no_out_voxels_df[['event_id', 'x', 'y', 'z', 'segclass']], 
                                            on = ['event_id', 'x', 'y', 'z'], how = 'left')
        merged_voxels['segclass'] = merged_voxels['segclass_x'].fillna(merged_voxels['segclass_y'])
        merged_voxels = merged_voxels.drop(['segclass_x', 'segclass_y'], axis = 1)
    
        #Comprobamos que la función de eliminar los voxeles no hace cosas raras al mergear (esto si tal mirar 
        #de poner luego en un test digo yo)
        if len(merged_voxels) != length:
            print('En el evento {} aumentan {} voxeles en merged_voxels, cuando no deberían'.format(event_id, len(merged_voxels) - length))
            length = len(merged_voxels)
            
        prueba = merged_voxels.drop(np.array(merged_voxels[merged_voxels.beersh_ener.isnull()].index))
        if prueba[prueba.event_id == event_id].reset_index(drop = True).equals(no_out_voxels_df.reset_index(drop = True)) != True:
            print('En el evento {} el reasignado de voxeles out no fue bien'.format(event_id))
            
    merged_voxels = merged_voxels.drop(np.array(merged_voxels[merged_voxels.beersh_ener.isnull()].index))
    return merged_voxels
