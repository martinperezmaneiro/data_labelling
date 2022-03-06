#!/usr/bin/env python
"""
This script merges dataset files such that dataset_id is always increasing consecutive integers array.
The final file contains
 - DATASET/TypeVoxels   - voxelized hits table with labels, Type stands for MC or Beersheba
 - DATASET/BinInfo      - table that stores info about bins
 - DATASET/EventsInfo   - table that contains EventID and binary classification label

It takes a CONFIG FILE with the following information:
 - files_in   - string with the beersheba files we want to join
 - file_out   - string with the final destination file
 - tag        - string that is a general file identificator, the same for all the incoming files
 - voxel_type - either 'MCVoxels' or 'BeershebaVoxels', the two options we have
"""


import tables as tb
from   glob   import glob
import re
import os
import sys
from invisible_cities.core  .configure import configure
import invisible_cities.io.dst_io as dio

if __name__ == "__main__":
    config  = configure(sys.argv).as_namespace
    filesin = glob(os.path.expandvars(config.files_in))
    fout = os.path.expandvars(config.file_out)
    
    tag = config.tag
    get_cutnum = lambda x:int(re.match(r"([a-z]+)([0-9]+)",
                                       x.split('/')[-1].replace(tag, '').split('.')[0], re.I).groups()[-1])
    
    files_to_merge = sorted(filesin, key = get_cutnum)
    print(files_to_merge)

    #Now we set the table path to the desired voxel type and the eventinfo
    group_name, voxel_type, bin_info, event_info = 'DATASET', config.voxel_type, 'BinsInfo', 'EventsInfo'
    
    voxels_tbpath = '/' + group_name + '/' + voxel_type
    events_tbpath = '/' + group_name + '/' + event_info

    #We save in the output file the initial file information
    initial_file = files_to_merge[0]
    voxels = dio.load_dst(initial_file, group_name, voxel_type)
    bininf = dio.load_dst(initial_file, group_name, bin_info)
    events = dio.load_dst(initial_file, group_name, event_info)
    
    with tb.open_file(fout, 'a') as h5out:
        dio.df_writer(h5out, voxels, group_name, voxel_type, columns_to_index=['dataset_id'])
        dio.df_writer(h5out, bininf, group_name, bin_info,   columns_to_index=['dataset_id'])
        dio.df_writer(h5out, events, group_name, event_info, columns_to_index=['dataset_id'], str_col_length=64)

    #Now check that it starts from 0 in the file out
    with tb.open_file(fout, 'r+') as h5out:
        min_dataset_id = h5out.get_node(events_tbpath).cols.dataset_id[0]

        if min_dataset_id>0:
            h5out.get_node(events_tbpath).cols.dataset_id[:]-=min_dataset_id
            h5out.get_node(voxels_tbpath).cols.dataset_id[:]-=min_dataset_id
            h5out.get_node(events_tbpath).flush()
            h5out.get_node(voxels_tbpath).flush()

    #Now we do a loop on the other files to fill the output file
    for filein in files_to_merge[1:]:
        print(filein)
        with tb.open_file(fout, 'a') as h5out:
            with tb.open_file(filein, 'r') as h5in:
                prev_id =  h5out.get_node(events_tbpath).cols.dataset_id[-1]+1

                evs = h5in.get_node(events_tbpath)[:]
                file_start_id = evs['dataset_id'][0]
                evs['dataset_id']+=prev_id-file_start_id
                h5out.get_node(events_tbpath).append(evs)
                h5out.get_node(events_tbpath).flush()
                del(evs)
                voxs = h5in.get_node(voxels_tbpath)[:]
                voxs['dataset_id']+=prev_id
                h5out.get_node(voxels_tbpath).append(voxs)
                h5out.get_node(voxels_tbpath).flush()
                del(voxs)

    with tb.open_file(fout, 'r+') as h5out:
        h5out.get_node(events_tbpath).cols.dataset_id.create_index()
        h5out.get_node(voxels_tbpath).cols.dataset_id.create_index()
