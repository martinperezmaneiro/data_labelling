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
 - voxel_type - either 'MC' or 'Beersheba'
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
    
    files_to_merge = sorted(filesin)

    #We copy in the output file the initial file information and delete the non wanted dataframes
    with tb.open_file(files_to_merge[0], 'r') as h5in:
        h5in.copy_file(fout, overwrite=True)

    #Now choose the table path for the kept voxels and for the deleted ones (we just have to delete them for the first file)
    if config.voxel_type == 'MC':
        voxel_tbpath = '/DATASET/MCVoxels'
        delete_voxel_tbpath = '/DATASET/BeershebaVoxels'

    if config.voxel_type == 'Beersheba':
        voxel_tbpath = '/DATASET/BeershebaVoxels'
        delete_voxel_tbpath = '/DATASET/MCVoxels'

    #Deleting the unwanted tables
    with tb.open_file(fout, 'a') as h5out:
        h5out.remove_node(h5out.get_node(delete_voxel_tbpath))
        h5out.remove_node(h5out.get_node('/DATASET/MCHits'))
        h5out.remove_node(h5out.get_node('/DATASET/IsauraInfo'))

    #Now check that it starts from 0 in the file out
    with tb.open_file(fout, 'r+') as h5out:
        min_dataset_id = h5out.get_node('/DATASET/EventsInfo').cols.dataset_id[0]

        if min_dataset_id>0:
            h5out.get_node('/DATASET/EventsInfo').cols.dataset_id[:]-=min_dataset_id
            h5out.get_node(voxel_tbpath).cols.dataset_id[:]-=min_dataset_id
            h5out.get_node('/DATASET/EventsInfo').flush()
            h5out.get_node(voxel_tbpath).flush()

    #Now we do a loop on the other files to fill the output file
    for filein in files_to_merge[1:]:
        print(filein)
        with tb.open_file(fout, 'a') as h5out:
            with tb.open_file(filein, 'r') as h5in:
                prev_id =  h5out.get_node('/DATASET/EventsInfo').cols.dataset_id[-1]+1

                evs = h5in.get_node('/DATASET/EventsInfo')[:]
                file_start_id = evs['dataset_id'][0]
                evs['dataset_id']+=prev_id-file_start_id
                h5out.get_node('/DATASET/EventsInfo').append(evs)
                h5out.get_node('/DATASET/EventsInfo').flush()
                del(evs)
                voxs = h5in.get_node(voxel_tbpath)[:]
                voxs['dataset_id']+=prev_id
                h5out.get_node(voxel_tbpath).append(voxs)
                h5out.get_node(voxel_tbpath).flush()
                del(voxs)

    with tb.open_file(fout, 'r+') as h5out:
        h5out.get_node('/DATASET/EventsInfo').cols.dataset_id.create_index()
        h5out.get_node(voxel_tbpath).cols.dataset_id.create_index()
