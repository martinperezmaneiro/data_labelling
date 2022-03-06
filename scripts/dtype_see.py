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
    
    tag = config.tag
    get_cutnum = lambda x:int(re.match(r"([a-z]+)([0-9]+)",
                                       x.split('/')[-1].replace(tag, '').split('.')[0], re.I).groups()[-1])
    
    files_to_merge = sorted(filesin, key = get_cutnum)

    #We copy in the output file the initial file information and delete the non wanted dataframes
    print(files_to_merge[0])
    with tb.open_file(files_to_merge[0], 'r') as h5in:
        print(h5in.root.DATASET.EventsInfo[:].dtype['event_id'])

    if config.voxel_type == 'Beersheba':
        voxel_tbpath = 'DATASET/BeershebaVoxels'
        delete_voxel_tbpath = '/DATASET/MCVoxels'

    #Now we do a loop on the other files to fill the output file
    for filein in files_to_merge[1:]:
        print(filein)
        with tb.open_file(filein, 'r') as h5in:
            print(h5in.root.DATASET.EventsInfo[:].dtype['event_id'])

