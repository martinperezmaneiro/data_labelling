'''
Script that computes some topological statistics to the labelled beersheba files
It computes per file some features and returns a df with length = number of files

Just need to set up the input files (if you have a bunch of them, make sure that
the structure only contains one numeric identificator separated by _ and not more),
and the name of the output file. Also you can decide whether or not apply a fiducial
cut with apply_fiducial variable, and below it the extension of this cut.

The output will be a dataframe where each line will contain information about
a file. It will have:
- file identificator
- number of events of the file
- % of events with track/blob elements out of the main track
- % of events with physically separated tracks + blobs
- % of events with interrupted tracks + blobs
- % of events with bad blob counting

To get the statistics of a bunch of files we just have to do a mean of the
desired feature weighted with the number of events.
'''

import os
import glob
import tables as tb
import pandas as pd
import numpy as np

import invisible_cities.io.dst_io       as dio
from invisible_cities.cities.components import index_tables

from utils.statistics_utils.stats_functions import apply_fiducial_cut, create_df_cloud_stats

filepath = "/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/labelled_data/0nubb/sep_track_evs/add_all_mc_voxels/"
files_in = filepath + "prod/beersheba_*.h5"
file_out = filepath + "truly_dropped.h5"
voxel_table = "BeershebaVoxels"
isaura_table = "IsauraInfo"

#NEXT100 FIDUCIAL CUT
apply_fiducial = False
rmax = 480 #500
zrange = [20, 1280] #[0, 1300]

#SEGCLASS
track_segclass = [2, 5]
blob_segclass  = [3, 6]

files_in = glob.glob(os.path.expandvars(files_in))

for i, slices in enumerate(files_in[0].split("/")[-1].split("_")):
    if slices.isnumeric():
        get_file_number = lambda file: int(file.split("/")[-1].split("_")[i])
        files_in = sorted(files_in, key = get_file_number)

#df = pd.DataFrame()

for i, file in enumerate(files_in):
    voxels = dio.load_dst(file, 'DATASET', voxel_table)

    with tb.open_file(file, 'r') as h5in:
        group = getattr(h5in.root, 'DATASET')
        if isaura_table not in group:
            has_isaura = False
            print("File doesn't have Isaura information")
        else:
            has_isaura = True
            print("File has Isaura information")
            isaura_info   = dio.load_dst(file, 'DATASET', isaura_table)

    if apply_fiducial and has_isaura:
        voxels_nocut = voxels.copy()
        voxels = apply_fiducial_cut(isaura_info, voxels, rmax = rmax, zrange = zrange)
        cut_eff = len(voxels.dataset_id.unique()) / len(voxels_nocut.dataset_id.unique())
        print("Fiducial cut applied with an efficiency of {:.4f}".format(cut_eff))

    rates_df, events_df = create_df_cloud_stats(i, file, voxels, track_segclass = track_segclass, blob_segclass  = blob_segclass)

    with tb.open_file(file_out, 'a') as h5out:
        #dio.df_writer(h5out, events_df, 'STATS', 'events', columns_to_index = ['filenumber'], str_col_length = 128)
        dio.df_writer(h5out, rates_df,  'STATS', 'rates',  columns_to_index = ['filenumber'], str_col_length = 128)

index_tables(file_out)
