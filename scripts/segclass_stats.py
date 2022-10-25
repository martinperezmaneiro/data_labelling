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

from utils.blob_distances_utils import bin_creator
from utils.statistics_utils.stats_functions import add_elem_number, apply_fiducial_cut, get_ghost_other_clouds, get_secondary_clouds, get_segclass_count, get_separated_segclass
from utils.statistics_utils.stats_plots     import plot_cloud_ener_distr, histogram_statistics, plot_secondary_clouds_elements


#NEXT100 FIDUCIAL CUT
apply_fiducial = False
rmax = 480 #500
zrange = [20, 1280] #[0, 1300]

#SEGCLASS
track_segclass = [2, 5]
blob_segclass  = [3, 6]

#drops the 0 energy voxels added
drop_auxiliar_voxels = True

files_in = "/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/labelled_data/0nubb/sep_track_evs/add_all_mc_voxels/prod/beersheba_*.h5"
file_out = "/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/labelled_data/0nubb/sep_track_evs/add_all_mc_voxels/truly_dropped.h5"

files_in = glob.glob(os.path.expandvars(files_in))

for i, slices in enumerate(files_in[0].split("/")[-1].split("_")):
    if slices.isnumeric():
        get_file_number = lambda file: int(file.split("/")[-1].split("_")[i])
        files_in = sorted(files_in, key = get_file_number)

#df = pd.DataFrame()

for i, file in enumerate(files_in):
    #MC_hits       = dio.load_dst(file, 'DATASET', 'MCHits')
    beersh_voxels = dio.load_dst(file, 'DATASET', 'BeershebaVoxels')
    #events_info   = dio.load_dst(file, 'DATASET', 'EventsInfo')
    #bins_info     = dio.load_dst(file, 'DATASET', 'BinsInfo')

    if drop_auxiliar_voxels:
        beersh_voxels = beersh_voxels[beersh_voxels.energy != 0]
    #beersh_voxels_voxels = add_elem_number(beersh_voxels)
    #bins = bin_creator(bins_info)

    with tb.open_file(file, 'r') as h5in:
        group = getattr(h5in.root, 'DATASET')
        if 'IsauraInfo' not in group:
            has_isaura = False
            print("File doesn't have Isaura information")
        else:
            has_isaura = True
            print("File has Isaura information")
            isaura_info   = dio.load_dst(file, 'DATASET', 'IsauraInfo')

    if apply_fiducial and has_isaura:
        beersh_voxels_nocut = beersh_voxels.copy()
        beersh_voxels = apply_fiducial_cut(isaura_info, beersh_voxels, rmax = rmax, zrange = zrange)
        cut_eff = len(beersh_voxels.dataset_id.unique()) / len(nocut_beersh_voxels.dataset_id.unique())
        print("Fiducial cut applied with an efficiency of {:.4f}".format(cut_eff))

    nevents_total = len(beersh_voxels.dataset_id.unique())
    secondary_clouds = get_secondary_clouds(beersh_voxels)
    #percentage of events with a track/blob element out of the main cloud
    blobtrack_out_evs  = secondary_clouds[(np.isin(secondary_clouds.segclass, [2, 3]))].dataset_id.unique()
    blobtrack_out_rate = len(blobtrack_out_evs) / nevents_total

    #track statistics
    track_counting = get_segclass_count(beersh_voxels, track_segclass)
    track_counting, separated_tracks, total_tracks_count, interr_tracks = get_separated_segclass(track_counting)
    ##physically separated tracks
    sep_track_evs = separated_tracks.dataset_id.unique()
    sep_track_evs_rate = len(sep_track_evs) /len(track_counting)
    ##interrupted tracks
    interr_track_evs = interr_tracks.dataset_id.unique()
    interr_track_evs_rate = len(interr_track_evs) / len(track_counting)

    #blob statistics
    blob_counting = get_segclass_count(beersh_voxels, blob_segclass)
    blob_counting, separated_blobs, total_blobs_count, interr_blobs = get_separated_segclass(blob_counting)
    ##physically separated blobs
    sep_blob_evs = separated_blobs.dataset_id.unique()
    sep_blob_evs_rate = len(sep_blob_evs) / len(blob_counting)
    ##interrupted blobs
    interr_blob_evs = interr_blobs.dataset_id.unique()
    interr_blob_evs_rate = len(interr_blob_evs) / len(blob_counting)
    ##bad blob count
    blob_count_false = beersh_voxels[(beersh_voxels.blob_success == False)][['dataset_id', 'binclass', 'nblob', 'blob_success']].drop_duplicates()
    blob_count_false_rate = len(blob_count_false) / nevents_total

    #events_df = pd.DataFrame([{'filenumber':i,
    #                           'nevents':nevents_total,
    #                           'blobtrack_out': np.array(blobtrack_out_evs),
    #                           'sep_track':np.array(sep_track_evs),
    #                           'interr_track':np.array(interr_track_evs),
    #                           'sep_blob':np.array(sep_blob_evs),
    #                           'interr_blob':np.array(interr_blob_evs),
    #                           'bad_blob_count':np.array(blob_count_false)}])
    #events_df['filename'] = file.split("/")[-1]
    rates_df  = pd.DataFrame([{'filenumber':i,
                               'nevents':nevents_total,
                               'blobtrack_out': blobtrack_out_rate,
                               'sep_track':sep_track_evs_rate,
                               'interr_track':interr_track_evs_rate,
                               'sep_blob':sep_blob_evs_rate,
                               'interr_blob':interr_blob_evs_rate,
                               'bad_blob_count':blob_count_false_rate}])
    rates_df['filename'] = file.split("/")[-1]

    with tb.open_file(file_out, 'a') as h5out:
        #dio.df_writer(h5out, events_df, 'STATS', 'events', columns_to_index = ['filenumber'], str_col_length = 128)
        dio.df_writer(h5out, rates_df,  'STATS', 'rates',  columns_to_index = ['filenumber'], str_col_length = 128)

index_tables(file_out)
