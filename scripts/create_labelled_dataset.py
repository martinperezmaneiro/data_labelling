#!/usr/bin/env python
"""
This script creates hdf5 files that contains:
 - DATASET/MCHits          - MC hits table with segmentation labels
 - DATASET/MCVoxels        - voxelized MC hits table with segmentation labels
 - DATASET/RecoVoxels      - voxelized Reco hits table with segmentation labels
 - DATASET/EventsInfo      - table that contains EventID, source and destination directory, and  binary classification label
 - DATASET/BinInfo         - table that stores info about bins
 - DATASET/IsauraInfo      - if added, table that contains the isaura tracks info

It takes a CONFIG FILE with the following information:
 - files_in              - string with the reco files we want to label
 - file_out              - string with the name of the output file that will contain all the labelled data from the input files

 - detector_db           - detector database name for 'sipm' binning
 - binning               - Type of binning. 'regular' uses the values below, 'sipm' uses the detector database for (x, y) and values
                           below for z.
 - min_pos               - tuple with the  min position of the hits for each coordinate (in mm)
 - min_pos               - tuple with the  max position of the hits for each coordinate (in mm)
 - voxel_size            - tuple with the size of the voxels for each coordinate (in mm)

 - data_type             - string with the kind of data to label ('doublescape' for double scape, '0nubb' for neutrinoless double beta events)
 - city                  - name of the city to label (after MC); now is adapted for 'beersheba' or 'sophronia'

 - blob_ener_loss_th     - threshold for the main blob class labelling (in terms of percentage of loss energy at the end
                           of the track with respect to the total track energy)
 - blob_ener_th          - threshold for the main blob class labelling (in terms of absolute energy lost at the end of the track)

 - mc_label              - bool that indicates if the process does the labelling to the MC data
 - reco_label            - bool that indicates if the process does the labelling to the Reco data, requires mc_label True
 - Rmax                  - value for the fiducial cut, if NaN the cut is not performed
 - ghost_label           - value for the voxels that cannot be assigned to any of the segmentation classes (they come from spureous hits)
 - max_distance          - value of the maximum distance between voxels to perform the group counting algorythm, usually sqrt(3); if None, grouping is not performed
 - add_isaura_info       - bool that indicates if we want to add the isaura tracks info to the file; we need to have the isaura
                           files in an analogue directory as the reco files_in (that just changes the name of the cities in it)
"""

import sys
import os
import tables as tb
import numpy  as np
import pandas as pd

from glob import glob
from time import time
from invisible_cities.io                import dst_io as dio
from invisible_cities.core  .configure  import configure
from invisible_cities.cities.components import index_tables
from invisible_cities.database          import load_db as db

from labelling.file_labelling import label_file, create_final_dataframes
from utils.grouping_utils     import label_event_elements
from utils.bin_utils          import bins_creator_sipm, bins_creator_regular, create_bins


data_type_mapping = {'doublescape':'conv', '0nubb':'none'}
column_name_dict = {'sophronia':['event', 'X', 'Y', 'Z', 'Ec'], 'beersheba':['event', 'X', 'Y', 'Z', 'E']}
group_name_dict = {'sophronia':'RECO', 'beersheba':'DECO'}
table_name_dict = {'sophronia':'Events', 'beersheba':'Events'}

if __name__ == "__main__":

    config   = configure(sys.argv).as_namespace
    filesin  = np.sort(glob(os.path.expandvars(config.files_in)))
    fileout  = os.path.expandvars(config.file_out)
    start_id = 0
    if os.path.isfile(fileout):
        raise Exception('output file exist, please remove it manually')
    
    # Create bins
    min_, max_, size_ = config.min_pos, config.max_pos, config.voxel_size
    if config.binning == 'sipm':
        sipm_db = db.DataSiPM(config.detector_db, 0)
        bin_info = bins_creator_sipm(sipm_db, min_[-1], max_[-1], size_[-1])
    if config.binning == 'regular':
        bin_info = bins_creator_regular(min_, max_, size_)
    bins, bin_info = create_bins(bin_info)

    for i, f in enumerate(filesin):
        start_time = time()
        print(i, f)
        # total_size, voxel_size, start_bin = config.total_size, config.voxel_size, config.start_bin

        #We check if a file has empty dataframes; it happens sometimes
        check_df = dio.load_dst(f, 'MC', 'hits')
        if check_df.empty:
            print('This file has empty dataframes')
            continue
        city_name = config.city
        label_file_dfs = label_file(f,
                                    bins,
                                    sig_creator       = data_type_mapping[config.data_type],
                                    blob_ener_loss_th = config.blob_ener_loss_th,
                                    blob_ener_th      = config.blob_ener_th,
                                    reco_group        = group_name_dict[city_name],
                                    reco_table        = table_name_dict[city_name],
                                    reco_columns      = column_name_dict[city_name],
                                    mc_label          = config.mc_label,
                                    reco_label        = config.reco_label,
                                    Rmax              = config.Rmax,
                                    evt_list          = None, 
                                    ghost_label       = config.ghost_label)
        labelled_MC_voxels, labelled_MC_hits, labelled_reco_voxels, eventInfo, binsInfo, isauraInfo = create_final_dataframes(label_file_dfs,
                                                                                                                              start_id,
                                                                                                                              f,
                                                                                                                              fileout,
                                                                                                                              bin_info,
                                                                                                                              detector_db       = config.detector_db,
                                                                                                                              binning           = config.binning,
                                                                                                                              Rmax              = config.Rmax,
                                                                                                                              blob_ener_loss_th = config.blob_ener_loss_th,
                                                                                                                              blob_ener_th      = config.blob_ener_th,
                                                                                                                              max_distance      = config.max_distance,
                                                                                                                              add_isaura_info   = config.add_isaura_info)
        if config.max_distance != None:
            if config.mc_label:
                labelled_MC_voxels = label_event_elements(labelled_MC_voxels, 
                                                          config.max_distance)
            if config.mc_label and config.reco_label:
                labelled_reco_voxels = label_event_elements(labelled_reco_voxels, 
                                                            config.max_distance, 
                                                            coords = ['xbin', 'ybin', 'zbin'],
                                                            ene_label = 'energy')
                
        start_id +=len(eventInfo)
        with tb.open_file(fileout, 'a') as h5out:
            dio.df_writer(h5out, labelled_MC_hits    , 'DATASET', 'MCHits'    , columns_to_index=['dataset_id'])
            dio.df_writer(h5out, labelled_MC_voxels  , 'DATASET', 'MCVoxels'  , columns_to_index=['dataset_id'])
            dio.df_writer(h5out, labelled_reco_voxels, 'DATASET', 'RecoVoxels', columns_to_index=['dataset_id'])
            dio.df_writer(h5out, eventInfo           , 'DATASET', 'EventsInfo', columns_to_index=['dataset_id'], str_col_length=128)

            if isauraInfo.empty:
                pass
            else:
                dio.df_writer(h5out, isauraInfo    , 'DATASET', 'IsauraInfo')

        print((time() - start_time)/60, 'mins')

    #I try writing here bins info to get only one line in the final dataframe
    with tb.open_file(fileout, 'a') as h5out:
        dio.df_writer(h5out, binsInfo          , 'DATASET', 'BinsInfo', str_col_length=16)
    #Ahora supuestamente los dfs marcados con columns_to_index con la siguiente función harían que la columna escogida pasara a ser su index
    #Pero creo que no funciona porque usan algo como .attr para sacar los atributos de cada df y yo probé y me dan vacíos, cuando entiendo que
    #deberían ser el columns_to_index para que haga algún cambio (mirar la función en IC para entender a lo que me refiero)
    index_tables(fileout)
