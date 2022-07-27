#!/usr/bin/env python
"""
This script creates hdf5 files that contains:
 - DATASET/MCHits          - MC hits table with segmentation labels
 - DATASET/MCVoxels        - voxelized MC hits table with segmentation labels
 - DATASET/BeershebaVoxels - voxelized beersheba hits table with segmentation labels
 - DATASET/EventsInfo      - table that contains EventID, source and destination directory, and  binary classification label
 - DATASET/BinInfo         - table that stores info about bins
 - DATASET/IsauraInfo      - if added, table that contains the isaura tracks info

It takes a CONFIG FILE with the following information:
 - files_in              - string with the beersheba files we want to label
 - file_out              - string with the name of the output file that will contain all the labelled data from the input files
 - total_size            - tuple with the size of the detector for each coordinate (in mm), the hits outside this margins will
                           not be processed
 - voxel_size            - tuple with the size of the voxels for each coordinate (in mm)
 - start_bin             - tuple with the  min position of the hits for each coordinate (in mm)
 - label_neighbours_name - string with the name of the neighbour labelling method
 - data_type             - string with the kind of data to label ('doublescape' for double scape, '0nubb' for neutrinoless double beta events)
 - blob_ener_loss_th     - threshold for the main blob class labelling (in terms of percentage of loss energy at the end
                           of the track with respect to the total track energy)
 - blob_ener_th          - threshold for the main blob class labelling (in terms of absolute energy lost at the end of the track)
 - simple                - bool that indicates a way of voxelization for the beersheba hits (not very relevant, for now always True)
 - relabel               - bool that indicates if the residual MC voxels are reassigned to an existent beersheba voxel
 - binclass              - bool that indicates if the process does the binary labelling
 - segclass              - bool that indicates if the process does the segmentation labelling, requires binclass True
 - Rmax                  - value for the fiducial cut, if NaN the cut is not performed
 - small_blob_th         - energy threshold for the blob hits to be marked as small blobs, so the voxelization always represents them
 - max_distance          - value of the maximum distance between voxels to perform the group counting algorythm, usually sqrt(3); if None, grouping is not performed
 - add_isaura_info       - bool that indicates if we want to add the isaura tracks info to the file; we need to have the isaura
                           files in an analogue directory as the beersheba files_in (that just changes the name of the cities in it)
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

from labelling.file_labelling import label_file, create_final_dataframes
from utils.grouping_utils     import label_event_elements

#We import the different functions to label the neighbours and create a dictionary with their keywords
#For now we are only using one, but this is made just in case we want to add more

from utils.beersheba_labelling_utils import label_neighbours_ordered

neighbours_functions_mapping = {'ordered':label_neighbours_ordered}
data_type_mapping = {'doublescape':'conv', '0nubb':'none'}


if __name__ == "__main__":

    config   = configure(sys.argv).as_namespace
    filesin  = np.sort(glob(os.path.expandvars(config.files_in)))
    fileout  = os.path.expandvars(config.file_out)
    start_id = 0
    if os.path.isfile(fileout):
        raise Exception('output file exist, please remove it manually')
    for i, f in enumerate(filesin):
        start_time = time()
        print(i, f)
        total_size, voxel_size, start_bin = config.total_size, config.voxel_size, config.start_bin

        #We check if a file has empty dataframes; it happens sometimes
        check_df = dio.load_dst(f, 'MC', 'hits')
        if check_df.empty:
            print('This file has empty dataframes')
            continue

        label_file_dfs = label_file(f,
                                    total_size,
                                    voxel_size,
                                    start_bin,
                                    neighbours_functions_mapping[config.label_neighbours_name],
                                    sig_creator = data_type_mapping[config.data_type],
                                    blob_ener_loss_th = config.blob_ener_loss_th,
                                    blob_ener_th = config.blob_ener_th,
                                    simple = config.simple,
                                    relabel = config.relabel,
                                    binclass = config.binclass,
                                    segclass = config.segclass,
                                    Rmax = config.Rmax,
                                    small_blob_th = config.small_blob_th)
        labelled_MC_voxels, labelled_MC_hits, labelled_beersheba, eventInfo, binsInfo, isauraInfo = create_final_dataframes(label_file_dfs,
                                                                                                                            start_id,
                                                                                                                            f,
                                                                                                                            fileout,
                                                                                                                            total_size,
                                                                                                                            voxel_size,
                                                                                                                            start_bin,
                                                                                                                            Rmax = config.Rmax,
                                                                                                                            blob_ener_loss_th = config.blob_ener_loss_th,
                                                                                                                            blob_ener_th = config.blob_ener_th,
                                                                                                                            small_blob_th = config.small_blob_th,
                                                                                                                            add_isaura_info = config.add_isaura_info)
        if config.max_distance != None:
            labelled_beersheba = label_event_elements(labelled_beersheba,
                                                      config.max_distance,
                                                      coords = ['xbin', 'ybin', 'zbin'],
                                                      ene_label = 'energy')
        start_id +=len(eventInfo)
        with tb.open_file(fileout, 'a') as h5out:
            dio.df_writer(h5out, labelled_MC_hits  , 'DATASET', 'MCHits'         , columns_to_index=['dataset_id'])
            dio.df_writer(h5out, labelled_MC_voxels, 'DATASET', 'MCVoxels'       , columns_to_index=['dataset_id'])
            dio.df_writer(h5out, labelled_beersheba, 'DATASET', 'BeershebaVoxels', columns_to_index=['dataset_id'])
            dio.df_writer(h5out, eventInfo         , 'DATASET', 'EventsInfo'     , columns_to_index=['dataset_id'], str_col_length=128)

            if isauraInfo.empty:
                pass
            else:
                dio.df_writer(h5out, isauraInfo    , 'DATASET', 'IsauraInfo')

        print((time() - start_time)/60, 'mins')

    #I try writing here bins info to get only one line in the final dataframe
    with tb.open_file(fileout, 'a') as h5out:
        dio.df_writer(h5out, binsInfo          , 'DATASET', 'BinsInfo')
    #Ahora supuestamente los dfs marcados con columns_to_index con la siguiente función harían que la columna escogida pasara a ser su index
    #Pero creo que no funciona porque usan algo como .attr para sacar los atributos de cada df y yo probé y me dan vacíos, cuando entiendo que
    #deberían ser el columns_to_index para que haga algún cambio (mirar la función en IC para entender a lo que me refiero)
    index_tables(fileout)
