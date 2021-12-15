#!/usr/bin/env python
"""
This script creates hdf5 files that contains:
 - DATASET/MCHits          - MC hits table with segmentation labels
 - DATASET/MCVoxels        - voxelized MC hits table with segmentation labels
 - DATASET/BeershebaVoxels - voxelized beersheba hits table with segmentation labels
 - DATASET/EventsInfo      - table that contains EventID, source directory and  binary classification label
 - DATASET/BinInfo         - table that stores info about bins

It takes a CONFIG FILE with the following information:
 - files_in   - string with the beersheba files we want to label
 - file_out   - string with the name of the output file that will contain all the labelled data from the input files
 - total_size - tuple with the size of the detector for each coordinate (in mm), the hits outside this margins will 
                not be processed
 - voxel_size - tuple with the size of the voxels for each coordinate (in mm)
 - start_bin  - tuple with the  min position of the hits for each coordinate (in mm)
 - label_neighbours_name - string with the name of the neighbour labelling method
 - blob_ener_loss_th - threshold for the main blob class labelling (in terms of percentage of loss energy at the end
                       of the track with respect to the total track energy)
 - blob_ener_th - threshold for the main blob class labelling (in terms of absolute energy lost at the end of the track)
 - simple - bool that indicates a way of voxelization for the beersheba hits (not very relevant, for now always True)
 - relabel - bool that indicates if the residual MC voxels are reassigned to an existent beersheba voxel
 - binclass - bool that indicates if the process does the binary labelling
 - segclass - bool that indicates if the process does the segmentation labelling, requires binclass True
 - Rmax - value for the fiducial cut, if NaN the cut is not performed
"""

import sys
import os
import tables as tb
import numpy  as np
import pandas as pd

from glob import glob
from time import time
from invisible_cities.io import dst_io    as dio
from invisible_cities.core  .configure import configure
from invisible_cities.cities.components import index_tables

from labelling.file_labelling import label_file, create_final_dataframes

#We import the different functions to label the neighbours and create a dictionary with their keywords
#For now we are only using one, but this is made just in case we want to add more

from utils.beersheba_labelling_utils import label_neighbours_ordered

neighbours_functions_mapping = {'ordered':label_neighbours_ordered}


if __name__ == "__main__":
    
    config   = configure(sys.argv).as_namespace
    filesin  = glob(os.path.expandvars(config.files_in))
    fileout  = os.path.expandvars(config.file_out)
    start_id = 0
    if os.path.isfile(fileout):
        raise Exception('output file exist, please remove it manually')
    for i, f in enumerate(filesin):
        start_time = time()
        print(i, f)
        total_size, voxel_size, start_bin = config.total_size, config.voxel_size, config.start_bin
        label_file_dfs = label_file(f,
                                    total_size,
                                    voxel_size,
                                    start_bin,
                                    neighbours_functions_mapping[config.label_neighbours_name],
                                    blob_ener_loss_th = config.blob_ener_loss_th,
                                    blob_ener_th = config.blob_ener_th,
                                    simple = config.simple,
                                    relabel = config.relabel,
                                    binclass = config.binclass,
                                    segclass = config.segclass,
                                    Rmax = config.Rmax)
        labelled_MC_voxels, labelled_MC_hits, labelled_beersheba, eventInfo, binsInfo = create_final_dataframes(label_file_dfs,
                                                                                                                start_id,
                                                                                                                f,
                                                                                                                total_size,
                                                                                                                voxel_size,
                                                                                                                start_bin,
                                                                                                                Rmax = config.Rmax)
        binsInfo  = binsInfo.drop(np.arange(1, len(binsInfo), 1), axis = 0)
        start_id +=len(eventInfo)
        with tb.open_file(fileout, 'a') as h5out:
            dio.df_writer(h5out, labelled_MC_hits  , 'DATASET', 'MCHits'         , columns_to_index=['dataset_id'])
            dio.df_writer(h5out, labelled_MC_voxels, 'DATASET', 'MCVoxels'       , columns_to_index=['dataset_id'])
            dio.df_writer(h5out, labelled_beersheba, 'DATASET', 'BeershebaVoxels', columns_to_index=['dataset_id'])
            dio.df_writer(h5out, eventInfo         , 'DATASET', 'EventsInfo'     , columns_to_index=['dataset_id'], str_col_length=128)
            dio.df_writer(h5out, binsInfo          , 'DATASET', 'BinsInfo')
        print((time() - start_time)/60, 'mins')
    #Ahora supuestamente los dfs marcados con columns_to_index con la siguiente función harían que la columna escogida pasara a ser su index
    #Pero creo que no funciona porque usan algo como .attr para sacar los atributos de cada df y yo probé y me dan vacíos, cuando entiendo que
    #deberían ser el columns_to_index para que haga algún cambio (mirar la función en IC para entender a lo que me refiero)
    index_tables(fileout)
        

