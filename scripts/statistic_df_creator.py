#!/usr/bin/env python
'''
This script creates a dataframe where each row has information for a certain file. The stored variables are:

cut_id - the number of cut of the saved file
nevent - number of events in this file
signal_nevent - number of double scape events in the file
nevent_bkg_lower_fail - number of background events with less than 1 blob
nevent_bkg_upper_fail - number of background events with more than 1 blob
nevent_sig_lower_fail - number of doublescape events with less than 2 blobs
nevent_sig_upper_fail - number of doublescape events with more than 2 blobs

'''

import os
import re

import numpy as np
import pandas as pd
import tables as tb

from glob import glob

from invisible_cities.io                import dst_io as dio
from invisible_cities.cities.components import index_tables

#Path to the labelled files where we want to perform the statistics and to the file where we want to save this info
filesin  = np.sort(glob(os.path.expandvars('/Users/mperez/NEXT/data_labelling/examples/h5files/prueba_cut24_g*.h5')))
fileout  = os.path.expandvars('/Users/mperez/NEXT/data_labelling/examples/h5files/prueba_statistics.h5') #os.getcwd() to get current directory

df = pd.DataFrame()

for i, file in enumerate(filesin):
    #cut = file.split("/")[-1].split("_")[1]  #for my test files
    cut = file.split('/')[-1].split('.')[0].split('_')[-1]  #for CESGA beersheba labelled files
    match = re.match(r"([a-z]+)([0-9]+)", cut, re.I)
    if match:
        items = match.groups()
    cut_id = int(items[-1])

    print(i, 'cutnum {}'.format(cut_id))
    
    beersh_vox = dio.load_dst(file, 'DATASET', 'BeershebaVoxels')
    
    assert np.isin('blob_success', beersh_vox.columns.values), 'Event grouping not performed for file {file}'.format(file = file)
    
    nevent = len(beersh_vox.dataset_id.unique())
    
    signal_nevent = sum(beersh_vox[['dataset_id', 'binclass']].drop_duplicates().binclass) 

    fail_event_rate = len(beersh_vox[beersh_vox.blob_success == False].dataset_id.unique())
    
    nevent_bkg_lower_fail = len(beersh_vox[(beersh_vox.nblob < 1) & (beersh_vox.binclass == 0)].dataset_id.unique()) 
    nevent_bkg_upper_fail = len(beersh_vox[(beersh_vox.nblob > 1) & (beersh_vox.binclass == 0)].dataset_id.unique()) 
    nevent_sig_lower_fail = len(beersh_vox[(beersh_vox.nblob < 2) & (beersh_vox.binclass == 1)].dataset_id.unique()) 
    nevent_sig_upper_fail = len(beersh_vox[(beersh_vox.nblob > 2) & (beersh_vox.binclass == 1)].dataset_id.unique()) 

    assert fail_event_rate == nevent_bkg_lower_fail + nevent_bkg_upper_fail + nevent_sig_lower_fail + nevent_sig_upper_fail
    df = pd.DataFrame([{'cut_id':cut_id,
                        'nevent':nevent,
                        'signal_nevent':signal_nevent,
                        'fail_event_rate':fail_event_rate,
                        'nevent_bkg_lower_fail':nevent_bkg_lower_fail, 
                        'nevent_bkg_upper_fail':nevent_bkg_upper_fail, 
                        'nevent_sig_lower_fail':nevent_sig_lower_fail,  
                        'nevent_sig_upper_fail':nevent_sig_upper_fail}])
    with tb.open_file(fileout, 'a') as h5out:
        dio.df_writer(h5out, df, 'stat', 'stat', columns_to_index=['cut_id'])

index_tables(fileout)
