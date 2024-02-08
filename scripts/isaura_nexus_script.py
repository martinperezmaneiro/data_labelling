import sys
import tables as tb
import pandas as pd

from utils.isaura_mc_utils import track_blob_info_creator_extractor, convert_df_to_hits
from invisible_cities.evm  import event_model       as evm
from invisible_cities.io.dst_io import df_writer

from invisible_cities.core.configure       import configure
from invisible_cities.core.system_of_units import keV, mm


if __name__ == "__main__":

    config   = configure(sys.argv).as_namespace
    infile   = config.files_in
    outfile  = config.file_out

    voxel_x, voxel_y, voxel_z = config.voxel_size
    vox_size = [voxel_x * mm, voxel_y * mm, voxel_z * mm]

    track_creator = track_blob_info_creator_extractor(vox_size=vox_size    , energy_type=evm.HitEnergy.E,
                                                    strict_vox_size=False, energy_threshold=config.ethr * keV,
                                                    min_voxels=config.min_vox   , blob_radius=config.blob_rad * mm)


    df = pd.read_hdf(infile, 'MC/hits')

    print('Analyzing file '+str(infile))    

    evts   = df.event_id.unique()
    tracks = []

    for i, evt in enumerate(evts):
        df_evt      = df[df.event_id == evt]

        hitcol      = evm.HitCollection(evt, 0)
        hitcol.hits = convert_df_to_hits(df_evt)
        if len(hitcol.hits) == 0:
            print('Event'+str(evt)+'has no reconstructed hits.')
            continue

        final_df, vox, hits = track_creator(hitcol)

        if len(final_df) == 0: continue

        tracks.append(final_df)

    df_all = pd.concat(tracks)
    with tb.open_file(outfile, 'a') as h5out:
        df_writer(h5out, df_all, 'DATASET', 'IsauraMCInfo')
    #df_all.to_hdf(outfile, key='DATASET/IsauraMCInfo', mode='w')