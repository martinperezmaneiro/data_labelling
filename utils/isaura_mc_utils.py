import numpy as np
import pandas as pd

from invisible_cities.evm  import event_model       as evm
from invisible_cities.reco import paolina_functions as plf
from typing                import Callable

from invisible_cities.types.ic_types       import xy


def track_blob_info_creator_extractor(vox_size : [float, float, float], energy_type : evm.HitEnergy, strict_vox_size : bool, energy_threshold : float, min_voxels : int, blob_radius : float) -> Callable:
    """ Wrapper of extract_track_blob_info"""
    def create_extract_track_blob_info(hitc):
        """This function extract relevant info about the tracks and blobs, as well as assigning new field of energy, track_id etc to the HitCollection object (NOTE: we don't want to erase any hits, just redifine some attributes. If we need to cut away some hits to apply paolina functions, it has to be on the copy of the original hits)"""
        voxels     = plf.voxelize_hits(hitc.hits, vox_size, strict_vox_size, energy_type)
        mod_voxels = voxels if energy_threshold == 0. else plf.drop_end_point_voxels(voxels, energy_threshold, min_voxels)[0]
        tracks     = plf.make_track_graphs(mod_voxels)
        vox_size_x = voxels[0].size[0]
        vox_size_y = voxels[0].size[1]
        vox_size_z = voxels[0].size[2]
        #sort tracks in energy
        tracks     = sorted(tracks, key = lambda t: sum([vox.Ehits for vox in t.nodes()]), reverse = True)
        track_hits = []
        df = pd.DataFrame(columns=['event', 'trackID', 'energy', 'length', 'numb_of_voxels',
                                   'numb_of_hits', 'numb_of_tracks', 'x_min', 'y_min', 'z_min',
                                   'x_max', 'y_max', 'z_max', 'r_max', 'x_ave', 'y_ave', 'z_ave',
                                   'extreme1_x', 'extreme1_y', 'extreme1_z',
                                   'extreme2_x', 'extreme2_y', 'extreme2_z',
                                   'blob1_x', 'blob1_y', 'blob1_z',
                                   'blob2_x', 'blob2_y', 'blob2_z',
                                   'eblob1', 'eblob2', 'ovlp_blob_energy',
                                   'vox_size_x', 'vox_size_y', 'vox_size_z'])
        for c, t in enumerate(tracks, 0):
            tID = c
            energy = sum([vox.Ehits for vox in t.nodes()])
            length = plf.length(t)
            numb_of_hits = len([h for vox in t.nodes() for h in vox.hits])
            numb_of_voxels = len(t.nodes())
            numb_of_tracks = len(tracks   )
            min_x = min([h.X for v in t.nodes() for h in v.hits])
            max_x = max([h.X for v in t.nodes() for h in v.hits])
            min_y = min([h.Y for v in t.nodes() for h in v.hits])
            max_y = max([h.Y for v in t.nodes() for h in v.hits])
            min_z = min([h.Z for v in t.nodes() for h in v.hits])
            max_z = max([h.Z for v in t.nodes() for h in v.hits])
            max_r = max([np.sqrt(h.X*h.X + h.Y*h.Y) for v in t.nodes() for h in v.hits])
            pos = [h.pos for v in t.nodes() for h in v.hits]
            e   = [getattr(h, energy_type.value) for v in t.nodes() for h in v.hits]
            ave_pos = np.average(pos, weights=e, axis=0)
            extr1, extr2 = plf.find_extrema(t)
            extr1_pos = extr1.XYZ
            extr2_pos = extr2.XYZ
            blob_pos1, blob_pos2 = plf.blob_centres(t, blob_radius)
            e_blob1, e_blob2, hits_blob1, hits_blob2 = plf.blob_energies_and_hits(t, blob_radius)
            overlap = False
            overlap = sum([h.Ec for h in set(hits_blob1).intersection(hits_blob2)])
            list_of_vars = [hitc.event, tID, energy, length, numb_of_voxels, numb_of_hits, numb_of_tracks, min_x, min_y, min_z, max_x, max_y, max_z, max_r, ave_pos[0], ave_pos[1], ave_pos[2], extr1_pos[0], extr1_pos[1], extr1_pos[2], extr2_pos[0], extr2_pos[1], extr2_pos[2], blob_pos1[0], blob_pos1[1], blob_pos1[2], blob_pos2[0], blob_pos2[1], blob_pos2[2], e_blob1, e_blob2, overlap, vox_size_x, vox_size_y, vox_size_z]
            df.loc[c] = list_of_vars
            try:
                types_dict
            except NameError:
                types_dict = dict(zip(df.columns, [type(x) for x in list_of_vars]))
            for vox in t.nodes():
                for hit in vox.hits:
                    hit.track_id = tID
                    track_hits.append(hit)
        track_hitc = evm.HitCollection(hitc.event, hitc.time)
        track_hitc.hits = track_hits
        #change dtype of columns to match type of variables
        df = df.apply(lambda x : x.astype(types_dict[x.name]))
        return df, mod_voxels, track_hitc
    return create_extract_track_blob_info


def convert_df_to_hits(df):
    return [evm.Hit(0, evm.Cluster(0, xy(h.x,h.y), xy(0,0), 0), h.z, h.energy, xy(0, 0))
            for h in df.itertuples(index=False)]