import numpy as np
import pandas as pd

# def add_hits_ext_label(hits, coords = ['x', 'y', 'z']):
#     '''
#     Uses labelled MC hits to find the extremes 
#     (doesn't need directly the particle tables, just labelled hits)
#     '''
#     hits_ = hits.copy()
#     bincl = hits_.binclass.unique()
#     assert len(bincl) == 1, 'There are hits of background and signal in the same dataset'
#     hits_['segclass'] = hits_['segclass'].replace(3, 2)

#     track_hits = hits_[hits_.segclass == 2]

#     grouped_hits = track_hits.groupby(['dataset_id', 'particle_id'])
#     #I use ext2 as the start of the track because for bkg it will always be the blob2; for signal
#     if bincl == 0:
#         ext2 = grouped_hits.apply(lambda x: x.loc[x['hit_id'].idxmin()])[coords]
#         ext2['ext'] = 2
#         ext1 = grouped_hits.apply(lambda x: x.loc[x['hit_id'].idxmax()])[coords]
#         ext1['ext'] = 1

#     if bincl == 1:
#         track_ends  = grouped_hits.apply(lambda x: x.loc[x['hit_id'].idxmax()]).reset_index(drop = True)
#         track_ends['row_num'] = track_ends.groupby('dataset_id').cumcount()
#         ext2 = track_ends[track_ends.row_num == 0][coords]
#         ext2['ext'] = 2
#         ext1 = track_ends[track_ends.row_num == 1][coords]
#         ext1['ext'] = 1
    
#     extremes = pd.concat([ext1, ext2]).reset_index()
#     hits = hits.merge(extremes, how='left').fillna(0)
#     hits['ext'] = hits['ext'].astype(int)
#     return hits

def add_ext_label(mchits, tracks_sig, tracks_bkg):
    '''
    Uses labelled MC hits to find the extremes.
    It is used inside the add_segclass label function, because it needs
    the separated sig/bkg tracks info.
    It gives them a label, called 'ext', which can be:
        1 - end   of background track, random for signal track
        2 - start of background track, random for signal track (for similarity with 'blob2')

    '''
    hits_sig = pd.merge(mchits, tracks_sig)
    track_ext_sig = hits_sig.groupby(['event_id', 'particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmax()]).reset_index(drop = True)
    track_ext_sig['ext'] = track_ext_sig.groupby('event_id').cumcount() + 1

    hits_bkg = pd.merge(mchits, tracks_bkg)
    track_start_bkg = hits_bkg.groupby(['event_id', 'particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmin()]).reset_index(drop=True)
    track_end_bkg   = hits_bkg.groupby(['event_id', 'particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmax()]).reset_index(drop=True)
    track_start_bkg['ext'] = 2
    track_end_bkg['ext']   = 1

    track_ext_bkg = pd.concat([track_start_bkg, track_end_bkg])
    track_ext = pd.concat([track_ext_bkg, track_ext_sig])
    return track_ext

def add_vox_ext_label(labelled_hits, labelled_voxels, bins, id_name = 'event_id', coords = ['x', 'y', 'z']):
    '''
    Uses labelled hits with extreme label and adds the extreme label to the voxelized hits
    '''
    ext_vox = labelled_hits[labelled_hits.ext != 0][[id_name] + coords + ['ext']]
    
    ext_vox[coords[0]] = pd.cut(ext_vox[coords[0]], bins = bins[0], labels = False)
    ext_vox[coords[1]] = pd.cut(ext_vox[coords[1]], bins = bins[1], labels = False)
    ext_vox[coords[2]] = pd.cut(ext_vox[coords[2]], bins = bins[2], labels = False)
    
    voxels_ext = labelled_voxels.merge(ext_vox, how='left').fillna(0)
    voxels_ext['ext'] = voxels_ext['ext'].astype(int)
    return voxels_ext