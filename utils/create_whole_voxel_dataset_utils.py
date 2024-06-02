import numpy as np
import pandas as pd


def get_extremes(labelhits):
    bincl = labelhits.binclass.unique()
    assert len(bincl) == 1, 'There are hits of background and signal in the same dataset'
    labelhits['segclass'] = labelhits['segclass'].replace(3, 2)

    track_hits = labelhits[labelhits.segclass == 2]

    #I use ext2 as the start of the track because for bkg it will always be the blob2; for signal
    if bincl == 0:
        ext2 = track_hits.groupby(['dataset_id', 'particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmin()])[['x', 'y', 'z']].values
        ext1 = track_hits.groupby(['dataset_id', 'particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmax()])[['x', 'y', 'z']].values

    if bincl == 1:
        track_ends  = track_hits.groupby(['dataset_id', 'particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmax()])
        ext2 = track_ends[track_ends.particle_id == 1][['x', 'y', 'z']].values
        ext1 = track_ends[track_ends.particle_id == 2][['x', 'y', 'z']].values
    
    return ext1, ext2

def get_bins(*args, min_size = (-500, -500, 0), max_size = (500, 500, 1300), nbins = (501, 501, 651)):
    bins_x = np.linspace(min_size[0], max_size[0], nbins[0])
    bins_y = np.linspace(min_size[1], max_size[1], nbins[1])
    bins_z = np.linspace(min_size[2], max_size[2], nbins[2])
    bins = (bins_x, bins_y, bins_z)
    return bins

def get_ext_flag(f, min_size = (-500, -500, 0), max_size = (500, 500, 1300), nbins = (501, 501, 651)):
    hits = pd.read_hdf(f, 'DATASET/MCHits')
    ext1, ext2 = get_extremes(hits)
    ext_df = pd.DataFrame({'dataset_id':hits.dataset_id.unique(), 'ext1':list(ext1), 'ext2':list(ext2)})

    bins = get_bins(min_size = min_size, max_size = max_size, nbins = nbins)
    voxelizer = lambda x: [np.histogram(x[i], bins[i])[0].nonzero()[0][0] for i in range(len(x))]
    ext_df.ext1 = ext_df.ext1.apply(voxelizer)
    ext_df.ext2 = ext_df.ext2.apply(voxelizer)
    ext_df = pd.melt(ext_df, id_vars='dataset_id', value_vars = ['ext1', 'ext2'], var_name = 'ext', value_name='voxel')#.sorted(['dataset_id'])
    ext_df[['x', 'y', 'z']] = ext_df.voxel.apply(pd.Series)
    ext_df.ext = ext_df.ext.apply(lambda x: int(x[-1]) if type(x) == str else int(x))
    return ext_df

def create_dataset_df(voxels, filenum, cols = ['dataset_id', 'x', 'y', 'z', 'ener', 'binclass', 'segclass', 'cloud', 'nhits', 'ext'], rename = {'dataset_id':'event', 'ener':'E', 'cloud':'track_id'}):
    df = voxels[cols].rename(columns = rename)
    df.insert(0, 'file_id', filenum)
    df['track_id'] = df['track_id'].apply(lambda x: int(x.split('_')[-1]))
    df['segclass'] = df['segclass'].apply(lambda x: int(x - 1))
    # to avoid the different int classes
    df['nhits'] = df['nhits'].astype('int16')
    return df

def create_dataset_file(file_dict, basedir, min_size = (-500, -500, 0), max_size = (500, 500, 1300), nbins = (501, 501, 651)):
    for name in file_dict:
        print(name)
        savedir = basedir.format(name, name.replace('/', '_'))
        for f in file_dict[name]:
            filenum = int(f.split('/')[-1].split('_')[-2])
            vox = pd.read_hdf(f, 'DATASET/MCVoxels')
            ext_df = get_ext_flag(f, min_size = min_size, max_size = max_size, nbins = nbins)
            vox_ext = vox.merge(ext_df.drop('voxel', axis = 1), on = ['dataset_id', 'x', 'y', 'z'], how = 'outer').fillna(0)
            vox_ext.ext = vox_ext.ext.astype(int)

            df = create_dataset_df(vox_ext, filenum)
            df.to_hdf(savedir, 'voxels', append = True)

        pd.read_hdf(f, 'DATASET/BinsInfo').to_hdf(savedir, 'bins')