import glob
from utils.create_whole_voxel_dataset_utils import create_dataset_file

basedir = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/'
savedir = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/{}/label/voxel_dataset_{}.h5'
pressures = ['1bar', '2bar', '5bar', '13bar']
data_type = ['0nubb', '1eroi']
min_size = (-1000, -1000, -1000) #(-500, -500, 0)
max_size = (1000, 1000, 1000) #(500, 500, 1300)
nbins    = (1001, 1001, 1001) #(501, 501, 651)


nexusfiles, labelfiles, graphfiles = {}, {}, {}

for dt in data_type:
    for p in pressures:
        nexusdir = basedir + '{p}/{dt}/nexus/*_{dt}.h5'.format(p = p, dt = dt)
        labeldir = basedir + '{p}/{dt}/label/prod/*_{dt}.h5'.format(p = p, dt = dt)
        graphdir = basedir + '{p}/{dt}/label/prod/*_{dt}_graph.h5'.format(p = p, dt = dt)

        nexusfiles[p + '/' + dt] = sorted(glob.glob(nexusdir), key = lambda x: int(x.split('/')[-1].split('_')[-2]))
        labelfiles[p + '/' + dt] = sorted(glob.glob(labeldir), key = lambda x: int(x.split('/')[-1].split('_')[-2]))
        graphfiles[p + '/' + dt] = sorted(glob.glob(graphdir), key = lambda x: int(x.split('/')[-1].split('_')[-3]))


create_dataset_file(labelfiles, savedir, min_size = min_size, max_size = max_size, nbins = nbins)