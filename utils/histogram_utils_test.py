import numpy as np
from .histogram_utils import *
from .data_io         import get_mchits_info

def container_creator_test(total_size, voxel_size):
    img = container_creator(total_size, voxel_size)
    
    for maxsize, voxsize, shape in zip(total_size, voxel_size, img.shape):
        assert int(maxsize/voxsize) == shape


def bins_test(img, steps = None, x0 = None):
    ndim  = img.ndim
    shape = img.shape
    
    bins = bin_creator(img, steps = steps, x0 = x0)
    
    assert len(bins) == ndim
    for length, bine, step, start in zip(shape, bins, steps, x0):
        assert len(bine) == length + 1 
        for n in bine:
            assert n == start
            start += step


def mcimg_test(nevent, df, total_size, steps = None, x0 = None):
    img  = container_creator(total_size, steps)
    bins = bin_creator(img, steps = steps, x0 = x0)
    mccoors, mcenes, ids = get_mchits_info(nevent, df)
    
    voxel_ener = mcimg(mccoors, mcenes, bins)
    
    assert voxel_ener.shape == img.shape
    assert sum(voxel_ener.flatten()) == sum(mcenes)


def closest_number_bins_test():
    bins = np.array([-5, 0, 5]) #mock array
    step = abs(bins[0]-bins[1])
    assert closest_number_bins(-1, bins, step, True)   == -5
    assert closest_number_bins(-1, bins, step, False)  == 0
    assert closest_number_bins(2.4, bins, step, True)  == 0
    assert closest_number_bins(2.4, bins, step, False) == 5



def frame_reductor_test(nevent, df, total_size, steps = None, x0 = None):
    img  = container_creator(total_size, steps)
    bins = bin_creator(img, steps = steps, x0 = x0)
    mccoors, mcenes, hits_id = get_mchits_info(nevent, df)
    
    reduced_img, reduced_bins = frame_reductor(mccoors, bins, 2)
    
    assert reduced_img.ndim == img.ndim
    for b, B in zip(reduced_bins, bins):
        assert all(np.isin(b, B))
