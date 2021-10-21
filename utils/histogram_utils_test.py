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
    img = container_creator(total_size, steps)
    mccoors, mcenes, ids = get_mchits_info(nevent, df)
    
    voxel_ener, bins = mcimg(img, mccoors, mcenes, steps = steps, x0 = x0)
    
    assert voxel_ener.shape == img.shape
    assert sum(voxel_ener.flatten()) == sum(mcenes)
