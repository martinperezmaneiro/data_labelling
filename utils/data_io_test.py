import numpy as np
from .data_io import *
from .histogram_utils import container_creator
from .labelling_utils import labelling

def get_mchits_info_test(nevent, df):
    mccoors, eners, ids = get_mchits_info(nevent, df)
    
    assert mccoors.shape[0] == eners.shape[0] == ids.shape[0]
    assert eners.shape == ids.shape


def histog_to_coord_test(nevent, df, total_size, steps = None, x0 = None):
    img = container_creator(total_size, steps)
    mccoors, mcenes, hits_id = get_mchits_info(nevent, df)
    
    voxel_id, voxel_ener, bins = labelling(img, mccoors, mcenes, hits_id, steps = steps, x0 = x0)
    coords = histog_to_coord(voxel_id, voxel_ener, bins)
    
    xcoor, ycoor, zcoor, eners, ids = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], coords[:, 4]
    assert len(xcoor) == len(ycoor) == len(zcoor) == len(eners) == len(ids)
    assert all(np.isin(xcoor, bins[0]))
    assert all(np.isin(ycoor, bins[1]))
    assert all(np.isin(zcoor, bins[2]))
    assert all(np.isin(eners, voxel_ener))
    assert all(np.isin(ids, hits_id))
