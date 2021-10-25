import numpy as np
from .labelling_utils import *
from .histogram_utils import container_creator
from .data_io         import get_mchits_info

def voxel_labelling_test(nevent, df, total_size, steps = None, x0 = None):
    img  = container_creator(total_size, steps)
    bins = bin_creator(img, steps = steps, x0 = x0)
    mccoors, mcenes, hits_id = get_mchits_info(nevent, df)
    
    voxel_id, voxel_ener, voxel_portion = labelling(img, mccoors, mcenes, hits_id, bins)
    
    id_nonzero   = voxel_id.nonzero()
    ener_nonzero = voxel_ener.nonzero()
    port_nonzero = voxel_portion.nonzero()
    
    voxel_id_unique = np.unique(voxel_id)
    voxel_id_unique = voxel_id_unique[voxel_id_unique != 0]
    id_unique       = np.unique(hits_id)
        
    assert voxel_id.shape == voxel_ener.shape == voxel_portion.shape == img.shape
    assert len(id_nonzero) == len(ener_nonzero) == len(port_nonzero)
    assert len(id_nonzero[0]) == len(ener_nonzero[0]) == len(port_nonzero[0])
    assert voxel_portion.max() <= 1. 
    assert all(np.isin(voxel_id_unique, id_unique))
    for i in range(len(ener_nonzero)):
        assert (ener_nonzero[i] == id_nonzero[i]).all() #check if they are the same nonzero coordinates
        assert (ener_nonzero[i] == port_nonzero[i]).all()
