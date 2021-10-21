import numpy as np
from .labelling_utils import *
from .histogram_utils import container_creator
from .data_io         import get_mchits_info

def labelling_test(nevent, df, total_size, steps = None, x0 = None):
    img = container_creator(total_size, steps)
    mccoors, mcenes, hits_id = get_mchits_info(nevent, df)
    
    voxel_id, voxel_ener, bins = labelling(img, mccoors, mcenes, hits_id, steps = steps, x0 = x0)
    
    id_nonzero   = voxel_id.nonzero()
    ener_nonzero = voxel_ener.nonzero()
    
    voxel_id_unique = np.unique(voxel_id)
    voxel_id_unique = voxel_id_unique[voxel_id_unique != 0]
    id_unique       = np.unique(hits_id)
        
    assert voxel_id.shape == voxel_ener.shape == img.shape
    assert len(id_nonzero) == len(ener_nonzero)
    assert len(id_nonzero[0]) == len(ener_nonzero[0])
    assert all(np.isin(voxel_id_unique, id_unique))
