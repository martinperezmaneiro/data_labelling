import numpy as np

def get_event_id(cutnum, ev_num):
    '''
    Given a cut number (one of the identifiers of a file) and a event number (i.e. number of the 
    position of an event inside a file) it returns the equivalent number for the event_id column in the MC data
    
    Args:
        cutnum: INT or STRING
    Number in the filename.
    
        ev_num: INT
    Desired event position.
    
    RETURN:
        nevent: INT
    Corresponding event_id to the desired event.
    
    '''
    cutnum = int(cutnum)
    nevent = cutnum * 2 * 1000000 + cutnum + ev_num
    return nevent


def get_mchits_info(nevent, df, identifyer = 'particle_id'):
    '''
    Gets the N coordinates, the energy and the particle_id for MC hits.
    
    Args: 
        nevent: INT
    Number that matches event_id in the data frame
    
        df: DATAFRAME
    Dataframe with the MC hits of a file.
    
    RETURNS:
        mccoors: NUMPYARRAY
    Coordinates of the hits. (N, D)
    
        eners: NUMPYARRAY
    Energy of each hit. (N,)
    
        ids: NUMPYARRAY
    Particle identifyer of each hit. (N,)
    
    '''
    event = df.loc[df['event_id'] == nevent]
    xhits, yhits, zhits = event['x'], event['y'], event['z']
    
    mccoors = np.array([xhits, yhits, zhits]).T 
    eners   = np.array(event['energy'])
    ids     = np.array(event[identifyer])
    return mccoors, eners, ids


def histog_to_coord(id_hist, ener_hist, ratio_hist, bins):
    '''
    Takes the histogram (i.e. any voxelization) and returns an array of the voxel coordinates, their energies and
    ratio energies, and their feature
    (the feature can be things like particle ID or segmentation labels).
    
    Args:
        id_hist: NUMPYARRAY
    D-dimensional histogram with an ID for each voxel.
    
        ener_hist: NUMPYARRAY
    D-dimensional histogram with the energy for each voxel.
    
        ratio_hist: NUMPYARRAY
    D-dimensional histogram with the ratio of the energy of the most important particle to the total energy, per voxel.
    
        bins: LIST OF ARRAYS
    D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.
    
    RETURN:
        coord: NUMPYARRAY
    Coordinates of the nonzero voxels and their feature, with the structure (D-coords, eners, ratio, features).
    The array has the length of the number of nonzero elements in the histogram.

    '''
    ndim          = ener_hist.ndim
    ener_nonzero  = ener_hist.nonzero()
    id_nonzero    = id_hist.nonzero()
    ratio_nonzero = ratio_hist.nonzero()
    
    nonzero = id_nonzero
    coord   = []
    for i in range(ndim):
        coord.append(bins[i][nonzero[i]])

    coord.append(ener_hist[nonzero])
    coord.append(ratio_hist[nonzero])
    coord.append(id_hist[nonzero])
    
    coord = np.array(coord).T
    assert len(coord) == len(np.array(nonzero).T)
    return coord
