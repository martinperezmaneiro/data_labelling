import numpy  as np
import pandas as pd

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


def get_mchits_info(nevent, df, identifyer = 'particle_id', binclass = False):
    '''
    Gets the N coordinates, the energy and the particle_id for MC hits.
    
    Args: 
        nevent: INT
    Number that matches event_id in the data frame
    
        df: DATAFRAME
    Dataframe with the MC hits of a file.

        identifyer: STR
    Name of the df column that we want to take as id/label for the hits.

        binclass: BOOL
    Used to decide whether we want the binclass information in the output or we dont.
    
    RETURNS:
        mccoors: NUMPYARRAY
    Coordinates of the hits. (N, D)
    
        eners: NUMPYARRAY
    Energy of each hit. (N,)
    
        ids: NUMPYARRAY
    Particle identifyer of each hit. (N,)

        binnum: INT
    Optionally, if binclass = True, the function will return an identificator
    of the event, whether it is signal or background.
    
    '''
    event = df.loc[df['event_id'] == nevent]
    xhits, yhits, zhits = event['x'], event['y'], event['z']
    
    mccoors  = np.array([xhits, yhits, zhits]).T 
    eners    = np.array(event['energy'])
    ids      = np.array(event[identifyer])
    if binclass == True:
        binnum = np.array(event['binclass'])[0]
        return mccoors, eners, ids, binnum
    else:
        return mccoors, eners, ids


def histog_to_coord(event_id, id_hist, ener_hist, ratio_hist, bins, binnum = None):
    '''
    Takes the histogram (i.e. any voxelization) and returns an array of the voxel coordinates, their energies and
    ratio energies, and their feature
    (the feature can be things like particle ID or segmentation labels).
    
    Args:
        event_id: INT
    Number of the event.

        id_hist: NUMPYARRAY
    D-dimensional histogram with an ID for each voxel.
    
        ener_hist: NUMPYARRAY
    D-dimensional histogram with the energy for each voxel.
    
        ratio_hist: NUMPYARRAY
    D-dimensional histogram with the ratio of the energy of the most important particle to the total energy, per voxel.
    
        bins: LIST OF ARRAYS
    D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.
    
        binum: INT
    Number to identificate the type of event (signal or background)
    
    RETURN:
        df: DATAFRAME
    Coordinates of the nonzero voxels, their features and binclass, with the structure (D-coords, eners,
    ratio, features, binclass). The df has the length of the number of nonzero elements in the histogram.
    '''

    column_names  = ['x', 'y', 'z', 'ener', 'ratio', 'segclass']
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

    if binnum != None:
        column_names.append('binclass')
        coord.append(binnum)
    
    data = {}
    data['event_id'] = event_id
    for col, value in zip(column_names, coord):
        data[col] = value
    
    df = pd.DataFrame(data)
    return df


def calculate_track_distances(tracks_info, hits_label):
    '''
    Function to use inside the add_segclass function (because it has the tracks info),
    after being already assigned the seglabels. If we wanted to do this before those are asigned, 
    our output would be named hits_part inside the add_segclass function.
    
    Args:
        tracks_info: DATAFRAME
    Contains the particle information (ids, name, creator process, all its energy...). It is
    a dataframe created in the add_segclass function.
        
        hits_label: DATAFRAME
    Contains the hits information (ids, name, energy, segclass...). It is the final step inside the
    add_segclass function (the argument takes the last hits_label in the function). It can be just hits_part,
    but in our case we will use it with more information.
        
    RETURNS:
        hits_dist: DATAFRAME
    Same dataframe as hits_label with a new column: the hits distances for the track hits.
    
    '''
    #Delete hits with BUFFER label (otherwise some hits are duplicated)
    hits_label = hits_label.loc[hits_label.label == 'ACTIVE'] 
    
    #Reordes hits with ascendent ID
    hits_label = hits_label.sort_values(['event_id', 'particle_id', 'hit_id'], ascending=[True, True, True])
    
    #Stick the hits_label info to the tracks info to have the hits of the tracks (doesn't have to be hits_label,
    #can be mchits but in the end we will add to the DF the distances data, so we want the DF with the segclass
    #already, as it is used at the end of the add_segclass function)
    hits_tracks = tracks_info[['event_id', 
                               'particle_id',
                               'creator_proc']].merge(hits_label[['event_id', 
                                                                'particle_id', 
                                                                'hit_id', 
                                                                'x', 'y', 'z', 
                                                                'energy']], 
                                                                 on = ['event_id', 'particle_id'])
    
    #Add next row coordinates to new columns to compare, without the first hits
    hits_tracks[['x1', 'y1', 'z1']] = hits_tracks[['x', 'y', 'z']].shift(-1)
    
    #Compute the distance for each hit with the previous one, except from the first hits
    hits_tracks['dist_hits'] = np.linalg.norm(
                                                hits_tracks[['x', 'y', 'z']].values 
                                                - hits_tracks[['x1', 'y1', 'z1']].values, axis=1)
    
    hits_tracks = hits_tracks.assign(cumdist = hits_tracks.groupby(['event_id', 'particle_id']).dist_hits.cumsum())
    
    #Merge the particle info with the new distance info for the hits in the tracks
    hits_dist = hits_label.merge(hits_tracks[['event_id', 'particle_id', 'hit_id', 'dist_hits', 'cumdist']],
                                   how = 'outer',
                                  on = ['event_id', 'particle_id', 'hit_id'])
    hits_dist.dist_hits = hits_dist.dist_hits.fillna(0)
    hits_dist.cumdist   = hits_dist.cumdist.fillna(0)
    return hits_dist
