import tables as tb
import pandas as pd
import numpy as np
import invisible_cities.io.dst_io as dio


def add_track_start_points(mchits, isaura_info):
    '''
    Function to extract from the MCHits DataFrame the starting point of a background track.

    Args:
        mchits: pd.DataFrame
    MC hits from the labelled file.

        isaura_info: pd.DataFrame
    Isaura track info from the labelled file.

    RETURNS:
        isaura_info_start: pd.DataFrame
    Contains the isaura info and the track start (just for the background tracks)
    '''

    track_start = mchits[(mchits.binclass == 0) & (mchits.segclass == 2) & (mchits.hit_id == 0)][['dataset_id', 'x', 'y', 'z']]

    track_start = track_start.rename({'x':'track_start_x', 'y':'track_start_y', 'z':'track_start_z'}, axis = 1)
    isaura_info_start = isaura_info.merge(track_start, on = 'dataset_id', how = 'outer')
    return isaura_info_start

def bin_transformer(x, y, z, bins_info):
    '''
    This function transforms the bins position into real coordinates.

    Args:
        x, y, z: ARRAY
    Each one is an array of positions of voxels.

        bins_info: pd.DataFrame
    Contains the information of the bins, stored in the labelled file.

    RETURNS:
        x, y, z: ARRAY
    Transformed positions.
    '''

    min_x, size_x = bins_info.min_x.values[0], bins_info.size_x.values[0]
    min_y, size_y = bins_info.min_y.values[0], bins_info.size_y.values[0]
    min_z, size_z = bins_info.min_z.values[0], bins_info.size_z.values[0]

    x = x * size_x + min_x
    y = y * size_y + min_y
    z = z * size_z + min_z

    return x, y, z

def get_centers(coords, energy):
    '''
    Args:
        coords: ARRAY
    Array with the 3 coordinates and shape (N, 3), being N the number of voxels.

        energy: ARRAY
    Array with the energy of each voxel.

    RETURNS:
        barycenter: ARRAY
    Contains the center weighted by energy.

        avg_center: ARRAY
    Contains the center.
    '''
    pondered_coords = [position * ener for position, ener in zip(coords, energy)]
    barycenter = sum(pondered_coords) / sum(energy)
    avg_center = sum(coords)/len(coords)

    return barycenter, avg_center

def bin_creator(bins_info):
    '''
    This function creates the bins for each coordinate from the BinsInfo table in our data.

    Args:
        bins_info: pd.DataFrame
    Information of the voxelization in the labelled file.

    RETURNS:
        bins_x, bins_y, bins_z: ARRAYS
    Binning in the three coordinates.

    '''

    min_x, max_x, nbins_x = bins_info.min_x.values[0], bins_info.max_x.values[0], int(bins_info.nbins_x.values[0])
    min_y, max_y, nbins_y = bins_info.min_y.values[0], bins_info.max_y.values[0], int(bins_info.nbins_y.values[0])
    min_z, max_z, nbins_z = bins_info.min_z.values[0], bins_info.max_z.values[0], int(bins_info.nbins_z.values[0])

    bins_x = np.linspace(min_x, max_x, nbins_x)
    bins_y = np.linspace(min_y, max_y, nbins_y)
    bins_z = np.linspace(min_z, max_z, nbins_z)

    return bins_x, bins_y, bins_z

def get_blob_centers(beersh_voxels, bins_info):
    '''
    For each blob element, we transform the voxels into coordinates (to take into account the different
    sizes in each axis), we search for the center and barycenter (weighted in energy) and then revoxelize.

    Args:
        beersh_voxels: pd.DataFrame
    Contains all the beersheba labelled voxels information.

        bins_info: pd.DataFrame
    Contains the voxelization information of the labelled events.

    RETURNS:
        blob_center_df: pd.DataFrame
    Contains per event and per element the barycenter and the normal center.

    '''
    blob_center_df = pd.DataFrame()
    blob_segclass_name = [i for i in beersh_voxels.elem.unique() if i.split('_')[0] == '3']
    for (dat_id, elem), elem_df in beersh_voxels.groupby(['dataset_id', 'elem']):
        if np.isin(elem, blob_segclass_name):
            binclass = elem_df.binclass.values[0]
            energy = elem_df.energy.values
            x, y, z = elem_df.xbin.values, elem_df.ybin.values, elem_df.zbin.values
            x, y, z = bin_transformer(x, y, z, bins_info)
            coords = np.array([x, y, z]).T

            barycenter, avg_center = get_centers(coords, energy)

            blob_center_df = blob_center_df.append({'dataset_id':dat_id,
                                                    'binclass':binclass,
                                                    'elem':elem,
                                                    'elem_ener': elem_df.elem_ener.values[0],
                                                    'elem_count': elem_df.elem_count.values[0],
                                                    'barycenter_x':barycenter[0],
                                                    'barycenter_y':barycenter[1],
                                                    'barycenter_z':barycenter[2],
                                                    'avg_center_x':avg_center[0],
                                                    'avg_center_y':avg_center[1],
                                                    'avg_center_z':avg_center[2]},
                                                    ignore_index=True)

    return blob_center_df

def distance_to_center(centers_df, center_name, isaura_blob_name, coords = ['x', 'y', 'z']):
    '''
    Computes the distance between two points in a dataframe with a certain name.

    Args:
        centers_df: pd.DataFrame
    Contains per event and track the desired points information.

        center_name: STR
    Base name for the kind of point we want to compare (either 'track_start_' or 'barycenter_')

        isaura_blob_name: STR
    Base name for the isaura blob we want to compare(either 'blob1_' or 'blob2_')

    RETURNS:
        dist: FLOAT
    Distance between the desired points.
    '''

    center_coords = np.array([centers_df[center_name + c].values[0] for c in coords])
    isau_blob_coords = np.array([centers_df[isaura_blob_name + c].values[0] for c in coords])
    dist = np.linalg.norm(center_coords - isau_blob_coords)
    return dist

def recalculate_barycenter(centers_df, isaura_blob_name):
    '''
    Takes several elements and recalculates their barycenter weighted with the element energy.

    Args:
        centers_df: pd.DataFrame
    Contains per event and track the desired points information.

        isaura_blob_name: STR
    Base name for the isaura blob we want to compare(either 'blob1_' or 'blob2_')

    RETURNS:
        centers_df: pd.DataFrame
    Contains the desired isaura blob to compare and the new regrouped barycenter.
    '''
    coor = np.array([centers_df.barycenter_x.values, centers_df.barycenter_y.values, centers_df.barycenter_z.values]).T
    barycenter, avg_center = get_centers(coor, centers_df.elem_ener.values)
    centers_df = centers_df[[isaura_blob_name + 'x',
                             isaura_blob_name + 'y',
                             isaura_blob_name + 'z']].drop_duplicates().assign(**{'barycenter_x':barycenter[0],
                                                                                  'barycenter_y':barycenter[1],
                                                                                  'barycenter_z':barycenter[2]})
    return centers_df

def distance_cases(isaura_info_blobs):
    '''
    Different situations to know which two points to compare in each case.
        If the event is bkg, the function takes all the blob elements and
        combines them into one center to compare with blob1, and the blob2
        is measured against the track start.

        If the event is signal, the first element is compared to blob1 and
        the second element (or the rest, if there are more than two elements)
        are compared to blob2; if there's only one blob, both blob1&blob2
        are compared to the same point (guessing that two elements grouped together)

    It is also computed the distance between blob1&2 and the oposite blob
    elements, just in case my energetic blob is the isaura's less energetic
    blob. After this output it will be chosen the best combination.

    The events where no blob element was found are not included here.

    Args:
        isaura_info_blobs: pd.DataFrame
    Contains the isaura main tracks information with all the track start + blob centers information
    
    RETURNS:
        dist_blob_df:
    Contains the distances between the label and isaura blobs (both combinations), and the coordinates
    to the labelling blobs
    '''
    dist_blob_df = pd.DataFrame()

    for (idx, bincl, elem_count), centers_df in isaura_info_blobs.groupby(['dataset_id', 'binclass', 'elem_count']):
        eblob1, eblob2 = centers_df.eblob1.values[0], centers_df.eblob2.values[0]
        centers_df_inv = centers_df.copy()

        if bincl == 0:
            label_blob2_x = centers_df.track_start_x.values[0]
            label_blob2_y = centers_df.track_start_y.values[0]
            label_blob2_z = centers_df.track_start_z.values[0]

            #For bkg, the blob2 is always the track start
            blob2_dist = distance_to_center(centers_df, 'track_start_', 'blob2_')
            #I also compute the distance with crossed blobs, just in case the tables had turned for
            #isaura; although this effect would be more relevant for the signal events
            blob1_dist_inv = distance_to_center(centers_df, 'track_start_', 'blob1_')

            if elem_count > 1:
                #If the bkg event happens to have more than one blob element, we recalculate the
                #barycenter with all the elements
                centers_df = recalculate_barycenter(centers_df, 'blob1_')
                centers_df_inv = recalculate_barycenter(centers_df_inv, 'blob2_')

            #Finally, we compute the distance between the barycenter and the blob1
            blob1_dist = distance_to_center(centers_df, 'barycenter_', 'blob1_')
            #And the inverse
            blob2_dist_inv = distance_to_center(centers_df_inv, 'barycenter_', 'blob2_')

            label_blob1_x = centers_df.barycenter_x.values[0]
            label_blob1_y = centers_df.barycenter_y.values[0]
            label_blob1_z = centers_df.barycenter_z.values[0]


        if bincl == 1:

            label_blob1_x = centers_df[centers_df.elem == '3_0'].barycenter_x.values[0]
            label_blob1_y = centers_df[centers_df.elem == '3_0'].barycenter_y.values[0]
            label_blob1_z = centers_df[centers_df.elem == '3_0'].barycenter_z.values[0]

            blob1_dist = distance_to_center(centers_df[centers_df.elem == '3_0'], 'barycenter_', 'blob1_')
            blob2_dist_inv = distance_to_center(centers_df[centers_df.elem == '3_0'], 'barycenter_', 'blob2_')

            if elem_count < 2:
                #Blob1 and blob2 will be referred to the same element
                blob2_dist = distance_to_center(centers_df, 'barycenter_', 'blob2_')
                blob1_dist_inv = distance_to_center(centers_df, 'barycenter_', 'blob1_')

                label_blob2_x, label_blob2_y, label_blob2_z = label_blob1_x, label_blob1_y, label_blob1_z

            if elem_count == 2:
                blob2_dist = distance_to_center(centers_df[centers_df.elem == '3_1'], 'barycenter_', 'blob2_')
                blob1_dist_inv = distance_to_center(centers_df[centers_df.elem == '3_1'], 'barycenter_', 'blob1_')

                label_blob2_x = centers_df[centers_df.elem == '3_1'].barycenter_x.values[0]
                label_blob2_y = centers_df[centers_df.elem == '3_1'].barycenter_y.values[0]
                label_blob2_z = centers_df[centers_df.elem == '3_1'].barycenter_z.values[0]

            if elem_count > 2:
                #We get rid of the 3_0 element as it will only contribute to the blob1
                centers_df = centers_df[centers_df.elem != '3_0']
                centers_df = recalculate_barycenter(centers_df, 'blob2_')
                blob2_dist = distance_to_center(centers_df, 'barycenter_', 'blob2_')

                centers_df_inv = recalculate_barycenter(centers_df_inv, 'blob1_')
                blob1_dist_inv = distance_to_center(centers_df_inv, 'barycenter_', 'blob1_')

                label_blob2_x = centers_df.barycenter_x.values[0]
                label_blob2_y = centers_df.barycenter_y.values[0]
                label_blob2_z = centers_df.barycenter_z.values[0]

        dist_blob_df = dist_blob_df.append({'dataset_id':idx,
                                            'binclass':bincl,
                                            'blob1_dist':blob1_dist,
                                            'blob2_dist':blob2_dist,
                                            'blob1_dist_inv':blob1_dist_inv,
                                            'blob2_dist_inv':blob2_dist_inv,
                                            'label_blob1_x':label_blob1_x,
                                            'label_blob1_y':label_blob1_y,
                                            'label_blob1_z':label_blob1_z,
                                            'label_blob2_x':label_blob2_x,
                                            'label_blob2_y':label_blob2_y,
                                            'label_blob2_z':label_blob2_z,
                                            'eblob1':eblob1,
                                            'eblob2':eblob2},
                                            ignore_index=True)

    return dist_blob_df

def take_best_dist_outcome(dist_blob_df):
    '''
    Sums the two barycenter-isaurablob distances and picks up the minimum of both

    Args:
        dist_blob_df: pd.DataFrame
    Contains per event the combinations of blob distances between isaura and labelling.

    RETURNS:
        dist_blob_df: pd.DataFrame
    Contains per event the minimum distance combination between the barycenter of
    labelled blobs and the isaura blobs. Also the binclass and the isaura blob energy.
    '''
    min_dist_mask = dist_blob_df.blob1_dist + dist_blob_df.blob2_dist < dist_blob_df.blob1_dist_inv + dist_blob_df.blob2_dist_inv
    normal_dist_df = dist_blob_df[min_dist_mask].drop(['blob1_dist_inv', 'blob2_dist_inv'], axis = 1)

    inv_dist_df = dist_blob_df[~min_dist_mask].drop(['blob1_dist', 'blob2_dist'], axis = 1)
    inv_dist_df = inv_dist_df.rename(columns={'blob1_dist_inv':'blob1_dist', 'blob2_dist_inv':'blob2_dist', 'eblob1':'eblob2', 'eblob2':'eblob1'})

    dist_blob_df = normal_dist_df.append(inv_dist_df).sort_values('dataset_id')

    return dist_blob_df

def get_dist_blob_isaura(mchits, beersh_voxels, isaura_info, bins_info):
    '''
    Searchs for the barycenters of the blob elements in an event and performs
    the whole distance comparison between the barycenters of the label blobs
    and the isaura blobs.

    Args:
        mchits: pd.DataFrame
    MC hits from the labelled file.

        beersh_voxels: pd.DataFrame
    Beersheba voxels from the labelled file.

        isaura_info: pd.DataFrame
    Isaura track info from the labelled file.

        bins_info: pd.DataFrame
    Voxel information from the labelled file.

    RETURNS:
        dist_blob_df: pd.DataFrame
    Contains per event the minimum distance combination between the barycenter of
    labelled blobs and the isaura blobs. Also the binclass and the isaura blob energy.

        isaura_info_blobs: pd.DataFrame
    Contains the isaura main track information with the barycenter and track start coordinates.

    '''

    isaura_info_start = add_track_start_points(mchits, isaura_info)
    blob_center_df = get_blob_centers(beersh_voxels, bins_info)

    isaura_info_blobs = isaura_info_start.merge(blob_center_df, on = 'dataset_id', how = 'outer')
    isaura_info_blobs = isaura_info_blobs[isaura_info_blobs.trackID == 0]

    dist_blob_df = distance_cases(isaura_info_blobs)
    dist_blob_df[['blob2_dist', 'eblob2']] = dist_blob_df.groupby('dataset_id')[['blob2_dist', 'eblob2']].fillna(method='bfill')
    dist_blob_df.drop_duplicates(inplace = True)
    dist_blob_df = take_best_dist_outcome(dist_blob_df)
    return dist_blob_df, isaura_info_blobs
