import numpy  as np
import pandas as pd

def add_elem_number(beersh_voxels):
    '''
    This function adds a column counting the number of total elements that are in an event.
    '''
    elem_df = beersh_voxels[['dataset_id', 'elem']].drop_duplicates()
    beersh_voxels = beersh_voxels.merge(elem_df.groupby('dataset_id').count(), on = ['dataset_id'])
    beersh_voxels.rename({'elem_x':'elem', 'elem_y':'event_elem_count'}, inplace = True, axis = 1)
    return beersh_voxels

def apply_fiducial_cut(isaura_info, beersh_voxels, rmax = 180, zrange = [20, 510]):
    '''
    This function uses the main track information to get the events that fit inside the fiducial cut.
    Then returns the beersheba voxels with just those events.

    '''
    events_to_preserve = isaura_info[(isaura_info.trackID == 0)        \
                                     & (isaura_info.r_max < rmax)      \
                                     & (isaura_info.z_min > zrange[0]) \
                                     & (isaura_info.z_max < zrange[1])].dataset_id.unique()
    beersh_voxels = beersh_voxels[np.isin(beersh_voxels.dataset_id, events_to_preserve)]
    return beersh_voxels

def get_ghost_other_clouds(beersh_voxels, ghostclass = 7, otherclass = [1, 4]):
    '''
    This function gets two dataframes with clouds that contain ONLY one kind of label, or ghost or other
    '''
    ghost_clouds = beersh_voxels[beersh_voxels.segclass == ghostclass][['dataset_id', 'elem', 'elem_ener', 'cloud', 'cloud_ener']].drop_duplicates()
    #eliminate the clouds that arent exclusively from a segclass by equaling the cloud and elem ener
    #need to round because some of them have a bit of discordance on the last decimals but not worrying
    ghost_clouds = ghost_clouds[round(ghost_clouds.elem_ener, 7) == round(ghost_clouds.cloud_ener, 7)]

    #take the ones that are not a main cloud (bc those are mistaken for sure i guess)
    other_clouds = beersh_voxels[np.isin(beersh_voxels.segclass, otherclass) & (beersh_voxels.cloud != 'c_0')][['dataset_id', 'elem', 'elem_ener', 'cloud', 'cloud_ener']].drop_duplicates()
    other_clouds = other_clouds[round(other_clouds.elem_ener, 7) == round(other_clouds.cloud_ener, 7)]

    return ghost_clouds, other_clouds

def get_secondary_clouds(beersh_voxels):
    '''
    Returns a dataframe with a line per kind of element in each cloud that is not the main cloud
    '''
    secondary_clouds = beersh_voxels[(beersh_voxels.cloud != 'c_0')][['dataset_id', 'binclass', 'cloud', 'segclass']]
    secondary_clouds.segclass = secondary_clouds.segclass.apply(lambda x: {1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:4}[x])
    secondary_clouds = secondary_clouds.drop_duplicates()
    return secondary_clouds

def get_segclass_count(beersh_voxels, segclass):
    '''
    Returns a dataframe with a line per kind of the selected segclass in each cloud
    '''
    segclass_counting = beersh_voxels[np.isin(beersh_voxels.segclass, segclass)][['dataset_id', 'binclass', 'segclass', 'cloud', 'elem_count']].drop_duplicates()
    segclass_counting.segclass = segclass_counting.segclass.apply(lambda x: {1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:4}[x])
    segclass_counting = segclass_counting.drop_duplicates()
    return segclass_counting

def get_separated_segclass(segclass_counting):
    '''
    This function takes the output for get segclass count and returns:

    segclass_counting: has the number of clouds for the selected segclass

    sep_segclass: events with physically separated clouds of a certain segclass

    total_elem_count: events with all separated clouds of a certain segclass (the events here
    that aren't inside sep_segclass are clouds with U-turn or interrupted by some other segclass voxels)

    interr_segclass: events with interrupted segclass groups by other kind of segclass
    '''
    assert len(segclass_counting.segclass.unique()) == 1, 'The DF includes more than one segclass'

    selected_segclass = segclass_counting.segclass.unique()[0]

    if   selected_segclass == 2: dct = {selected_segclass: 'track'}
    elif selected_segclass == 3: dct = {selected_segclass: 'blob'}
    else: print('The selected DF does not belong to any possible segclass')

    cloud_per_segname = 'cloud_per_' + dct[selected_segclass]

    separated_segclass = segclass_counting.merge(segclass_counting.groupby('dataset_id').size().rename(cloud_per_segname), on = ['dataset_id'])
    segclass_counting  = segclass_counting.drop('cloud', axis = 1).drop_duplicates()

    if dct[selected_segclass] == 'blob':
        #we should divide between sig and bkg for blobs bc they have different blob number
        bkg_sep_seg = separated_segclass[(separated_segclass.binclass == 0) &\
                                         (separated_segclass[cloud_per_segname] > 1)].drop('cloud', axis = 1).drop_duplicates()

        sig_sep_seg = separated_segclass[(separated_segclass.binclass == 1) &\
                                         (separated_segclass[cloud_per_segname] > 2)].drop('cloud', axis = 1).drop_duplicates()

        sep_segclass = pd.concat([bkg_sep_seg, sig_sep_seg])

        bkg_elem_count = segclass_counting[(segclass_counting.binclass == 0) &\
                                           (segclass_counting.elem_count > 1)]
        sig_elem_count = segclass_counting[(segclass_counting.binclass == 1) &\
                                           (segclass_counting.elem_count > 2)]

        total_elem_count = pd.concat([bkg_elem_count, sig_elem_count])

        interr_segclass = total_elem_count[~np.isin(total_elem_count.dataset_id, sep_segclass.dataset_id)]

    elif dct[selected_segclass] == 'track':
        #these are the physically sep same segclass clouds
        sep_segclass = separated_segclass[separated_segclass[cloud_per_segname] > 1].drop('cloud', axis = 1).drop_duplicates()

        #these are all the sep same segclass clouds; all of the events here that
        #are not physically sep, its because they have an u-turn or so
        total_elem_count = segclass_counting[segclass_counting.elem_count > 1]

        #we can extract those interrupted tracks then:
        interr_segclass = total_elem_count[~np.isin(total_elem_count.dataset_id, sep_segclass.dataset_id)]


    else:
        print('The selected segname does not belong to any possible segclass')

    return segclass_counting, sep_segclass, total_elem_count, interr_segclass


def get_ev_list_stats(labelled_beersheba, track_segclass = [2, 5], blob_segclass = [3, 6]):
    
    #blob/track out of the main cloud
    secondary_clouds = get_secondary_clouds(labelled_beersheba)
    blobtrack_out_evs = secondary_clouds[(np.isin(secondary_clouds.segclass, [2, 3]))].dataset_id.unique()
    
    #TRACKS
    track_counting = get_segclass_count(labelled_beersheba, track_segclass)
    track_counting, separated_tracks, total_tracks_count, interr_tracks = get_separated_segclass(track_counting)

    #physically separated tracks
    sep_track_evs = separated_tracks.dataset_id.unique()

    #interrupted tracks
    interr_track_evs = interr_tracks.dataset_id.unique()
    
    #BLOBS
    blob_counting = get_segclass_count(labelled_beersheba, blob_segclass)
    blob_counting, separated_blobs, total_blobs_count, interr_blobs = get_separated_segclass(blob_counting)

    ##physically separated blobs
    sep_blob_evs = separated_blobs.dataset_id.unique()

    ##interrupted blobs
    interr_blob_evs = interr_blobs.dataset_id.unique()

    ##good blob count
    blob_count_succ = labelled_beersheba[(labelled_beersheba.blob_success == True)][['dataset_id', 'binclass', 'nblob', 'blob_success']].drop_duplicates()
    blob_count_evs = blob_count_succ.dataset_id.unique()
    
    var_list = ['blobtrack_out', 'sep_track', 'interr_track', 'sep_blob', 'interr_blob', 'blob_count_succ']
    
    return [blobtrack_out_evs, sep_track_evs, interr_track_evs, sep_blob_evs, interr_blob_evs, blob_count_evs], var_list


def create_df_cloud_stats(i, file, voxel_df, track_segclass = [2, 5], blob_segclass  = [3, 6]):
    ''' 
    Final breaking tracks stats creator
    '''
    nevents_total = len(voxel_df.dataset_id.unique())
    secondary_clouds = get_secondary_clouds(voxel_df)
    #percentage of events with a track/blob element out of the main cloud
    blobtrack_out_evs  = secondary_clouds[(np.isin(secondary_clouds.segclass, [2, 3]))].dataset_id.unique()
    blobtrack_out_rate = len(blobtrack_out_evs) / nevents_total

    #track statistics
    track_counting = get_segclass_count(voxel_df, track_segclass)
    track_counting, separated_tracks, total_tracks_count, interr_tracks = get_separated_segclass(track_counting)
    ##physically separated tracks
    sep_track_evs = separated_tracks.dataset_id.unique()
    sep_track_evs_rate = len(sep_track_evs) /len(track_counting)
    ##interrupted tracks
    interr_track_evs = interr_tracks.dataset_id.unique()
    interr_track_evs_rate = len(interr_track_evs) / len(track_counting)

    #blob statistics
    blob_counting = get_segclass_count(voxel_df, blob_segclass)
    blob_counting, separated_blobs, total_blobs_count, interr_blobs = get_separated_segclass(blob_counting)
    ##physically separated blobs
    sep_blob_evs = separated_blobs.dataset_id.unique()
    sep_blob_evs_rate = len(sep_blob_evs) / len(blob_counting)
    ##interrupted blobs
    interr_blob_evs = interr_blobs.dataset_id.unique()
    interr_blob_evs_rate = len(interr_blob_evs) / len(blob_counting)
    ##bad blob count
    blob_count_false = voxel_df[(voxel_df.blob_success == False)][['dataset_id', 'binclass', 'nblob', 'blob_success']].drop_duplicates()
    blob_count_false_rate = len(blob_count_false) / nevents_total

    events_df = pd.DataFrame([{'filenumber':i,
                              'nevents':nevents_total,
                              'blobtrack_out': np.array(blobtrack_out_evs),
                              'sep_track':np.array(sep_track_evs),
                              'interr_track':np.array(interr_track_evs),
                              'sep_blob':np.array(sep_blob_evs),
                              'interr_blob':np.array(interr_blob_evs),
                              'bad_blob_count':np.array(blob_count_false)}])
    events_df['filename'] = file.split("/")[-1]

    rates_df  = pd.DataFrame([{'filenumber':i,
                                'nevents':nevents_total,
                                'blobtrack_out': blobtrack_out_rate,
                                'sep_track':sep_track_evs_rate,
                                'interr_track':interr_track_evs_rate,
                                'sep_blob':sep_blob_evs_rate,
                                'interr_blob':interr_blob_evs_rate,
                                'bad_blob_count':blob_count_false_rate}])
    rates_df['filename'] = file.split("/")[-1]

    return rates_df, events_df