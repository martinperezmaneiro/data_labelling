import pandas as pd
import numpy as np

from utils.histogram_utils           import bin_creator, container_creator
from utils.beersheba_labelling_utils import assign_nlabels, voxelize_beersh, merge_MC_beersh_voxels
#label_neighbours_ordered because it would be entered as a function imput

def labelling_beersheba(beersh_dir, total_size, voxel_size, start_bin, labelled_MC_voxels, label_neighbours_function, simple = True, relabel = True, fix_track_connection = False, Rmax = np.nan):
    '''
    Takes the beersheba file, voxelizes its hits and labels them with the help of the labelled MC voxels,
    the output of the labelling_MC function, depending on the chosen neighbour labelling method.

    Args:
        beersh_dir: STR
    Directory of the beersheba file.

        total_size: TUPLE
    Contains the max size of the detector.

        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate.

        start_bin: TUPLE
    Contains the first voxel position for each coordinate.

        labelled_MC_voxels: DATAFRAME
    Contains the MC data labelled voxels that will be the base of the labelling.

        label_neighbours_function: FUNCTION
    Selected function to perform the neighbour labelling (so I can easily change the method)

        simple: BOOL
    If True, in the voxelization we will only use hits energy information. Else, the voxelization would include
    the information of some feature (with its ratio), which in beersheba data it's just the npeak variable. I
    don't really know if this is an important information.

        relabel: BOOL
    If True, the merge_MC_beersh_voxels would try to include the external MC labelled voxels to some empty beersheba
    voxels, so we can benefit from this information. Else, this info will be lost and we would stick only to the
    true coincident voxels.

        fix_track_connection: STR
    Used to solve the beersheba track desconnection problem (temporary) by adding the MC track voxels.
    If 'track', only track MC voxels will be added. If 'all', all the MC voxels are added.

        Rmax: NaN or FLOAT
    Value to perform the fiducial cut of the hits. If NaN, the cut is not done.

    RETURNS:
        mc_beersh_voxels: DATAFRAME
    Contains all the beersheba labelled voxels. It has their positions, energies, segclass, binclass; ener and ratio
    values are also included but only the MC voxels have them
    '''

    detector_frame = container_creator(total_size, voxel_size)
    detector_bins  = bin_creator(detector_frame, steps = voxel_size, x0 = start_bin)

    nlabel_dict = assign_nlabels()

    #Beersheba hits voxelization
    beersh_voxels = voxelize_beersh(beersh_dir, total_size, voxel_size, start_bin, labelled_vox = labelled_MC_voxels, simple = simple, Rmax = Rmax)

    #Joining of the MC and beersheba voxels, and discrepancies correction
    mc_beersh_voxels = merge_MC_beersh_voxels(labelled_MC_voxels, beersh_voxels, relabel = relabel, fix_track_connection = fix_track_connection)

    for event_id, df in mc_beersh_voxels.groupby('event_id'):
        #if event_id % 50 == 0:
        #    print(event_id)

        event_neighbours_labelled = label_neighbours_function(df, detector_bins, voxel_size, start_bin, nlabel_dict)
        mc_beersh_voxels = mc_beersh_voxels.merge(event_neighbours_labelled.segclass,
                                                  left_index = True,
                                                  right_index = True,
                                                  how = 'outer')
        mc_beersh_voxels['segclass'] = mc_beersh_voxels['segclass_y'].fillna(mc_beersh_voxels['segclass_x'])
        mc_beersh_voxels = mc_beersh_voxels.drop(['segclass_x', 'segclass_y'], axis = 1)

        #Check if the labelling has sense (just check that all the new classes are consistent to the original ones)
        unique_seg = df.segclass.unique()[~np.isnan(df.segclass.unique())]
        for i in unique_seg:
            unique_seg = np.append(unique_seg, i + 3)
        unique_seg = np.append(unique_seg, 7)

        #We have to order them to coincide with the df bc after merging the order changes
        mc_beersh_voxels_ev = mc_beersh_voxels[mc_beersh_voxels.event_id == event_id].sort_values(['event_id', 'x', 'y', 'z'])
        assert (np.isin(mc_beersh_voxels_ev.segclass, unique_seg)).all()

        #Check that the merge was sucessful
        assert pd.to_numeric(mc_beersh_voxels_ev.segclass, downcast = 'integer').equals(event_neighbours_labelled.segclass)

    #Turn into an integer
    mc_beersh_voxels.segclass = pd.to_numeric(mc_beersh_voxels.segclass, downcast = 'integer')
    #Order again to avoid weird labelling
    mc_beersh_voxels = mc_beersh_voxels.sort_values(['event_id', 'x', 'y', 'z'])
    return mc_beersh_voxels
