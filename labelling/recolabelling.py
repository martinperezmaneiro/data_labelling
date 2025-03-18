import pandas as pd
import numpy as np

# from utils.histogram_utils           import bin_creator, container_creator
from utils.reco_labelling_utils import voxelize_reco, label_reco_event
# from utils.beersheba_labelling_utils import assign_nlabels, merge_mc_reco_voxels

#label_neighbours_ordered because it would be entered as a function imput

def labelling_reco(reco_dir, bins, labelled_MC_voxels, group = 'RECO', table = 'Events', column_names = ['event', 'X', 'Y', 'Z', 'Ec'], ghost_label = 0):
    '''
    Takes the reco file, voxelizes its hits and labels them with the help of the labelled MC voxels,
    the output of the labelling_MC function, depending on the chosen neighbour labelling method.

    Args:
        reco_dir: STR
    Directory of the reco file.

        bins: LIST
    Contains the binning in the 3 dimensions.

        labelled_MC_voxels: DATAFRAME
    Contains the MC data labelled voxels that will be the base of the labelling.

        group: STR
    Name of the group to label.

        table: STR
    Name of the hits table to label.

        column_names: LIST
    Name of the column names of the 
    RETURNS:
        labelled_reco_voxels: DATAFRAME
    Contains all the reco labelled voxels. It has their positions, energies, segclass, binclass; ener and ratio
    values are also included but only the MC voxels have them
    '''

    # img  = container_creator(total_size, voxel_size)
    # bins = bin_creator(img, steps = voxel_size, x0 = start_bin)

    # nlabel_dict = assign_nlabels()

    #reco hits voxelization
    reco_voxels = voxelize_reco(reco_dir, bins, labelled_vox = labelled_MC_voxels, group = group, table = table, column_names = column_names)

    # !!!!!!!!!!!!!!!!!!!! DEPRECATED !!!!!!!!!!!!!!!!!!!!!!!!
    # #Joining of the MC and reco voxels, and discrepancies correction
    # mc_reco_voxels = merge_mc_reco_voxels(labelled_MC_voxels, reco_voxels, relabel = relabel, fix_track_connection = fix_track_connection)
    # del reco_voxels

    # for event_id, df in mc_reco_voxels.groupby('event_id'):
    #     #if event_id % 50 == 0:
    #     #    print(event_id)

    #     event_neighbours_labelled = label_neighbours_function(df, detector_bins, voxel_size, start_bin, nlabel_dict)
    #     mc_reco_voxels = mc_reco_voxels.merge(event_neighbours_labelled.segclass,
    #                                               left_index = True,
    #                                               right_index = True,
    #                                               how = 'outer')
    #     mc_reco_voxels['segclass'] = mc_reco_voxels['segclass_y'].fillna(mc_reco_voxels['segclass_x'])
    #     mc_reco_voxels = mc_reco_voxels.drop(['segclass_x', 'segclass_y'], axis = 1)

    #     #Check if the labelling has sense (just check that all the new classes are consistent to the original ones)
    #     unique_seg = df.segclass.unique()[~np.isnan(df.segclass.unique())]
    #     for i in unique_seg:
    #         unique_seg = np.append(unique_seg, i + 3)
    #     unique_seg = np.append(unique_seg, 7)

    #     #We have to order them to coincide with the df bc after merging the order changes
    #     mc_reco_voxels_ev = mc_reco_voxels[mc_reco_voxels.event_id == event_id].sort_values(['event_id', 'x', 'y', 'z'])
    #     assert (np.isin(mc_reco_voxels_ev.segclass, unique_seg)).all()

    #     #Check that the merge was sucessful
    #     assert pd.to_numeric(mc_reco_voxels_ev.segclass, downcast = 'integer').equals(event_neighbours_labelled.segclass)
    #     del mc_reco_voxels_ev, event_neighbours_labelled

    # Define the 26 neighbor shifts
    neighbor_shifts = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

    # Label event by event
    labelled_reco_voxels = pd.DataFrame()
    for event_id, group in reco_voxels.groupby(column_names[0]):
        mc_ev   = labelled_MC_voxels[labelled_MC_voxels['event_id'] == event_id]
        reco_ev = group
        labeled_voxels_event = label_reco_event(mc_ev, reco_ev, neighbor_shifts, ghost_label=ghost_label)
        labelled_reco_voxels = pd.concat([labelled_reco_voxels, labeled_voxels_event], ignore_index=True)

    #Turn into an integer
    labelled_reco_voxels.segclass = pd.to_numeric(labelled_reco_voxels.segclass, downcast = 'integer')

    # Add ext label to reco voxels
    labelled_reco_voxels = labelled_reco_voxels.merge(labelled_MC_voxels[['x', 'y', 'z', 'extlabel', 'event_id']], on = ['x', 'y', 'z', 'event_id'], how='left')
    labelled_reco_voxels['extlabel'] = labelled_reco_voxels['extlabel'].fillna(0).astype(int)

    # Add decolabel to reco voxels
    mc_label_voxels_ = labelled_MC_voxels[['x', 'y', 'z', 'event_id']].copy()
    mc_label_voxels_['decolabel'] = 1
    labelled_reco_voxels = labelled_reco_voxels.merge(mc_label_voxels_, on = ['x', 'y', 'z', 'event_id'], how = 'left')
    labelled_reco_voxels['decolabel'] = labelled_reco_voxels['decolabel'].fillna(0).astype(int)

    #Order again to avoid weird labelling
    labelled_reco_voxels = labelled_reco_voxels.sort_values(['event_id', 'x', 'y', 'z'])
    return labelled_reco_voxels
