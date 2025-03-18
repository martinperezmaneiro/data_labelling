import pandas as pd
import numpy as np

from utils.reco_labelling_utils import voxelize_reco, label_reco_event

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

    #reco hits voxelization
    reco_voxels = voxelize_reco(reco_dir, bins, labelled_vox = labelled_MC_voxels, group = group, table = table, column_names = column_names)

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
