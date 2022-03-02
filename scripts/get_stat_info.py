'''
This script helps to easlily visualize the statistics info which were extracted from the 
voxel grouping (i.e., the different ways an event was labelled with respect to the realiy,
basically knowing how many of the eventshad a good blob counting
'''

import invisible_cities.io.dst_io as dio

file = '/Users/mperez/NEXT/data_labelling/examples/h5files/statistics_4mm_fit_corrected.h5'
stat_df = dio.load_dst(file, 'stat', 'stat')

nevent = sum(stat_df.nevent)
signal_nevent = sum(stat_df.signal_nevent)
nevent_bkg_lower_fail = sum(stat_df.nevent_bkg_lower_fail)
nevent_bkg_upper_fail = sum(stat_df.nevent_bkg_upper_fail)
nevent_sig_lower_fail = sum(stat_df.nevent_sig_lower_fail)
nevent_sig_upper_fail = sum(stat_df.nevent_sig_upper_fail)
total_fails = nevent_bkg_lower_fail + nevent_bkg_upper_fail + nevent_sig_lower_fail + nevent_sig_upper_fail

print('Out of {nevent} events, the {percentage:.3f}% were double scape'.format(nevent = nevent, percentage = signal_nevent * 100 / nevent))
print('####')
print('Out of {nevent} events, the {percentage:.3f}% did not match binclass with the counted blobs'.format(nevent = nevent, percentage = total_fails * 100 / nevent))
print('Out of {nevent} events, the {percentage:.3f}% were from the bkg with less than 1 blob'.format(nevent = nevent, percentage = nevent_bkg_lower_fail * 100 / nevent))
print('Out of {nevent} events, the {percentage:.3f}% were from the bkg with more than 1 blob'.format(nevent = nevent, percentage = nevent_bkg_upper_fail * 100 / nevent))
print('Out of {nevent} events, the {percentage:.3f}% were from the dsc with less than 2 blob'.format(nevent = nevent, percentage = nevent_sig_lower_fail * 100 / nevent))
print('Out of {nevent} events, the {percentage:.3f}% were from the dsc with more than 2 blob'.format(nevent = nevent, percentage = nevent_sig_upper_fail * 100 / nevent))
