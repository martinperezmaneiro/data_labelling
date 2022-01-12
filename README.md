# data_labelling
This repository will contains the required functions to label the hits and subsequently voxelize the events and label those voxels.

The final function is label_file, in labelling/file_labelling.py: this function voxelizes and labels a whole beersheba file.
It runs the labelling_MC function, contained in labelling/MClabelling.py, which would voxelize and label the MC hits with three classes: other, track, blob. The criteria for the label choices is explained in utils/labelling_utils.py, in the add_segclass function.
Then runs labelling_beersheba, in labelling/beershebalabelling.py, using a neighbour labelling algorithm which follows the natural order to fill voxels with labels, starting from the MC labels. It is explained in utils/beersheba_labelling_utils.py within the count_neighbours and label_neighbours_ordered functions. I have only one method implemented, but other one was contemplated in the exmples/example_beersh_label_2.ipynb notebook, which is based on a kind of convergence method.

The scripts and templates folders contains the required files to perform the labelling in the CESGA machines.