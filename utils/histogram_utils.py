import numpy as np


def container_creator(total_size, voxel_size):
    '''
    Creates a container of the required size in order to get the bin_creator() function create exactly the needed bins
    
    Args:
        total_size: TUPLE or LIST-like
    Maximum size of the container in each axis.
    
        voxel_size: TUPLE or LIST-like
    Size of the voxel for each axis.
    
    RETURN:
        img: NUMPYARRAY
    Container with the desired shape.
    
    '''
    xmaxsize, ymaxsize, zmaxsize = total_size[0], total_size[1], total_size[2]
    
    if voxel_size == None:
        voxel_size = (1., 1., 1.)
        
    xvoxsize, yvoxsize, zvoxsize = voxel_size[0], voxel_size[1], voxel_size[2]
    
    img = np.array([[[0] * int(zmaxsize / zvoxsize)] * int(ymaxsize / yvoxsize)] * int(xmaxsize / xvoxsize))
    return img


def bin_creator(img, steps = None, x0 = None):
    '''
    This function creates the bins for an histogram given a certain space.
    The size of the bins is controled by the steps and the max value of the bins by the img shape.
    
    Args:
        img: NUMPYARRAY
    Its shape provides the maximum value of the bins.
        
        steps: TUPLE (default = None)
    Desired distance between bins (i.e. bin size). The tuple size has to match img ndim.
        
        x0: TUPLE
    Desired lower value for the bins. The tuple size has to match img ndim.
    
    RETURN:
        bins: LIST OF ARRAYS
    It returns a ndim long list, in which each element is an array for a spatial coordinate with the desired bins.
    
    '''
    ndim  = img.ndim
    shape = img.shape
    steps = np.ones(ndim)  if steps is None else steps
    x0s   = np.zeros(ndim) if x0    is None else x0

    ibins = [np.linspace(0, n, n + 1) for n in shape]
    bins  = [x0 + step * ibin for x0, step, ibin in zip(x0s, steps, ibins)]
    
    return bins


def mcimg(mccoors, mcenes, bins):
    '''
    This function creates a D-dimensional histogram weighted with the energies of particle hits.
    Thus, it voxelizes the space with certain voxel size and each voxel will contain the energy of
    the hits that fall inside.
    
    Args:
        mccoors: NUMPYARRAY
    Coordinates of the particle hits. Having N hits, this sould be shaped as (N, D).
    
        mcenes: NUMPYARRAY
    Energies of the particle hits. Having N hits, hits should be shaped as (N,).
    
        bins: LIST OF ARRAYS
    D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.
    It matches the bins local variable.
    
    RETURN:
        mcimg: NUMPYARRAY
    D-dimensional histogram with the energy counting in the appropriate bin.
    '''
    mcimg, _  = np.histogramdd(mccoors, bins, weights = mcenes)
    
    return mcimg


def closest_number_bins(innum, bin_array, step, ismin): #maybe add step as arg
    '''
    This function will round a number to its closest (but yet above it) number in an array. With above it, 
    it is meant that the original number would always fall inside the interval between the rounded number 
    in the array and 0 (this explanation is needed for negative numbers). 
    We want this to get the appropriate bin number depending on whether the number is the minimum or the 
    maximum coordinate in an event.    
    
    Args:
        innum: FLOAT
    Maximum or minimum coordinate for an axis for all the hits in an event.
    
        bin_array: NUMPYARRAY
    Bins for an axis.
    
        step: INT
    Size of the bins.
    
        ismin: BOOL
    True if the innum is the minimum coordinate of the event, False if it is the maximum.
    
    RETURN:
        outnum: FLOAT
    Desired number of the bin_array, closest to innum and containing it in an interval between outnum and 0.
    
    '''
    if innum < bin_array[0] or innum > bin_array[-1]:
        raise ValueError('Number out of bin boundaries')
        
    for i in range(len(bin_array) - 1): 
        if abs(bin_array[i] - innum) < step and abs(bin_array[i + 1] - innum) < step:
            if ismin == True:
                outnum = bin_array[i]
            else:
                outnum = bin_array[i + 1]     
    return outnum


def frame_reductor(hits_coord, detector_bins, loose):
    '''
    This function will reduce the frame of work from the detector size to the size of the event in order
    to handle smaller containor arrays.
    
    Args:
        hits_coord: NUMPYARRAY
    Coordinates of the hits of an event.
    
        detector_bins: NUMPYARRAY
    Bins for the detector.
    
        loose: INT
    Number of blank voxels to add around the filled voxels in the 3 spatial dimensions.
    
    RETURNS:
        reduced_frame: NUMPYARRAY
    Adjusted frame to the event hits.
    
        reduced_bins: NUMPYARRAY
    Adjusted bins to the event hits.
    
    '''
    steps = [abs(bins[0] - bins[1]) for bins in detector_bins]
    extremes = [[coor.min(), coor.max()] for coor in hits_coord.T]
    extreme_bins = [[closest_number_bins(ex[0], bins, step, True), closest_number_bins(ex[1], bins, step, False)] 
    for ex, bins, step in zip(extremes, detector_bins, steps)]
    
    x0 = [x[0] - loose * step for x, step in zip(extreme_bins, steps)]
    reduced_size = [abs(ex_bin[0] - ex_bin[1]) + 2 * loose * step 
                         for ex_bin, step in zip(extreme_bins, steps)]
    reduced_frame = container_creator(reduced_size, steps)
    reduced_bins = bin_creator(reduced_frame, steps = steps, x0 = x0)
    
    return reduced_frame, reduced_bins
