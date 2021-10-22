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


def mcimg(img, mccoors, mcenes, steps = None, x0 = None):
    '''
    This function creates a D-dimensional histogram weighted with the energies of particle hits.
    Thus, it voxelizes the space with certain voxel size and each voxel will contain the energy of
    the hits that fall inside.
    
    Args:
        img: NUMPYARRAY
    Array with the shape of the image (i.e. the full detector space). Together with steps and x0 will
    create the desired bins for the histogram.
    
        mccoors: NUMPYARRAY
    Coordinates of the particle hits. Having N hits, this sould be shaped as (N, D).
    
        mcenes: NUMPYARRAY
    Energies of the particle hits. Having N hits, hits should be shaped as (N,).
    
        steps: TUPLE (default = None)
    Desired distance between bins (i.e. bin size). The tuple size has to match img ndim.
        
        x0: TUPLE (default = None)
    Desired lower value for the bins. The tuple size has to match img ndim.
    
    RETURN:
        mcimg: NUMPYARRAY
    D-dimensional histogram with the energy counting in the appropriate bin.
    
        bins: LIST OF ARRAYS
    D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.
    It matches the bins local variable.


    '''
    bins  = bin_creator(img, steps, x0)
    mcimg, _  = np.histogramdd(mccoors, bins, weights = mcenes)
    
    return mcimg, bins
