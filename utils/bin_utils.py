import numpy as np

def bins_creator_sipm(datasipm, zmin, zmax, zbin):
    xsipm, ysipm = datasipm["X"], datasipm["Y"]

    xmin, xmax = xsipm.min(), xsipm.max()
    ymin, ymax = ysipm.min(), ysipm.max()

    xbin = round(xsipm.drop_duplicates().sort_values().diff().dropna().max(), 3)
    ybin = round(ysipm.drop_duplicates().sort_values().diff().dropna().max(), 3)

    min_ = (round(xmin - xbin / 2, 3), round(ymin - ybin / 2, 3), zmin)
    max_ = (round(xmax + xbin / 2, 3), round(ymax + ybin / 2, 3), zmax)
    size_ = (xbin, ybin, zbin)
    total_ = (round(xmax - xmin, 3), round(ymax - ymin, 3), zmax - zmin)
    
    bin_info = dict(min = min_, 
                    max = max_, 
                    size = size_, 
                    total = total_)
    return bin_info

def bins_creator_regular(min_, max_, size_):
    total_ = tuple(np.round(np.array(max_) - np.array(min_), 2))
    bin_info = dict(min = min_, max = max_, size = size_, total = total_)
    return bin_info

def create_bins(bin_info):
    bins = []
    nbins_ = []
    for mi, ma, si in zip(bin_info['min'], bin_info['max'], bin_info['size']):
        b = np.arange(mi, ma + si, si)
        bins.append(b)
        nbins_.append(len(b) - 1)
    bin_info['nbins'] = tuple(nbins_)
    return bins, bin_info