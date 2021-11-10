import numpy as np

import matplotlib        as mpl
import matplotlib.pyplot as plt

from   mpl_toolkits.mplot3d import Axes3D

def plot_3d_histo_hits(hist, bins, cmap = mpl.cm.jet, clabel = 'Energy'):
    '''
    This function plots the 3 dimension histogram created from the particles hits.
    Is a scatter of the bins central positions in order to appreciate all of them.
    
    Args:
        hist: NUMPYARRAY
    Histogram of the particle hits of an event (from mcimg function)
        bins: LIST OF ARRAYS
    D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.
    It will give us the correct positions of the hits. (from mcimg function)


    '''
    #Función para ver los puntos del histograma, siendo los colores la frecuencia de aparicion es decir, las veces
    #que un hit cayó en un bin
    nonzero = hist.nonzero() #non zero position of the values, i.e. the localization of the points to plot
    
    fig  = plt.figure(figsize=(15, 15), frameon=False)
    gs   = fig.add_gridspec(2, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=hist.min(), vmax=hist.max())

    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(hist[nonzero])) #aqui los valores del histo
    colors = np.rollaxis(colors, 0, 2)
    
    
    ax.scatter(bins[0][nonzero[0]], bins[1][nonzero[1]], bins[2][nonzero[2]], c=colors, marker='o')
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')

    
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    cb.set_label(clabel)
    plt.show()


def plot_3d_histo_vox(hist, bins, th=0, square_voxels = False, edgecolor=None, cmap=mpl.cm.jet, clabel = 'Energy'):
    '''
    This function plots the 3 dimension histogram created from the particles hits.
    This shows the voxels with the containing energy. Note that the axis labels dont represent 
    
    Args:
        hist: NUMPYARRAY
    Histogram of the particle hits of an event (from mcimg function)
    
        bins: LIST OF ARRAYS
    D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.
    It will give us the correct positions of the hits. (from mcimg function)
    
        th: FLOAT (default = 0)
    Threshold for the minimum value of the voxels.
    
        square_voxels: BOOL (default = False)
    If True, the plot will show square voxels because it scales the plot to be equal in each axis.
    '''
    
    nonzero = hist.nonzero()
    xmin, xmax = nonzero[0].min(), nonzero[0].max()
    ymin, ymax = nonzero[1].min(), nonzero[1].max()
    zmin, zmax = nonzero[2].min(), nonzero[2].max()

    nbinsX = int(np.ceil((xmax-xmin))) + 2 #esto marca el rango del plot que se verá (el +2 es para que quede 'holgado')
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2
    
    if square_voxels == True:
        nbinmax = max(nbinsX, nbinsY, nbinsZ) 
        nbinsX, nbinsY, nbinsZ = nbinmax, nbinmax, nbinmax
    
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([nonzero[0]-xmin+1, #esto como que hace que los voxeles se ploteen reseteando posiciones
                          nonzero[1]-ymin+1, #es decir, si la posicion mas baja del eje x es 15 y la más alta 20
                          nonzero[2]-zmin+1]) #esto hará que se pase a 1 y 5
    xarr[tuple(nonzeros)] = hist[nonzero]
    dim     = xarr.shape
    voxels  = xarr > th

    fig  = plt.figure(figsize=(15, 15), frameon=False)
    gs   = fig.add_gridspec(2, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=xarr.min(), vmax=xarr.max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors, edgecolor=edgecolor)
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')
    
    ax.set_xlabel('X ')
    ax.set_xticks(np.arange(nbinsX))
    ax.set_xticklabels([int(i) for i in np.unique(bins[0][nonzero[0]])], rotation = 45)
    ax.set_ylabel('Y ')
    ax.set_yticks(np.arange(nbinsY))
    ax.set_yticklabels([int(i) for i in np.unique(bins[1][nonzero[1]])], rotation = -45)
    ax.set_zlabel('Z ')
    ax.set_zticks(np.arange(nbinsZ))
    ax.set_zticklabels([int(i) for i in np.unique(bins[2][nonzero[2]])])
    cb.set_label(clabel)

    plt.show()


def plot_projections(hits, value='energy', coords = ['x', 'y', 'z'], cmap = mpl.cm.jet, th = 0):
    fig, axs = plt.subplots(nrows=1, ncols=3,
                                        figsize=(12, 6))
    coors_pairs = itertools.combinations(coords, 2)
    cmap.set_under('white')
    for i, coor_pair in enumerate(coors_pairs):
        sel = hits.groupby(list(coor_pair))[value].sum()
        ind0 = sel.index.get_level_values(coor_pair[0])
        ind1 = sel.index.get_level_values(coor_pair[1])
        newind0 = np.arange(ind0.min(), ind0.max()+1)
        newind1 = np.arange(ind1.min(), ind1.max()+1)
        xx, yy = np.meshgrid(newind0, newind1)
        newind = pd.Index(list(zip(xx.flatten(), yy.flatten())), name=tuple(coor_pair))
        sel = sel.reindex(newind,  fill_value=0).reset_index()
        sel = pd.pivot_table(sel, values=value, index=[coor_pair[0]],
                        columns=[coor_pair[1]], aggfunc=np.sum)
        #print((newind0.min(),newind0.max(), newind1.min(),  newind1.max()))
        axs[i].imshow(sel.T, origin='lower', vmin=th+np.finfo(float).eps, extent=(newind0.min(),newind0.max(), newind1.min(),  newind1.max()),
                      cmap=cmap, aspect='auto')
        axs[i].set_xlabel(coor_pair[0])
        axs[i].set_ylabel(coor_pair[1])
    fig.tight_layout()

    plt.show()

    
def plot_3d_vox(hits_digitized, value='energy', coords = ['x', 'y', 'z'], th=0, edgecolor='k', linewidth = .3, cmap=mpl.cm.jet, opacity = 1):

    xmin, xmax = hits_digitized[coords[0]].min(), hits_digitized[coords[0]].max()
    ymin, ymax = hits_digitized[coords[1]].min(), hits_digitized[coords[1]].max()
    zmin, zmax = hits_digitized[coords[2]].min(), hits_digitized[coords[2]].max()

    nbinsX = int(np.ceil((xmax-xmin))) + 2
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([hits_digitized[coords[0]].values-xmin+1,
                          hits_digitized[coords[1]].values-ymin+1,
                          hits_digitized[coords[2]].values-zmin+1])
    nonzeros = nonzeros.astype(int)
    xarr[tuple(nonzeros)] = hits_digitized[value].values
    dim     = xarr.shape
    voxels  = xarr > th

    fig  = plt.figure(figsize=(15, 15), frameon=False)
    gs   = fig.add_gridspec(2, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=xarr.min(), vmax=xarr.max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors * opacity, edgecolor=edgecolor, linewidth = linewidth)
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    cb.set_label (value)

    plt.show()

    
def plot_3d_hits(hits, value='energy', coords = ['x', 'y', 'z'], cmap = mpl.cm.jet):
    fig  = plt.figure(figsize=(15, 15), frameon=False)
    gs   = fig.add_gridspec(2, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=hits.loc[:, value].min(), vmax=hits.loc[:, value].max())

    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(hits.loc[:, value]))
    colors = np.rollaxis(colors, 0, 2)

    ax.scatter(hits[coords[0]], hits[coords[1]], hits[coords[2]], c=colors, marker='o')
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')


    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    cb.set_label (value)

    plt.show()
