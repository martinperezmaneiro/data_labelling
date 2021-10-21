import matplotlib as mpl
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D

def plot_3d_histo_hits(hist, bins, cmap = mpl.cm.jet):
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
    cb.set_label('Energy')
    plt.show()


def plot_3d_histo_vox(hist, bins, th=0, square_voxels = False, edgecolor=None, cmap=mpl.cm.jet):
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
    cb.set_label('Energy')

    plt.show()
