import numpy as np
from .histogram_utils import *

def labelling(img, mccoors, mcenes, hits_id, steps = None, x0 = None):
    '''
    This function creates a D-dimensional array that corresponds a voxelized space (we will call it histogram).
    The bins of this histogram will take the value of the ID hits that deposit more energy within them.
    
    i.e., in a voxel with several hits, the function will label the voxel as the kind of hit that layed more energy,
    regardless of the number of hits. For a 8 hit voxel:
     - Hit A kind: 3 hits with energies 2, 2, 4    ---> total = 8
     - Hit B kind: 1 hit  with energy   6          ---> total = 6
     - Hit C kind: 4 hits with energies 1, 1, 2, 1 ---> total = 5
    This voxel will be labelled as kind A.
    
    The IDs of the hits may be the kind of particle, or even the segmentation classes (track, blob, other...)
    Also, gives an histogram of the energies of the hits for each voxel, using mcimg function.
    
    Args:
        img: NUMPYARRAY
    Array with the shape of the image (i.e. the full detector space). Together with steps and x0 will
    create the desired bins for the histogram.
    
        mccoors: NUMPYARRAY
    Coordinates of the particle hits. Having N hits, this sould be shaped as (N, D).
    
        mcenes: NUMPYARRAY
    Energies of the particle hits. Having N hits, this should be shaped as (N,).
    
        hits_id: NUMPYARRAY
    IDs for each hit. They define the kind of voxeles we will have. Having N hits, this should be shaped as (N,).
    
        steps: TUPLE (default = None)
    Desired distance between bins (i.e. bin size). The tuple size has to match img ndim.
        
        x0: TUPLE (default = None)
    Desired lower value for the bins. The tuple size has to match img ndim.
    
    RETURN:
        mc_hit_id: NUMPYARRAY
    D-dimensional histogram with the IDs for the voxels.
    
        mc_hit_ener: NUMPYARRAY
    D-dimensional histogram with the energies of the voxels.

        mc_hit_portion: NUMPYARRAY
    D-dimensional histogram with the ratio of the energy of the most important particle to the total energy, per voxel.
    
        bins: LIST OF ARRAYS
    D-dim long list, in which each element is an array for a spatial coordinate with the desired bins.
    '''
    
    bins           = bin_creator(img, steps, x0) 
    mc_hit_id      = np.zeros(img.shape)   #array 3d a llenar con los identificadores
    mc_hit_portion = np.zeros(img.shape)   #array 3d a llenar con el porcentaje de energía de la total que se lleva la partícula más importante
    unique_hits    = np.unique(hits_id)    #lista de identificadores de los hits (identificador puede ser tipo de particula, label, etc...)
    
    mc_hit_ener, _ = mcimg(img, mccoors, mcenes, steps = steps, x0 = x0) #histograma de energías 

    #Bucle en los identificadores de los hits para hacer un histograma de energía por tipo de hit
    histograms, nonzero = [], []        #lista de histogramas y de sus coordenadas no nulas

    for hit_id in unique_hits:
        hit_id_mask = hits_id == hit_id                           #mascara de cada tipo de hit
        vox, _ = np.histogramdd(mccoors[hit_id_mask], 
                                bins, 
                                weights = mcenes[hit_id_mask]) #histograma de energia por tipo de hit
        histograms.append(vox)                                 #lista de histogramas
        nonzero.append(np.array(vox.nonzero()).T)              #lista con las coordenadas no nulas
    
    #Bucle recorriendo los voxeles no nulos para comparar el valor de cada histograma
    
    for nz in nonzero: #recorre cada tipo de hit con voxeles no nulos particulares (los hits tipo 1 cayeron en ciertos voxeles, los tipo 2 en otros etc...)
        for i in nz:   #aqui recorre cada coordenada de cada tipo de hit
            nonzero_coors = tuple(i) 
            
            if mc_hit_id[nonzero_coors] != 0: #si cierto voxel ya ha sido llenado (es decir, si su valor no es cero)
                continue                      #pasamos de volver a hacerle cosas
                
            #Bucle en los histogramas para ver cual tiene el valor más grande en cada voxel, 
            #revelándome así qué tipo de voxel es
            
            vox_eners = [] #contenedor de los valores de voxel para todos los histogramas
            for histo in histograms:
                vox_eners.append(histo[nonzero_coors]) 
                
            vox_eners = np.array(vox_eners)
            assert len(vox_eners) == len(unique_hits) 
            selected_id = vox_eners.argmax() #mira la posición del elemento mayor en vox_eners
            mc_hit_id[nonzero_coors] = unique_hits[selected_id]   #toma dicha posición de la lista unique_hits y se la asigna a la posición correspondiente en el array vacío 
            
            max_ener   = vox_eners[selected_id]     #energía de la partícula más importante en un voxel 
            total_ener = mc_hit_ener[nonzero_coors] #la energía total contenida en ese voxel
            mc_hit_portion[nonzero_coors] = max_ener / total_ener
            
    return mc_hit_id, mc_hit_ener, mc_hit_portion, bins
