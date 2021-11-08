import numpy  as np
import pandas as pd

from utils.data_utils      import histog_to_coord
from utils.histogram_utils import container_creator, bin_creator
from utils.labelling_utils import add_hits_labels_MC, voxel_labelling_MC

from invisible_cities.io   import dst_io as dio

def labelling_MC(directory, total_size, voxel_size, start_bin, blob_ener_loss_th = None, blob_ener_th = None):
    '''
    Performs hit labelling (binclass and segclass), voxelization of the hits (gives us the energy 
    per voxel, adding up all the hits that fall inside a voxel) and voxel segclass labelling. 

    Args:
        directory: STR
    Contains the directory of a file with several events with Monte Carlo information.
    
        total_size: TUPLE 
    Contains the max size of the detector.
    
        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate.
    
        start_bin: TUPLE
    Contains the first voxel position for each coordinate.
        
        binclass: BOOL
    Used to decide whether we want the binclass information in the output or we dont.
    
        blob_ener_loss_th: FLOAT
    Energy loss percentage of total track energy for the last hits that establishes a threshold for the blob class.

        blob_ener_th: FLOAT
    Energy threshold for the last hits of a track to become blob class.
    
    RETURNS:
        voxelization_df: DATAFRAME
    It contains the positions, energies and labels for each voxel of each event in a single file.
    
    '''
    #Creo el frame del detector y obtengo sus bins
    img  = container_creator(total_size, voxel_size)
    bins = bin_creator(img, steps = voxel_size, x0 = start_bin)

    #Obtenemos la información de partíuclas y hits de un fichero en concreto
    mcpart = dio.load_dst(directory, 'MC', 'particles') 
    mchits = dio.load_dst(directory, 'MC', 'hits') 
    
    #Etiquetamos los hits 
    labelled_hits = add_hits_labels_MC(mchits, mcpart, blob_ener_loss_th = blob_ener_loss_th, blob_ener_th = blob_ener_th)
    
    #Creamos el df donde vamos a añadir la información de los voxeles etiquetados
    voxelization_df = pd.DataFrame()
    
    #Recorremos evento a evento el DF con los hits etiquetados para hacerle a cada uno su histograma y
    #finalmente extraer las coordenadas
    for event_id, event_hits in labelled_hits.groupby('event_id'):
        xhits, yhits, zhits = event_hits['x'], event_hits['y'], event_hits['z']
    
        mccoors  = np.array([xhits, yhits, zhits]).T 
        mcenes   = np.array(event_hits['energy'])
        labels   = np.array(event_hits['segclass'])
        binclass = np.array(event_hits['binclass'])[0]
        
        label_histo, ener_histo, ratio_histo = voxel_labelling_MC(img, mccoors, mcenes, labels, bins)
        
        voxelization_df = voxelization_df.append(histog_to_coord(event_id, label_histo, ener_histo, ratio_histo, bins, binnum = binclass))
    
    voxelization_df.reset_index()
    for colname in voxelization_df.columns:
        voxelization_df[colname] = pd.to_numeric(voxelization_df[colname], downcast = 'integer')
    
    return voxelization_df
