import numpy  as np
import pandas as pd

from utils.data_utils      import histog_to_coord
from utils.histogram_utils import container_creator, bin_creator
from utils.labelling_utils import add_hits_labels_MC, voxel_labelling_MC, hit_data_cuts, add_small_blob_mask

from invisible_cities.io   import dst_io as dio

def labelling_MC(directory, total_size, voxel_size, start_bin, sig_creator = 'conv', blob_ener_loss_th = None, blob_ener_th = None, Rmax = np.nan, small_blob_th = 0.1):
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

        sig_creator: STR
    If 'conv', signal will be the double scape data.
    If 'none', signal will be the neutrinoless decay data.

        blob_ener_loss_th: FLOAT
    Energy loss percentage of total track energy for the last hits that establishes a threshold for the blob class.

        blob_ener_th: FLOAT
    Energy threshold for the last hits of a track to become blob class.

        Rmax: NaN or FLOAT
    Value to perform the fiducial cut of the hits. If NaN, the cut is not done.

        small_blob_th: FLOAT
    Threshold for the energy of a group of blob hits to become marked as small.

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

    #Seleccionamos los hits activos
    mchits = mchits[mchits.label == 'ACTIVE']

    #Etiquetamos los hits
    labelled_hits = add_hits_labels_MC(mchits, mcpart, sig_creator = sig_creator,
                                       blob_ener_loss_th = blob_ener_loss_th, blob_ener_th = blob_ener_th)

    #Hacemos los cortes en los hits
    labelled_hits = hit_data_cuts(labelled_hits, bins, Rmax = Rmax)

    #Hacemos la máscara de blob pequeño
    labelled_hits = add_small_blob_mask(labelled_hits, small_blob_th = small_blob_th)

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
        small_b  = np.array(event_hits['small_b'])

        label_histo, ener_histo, ratio_histo = voxel_labelling_MC(img, mccoors, mcenes, labels, small_b,  bins)

        voxelization_df = voxelization_df.append(histog_to_coord(event_id, label_histo, ener_histo, ratio_histo, bins, binnum = binclass))

    voxelization_df.reset_index()

    #Con esto reducimos los voxeles a meros puntos por sencillez, ya que nos deshacemos del tamaño de voxel
    #y ponemos su origen en 0
    #(el tamaño se tuvo ya en cuenta en la voxelizacion y por tanto ahora esto es indiferente)
    for coord, (size, start) in zip(['x', 'y', 'z'], zip(voxel_size, start_bin)):
        voxelization_df[coord] = voxelization_df[coord] - start
        voxelization_df[coord] = voxelization_df[coord] / size

    #Hacemos enteras las coord y labels
    for colname in voxelization_df.columns:
        voxelization_df[colname] = pd.to_numeric(voxelization_df[colname], downcast = 'integer')

    return voxelization_df, labelled_hits
