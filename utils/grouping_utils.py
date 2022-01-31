import numpy as np
import pands as pd

import networkx as nx
import itertools

def create_graph(df, max_distance, coords):
    '''
    Takes a dataframe and creates a graph with the coordinates as nodes, which are connected by edges 
    if they are separated less than certain distance.
    
    Args:
        df: DATAFRAME
    Contains spatial information (at least).
        
        max_distance: FLOAT
    Indicates the maximum distance between nodes to be connected.
        
        coords: LIST OF STR
    Indicates the names of the df columns that have the coordinates info.
    
    RETURNS: 
        graph: NETWORKX GRAPH
    Graph with the nodes and their connections.
    '''
    
    nodes = [tuple(x) for x in df[coords].to_numpy()]
    
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    
    #Ahora hacemos los edges para contar las componentes conexas
    for va, vb in itertools.combinations(graph.nodes(), 2):
        va_arr, vb_arr = np.array(va), np.array(vb)
        dis = np.linalg.norm(va_arr-vb_arr)
        if dis <= max_distance:
            graph.add_edge(va, vb, distance = dis)
    return graph


def add_group_label(global_label, subgraphs, coords, group_label):
    '''
    It creates a dataframe for each subgraph with its nodes and a new label for each one, referred
    to what group it correspond. The label will be A_B, being A the segclass or the name of the 
    general group, and B the specific identificator of this group (as A can have various groups).
    
    Args:
        global_label: STR or INT
    The global name of the group, being usually segclass labels or "c" for cloud.
    
        subgraphs: LIST
    It contains all the connected graphs individually from a global event.
    
        coords: LIST
    Contains the names of the coordinate colums for the dataframe, usually ['x', 'y', 'z']
    
        group_label: STR
    Name for the new column that contains the new information about each group of voxels. We will
    use 'element' for the small groups of voxels separated by class, and 'cloud' for each set of voxels
    in an event that are close (usually we are going to find the main track and then some other small things)
    
    RETURNS:
        subgraphs_dfs: DATAFRAME
    A dataframme with all the voxeles categorized, either because they belong to a certain group of voxels
    of the same class or because they belong to a group of close voxels in general that form a cloud.
    '''
    
    subgraphs_dfs = pd.DataFrame()
    for j, sgph in enumerate(subgraphs):
        nodes_array = np.array(list(sgph.nodes))
        element_label = np.array([[str(global_label) + '_' + str(j)]] * len(nodes_array))
        
        nodes_array = np.append(nodes_array, element_label, axis = 1)
        
        subgraphs_dfs = subgraphs_dfs.append(pd.DataFrame(nodes_array, columns = coords + [group_label]))
    subgraphs_dfs[coords] = subgraphs_dfs[coords].astype(int)
    return subgraphs_dfs



def order_group_label(global_label, df, subgraph_df, coords, ene_label, group_label):
    '''
    This function takes the output dataframe of add_group_label and joins it to the source dataframe
    where all the voxels information is stored, so we can order each element or cloud by their energy.
    Using the A_B type of labelling, B will take numbers from 0 to the number of groups, being 0 the most
    energetic group.
    
    Args:
        global_label: STR or INT
    The global name of the group, being usually segclass labels or "c" for cloud.

        df: DATAFRAME
    Contains all the voxels information. We want to merge it with the subgraph_df we obtained from the
    add_group_label function so we can sum all the energy per group.
    
        subgraph_df: DATAFRAME
    Contains positions and their group label. It was obtained in the add_group_label function.
    
        coords: LIST
    Contains the names of the coordinate colums for the dataframe, usually ['x', 'y', 'z']
    
        ene_label: STR
    Name for the energy column in the general dataframe (df variable).
    
        group_label: STR
    Name for the new column that contains the new information about each group of voxels. We will
    use 'element' for the small groups of voxels separated by class, and 'cloud' for each set of voxels
    in an event that are close (usually we are going to find the main track and then some other small things)
        
    RETURNS:
        subgraph_df: DATAFRAME
    Contains the unified information from df and the subgraph_df input variable, but with the group labels now
    ordered by energy.
    '''
    
    #We join each subgraph voxel with its energy from the event dataframe
    subgraph_df = subgraph_df.merge(df[coords + [ene_label]], on = coords)
    
    #We create a df with the element label and the sum of the energy of all the voxels of the same element
    #Then arrange it and order from the highest energy to the lowest
    group_label_ener = group_label + '_ener'
    
    elem_ener_df = subgraph_df.groupby(group_label).agg({ene_label:[(group_label_ener, sum)]})
    elem_ener_df.columns = elem_ener_df.columns.get_level_values(1)
    elem_ener_df.reset_index(inplace=True)
    elem_ener_df = elem_ener_df.sort_values(group_label_ener, ascending = False)
    elem_ener_df = elem_ener_df.reset_index(drop = True)
    
    #We now sort the element labels in an array to append to elem_ener_df just like that, so now they
    #are sorted by energy
    order_idx = np.array([str(global_label) + '_' + str(idx) for idx in elem_ener_df.index]) 
    order_idx = np.sort(order_idx).T
    elem_ener_df = elem_ener_df.join(pd.DataFrame(order_idx, columns = ['order_elem']))

    #We join the new ordered element label and merge it with its coordinates by 'element'
    #We drop the energy so is not repeated, the old element and rename the new ordered element as simply element
    subgraph_df = subgraph_df.merge(elem_ener_df, on = [group_label]).drop([ene_label, group_label], axis = 1)
    subgraph_df = subgraph_df.rename(columns={'order_elem': group_label})
    
    return subgraph_df



def label_ordered_elements(df, max_distance, coords, ene_label, seg_label = 'segclass', group_label = 'elem'):
    '''
    This function uses the add_group_label to give to each group of voxels with the same segclass a label that
    identifies them, and then the order_group_label orders their label names by their energy.
    
    Args:
        df: DATAFRAME
    Contains all the voxels information. We will group it by segclass label and then perform the group label
    for each segclass.
    
        max_distance: FLOAT
    Indicates the maximum distance between nodes to be connected.
    
        coords: LIST
    Contains the names of the coordinate colums for the dataframe, usually ['x', 'y', 'z']
    
        ene_label: STR
    Name for the energy column in the general dataframe (the df variable).
    
        seg_label: STR
    Name for the segclass column in the general dataframe (the df variable).
    
        group_label: STR
    Name for the new column that contains the new information about each group of voxels. We will
    use 'element' for the small groups of voxels separated by class.
        
    RETURNS:
        df: DATAFRAME
    The input df variable but with two new columns: elem, that contains the segclass_group label, and 
    elem_ener, that contains the energy for each group.
    '''
    
    df_element = pd.DataFrame()
    for seg, seg_df in df.groupby(seg_label):
        graph = create_graph(seg_df, max_distance, coords)
            
        #Dividimos en subgraphs conectadas       
        subgraphs = [graph.subgraph(sc) for sc in nx.connected_components(graph)]
    
        #Labelleamos los distintos elementos conectados de una misma segclass
        subgraph_df = add_group_label(seg, subgraphs, coords, group_label)
    
        #aqui va a ir la funcion de ordenar por energía!!
        subgraph_df = order_group_label(seg, seg_df, subgraph_df, coords, ene_label, group_label)
    
        #vamos juntando todos para tener al final un df completo
        df_element = df_element.append(subgraph_df)
    
    df = df.merge(df_element, on = coords)
    return df



def label_ordered_clouds(df, max_distance, coords, ene_label, group_label = 'cloud', cloud_label = 'c'):
    '''
    Analogue function to the label_ordered_elements but done for all the voxels, so now the groups are the 
    clouds of each event.
    
    Args:
        df: DATAFRAME
    Contains all the voxels information.
    
        max_distance: FLOAT
    Indicates the maximum distance between nodes to be connected.
    
        coords: LIST
    Contains the names of the coordinate colums for the dataframe, usually ['x', 'y', 'z']
    
        ene_label: STR
    Name for the energy column in the general dataframe (the df variable).
    
        seg_label: STR
    Name for the segclass column in the general dataframe (the df variable).
    
        group_label: STR
    Name for the new column that contains the new information about each group of voxels. We will
    use 'cloud' for the groups of voxels held in an event.
    
        cloud_label: STR
    Identificator for the label of each group; we will use 'c'.
        
    RETURNS:
        df: DATAFRAME
    The input df variable but with two new columns: cloud, that contains the c_group label, and 
    cloud_ener, that contains the energy for each group.
    '''
    
    #Creo la graph con todos los voxeles de un evento
    graph = create_graph(df, max_distance, coords)
    
    #Separo en subgraphs
    subgraphs = [graph.subgraph(sc) for sc in nx.connected_components(graph)]
    
    #Les añado la etiqueta a los voxeles de cada subgraph cloud
    subgraph_df = add_group_label(cloud_label, subgraphs, coords, group_label)
    
    #Ordeno las clouds de mayor a menor energía
    subgraph_df = order_group_label(cloud_label, df, subgraph_df, coords, ene_label, group_label)
    
    #Uno esta nueva información al df principal
    df = df.merge(subgraph_df, on = coords)
    return df



def transform_into_df(voxels_info,  
                      coords = ['x', 'y', 'z'], 
                      identifyer = 'dataset_id', 
                      ene_label = 'ener', 
                      seg_label = 'segclass'):
    '''
    This function transforms the input into a DF to use the element labelling functions.
    
    Args: 
        voxels_info: DATAFRAME or TUPLE/ARRAY/LIST of TUPLES/ARRAYS/LISTS in any combination
    It should contain per event its dataset_id and the positions, energy and segclass of the voxels. 
    If we don't use a DataFrame, the input must have the structure (dataset_id, x, y, z, energy, segclass).
    Its shape will be (6, N).
    
        coords: LIST
    Contains the names of the coordinate colums for the dataframe, usually ['x', 'y', 'z']
    
        identifyer: STR
    Name for the column of the event identifyer.
    
        ene_label: STR
    Name for the energy column.
    
        seg_label: STR
    Name for the segclass column.
    
    RETURNS: 
        voxels_info: DATAFRAME
    It's simply the same DF as the input or the set of values turned into a DF
    '''
    if type(voxels_info) == type(pd.DataFrame()):
        pass
    else:
        positions = np.array(voxels_info).T
        voxels_info = pd.DataFrame(positions, columns = [identifyer] + coords + [ene_label, seg_label])
    return voxels_info



def label_event_elements(labelled_voxels, 
                         max_distance, 
                         coords = ['x', 'y', 'z'], 
                         identifyer = 'dataset_id', 
                         ene_label = 'ener', 
                         seg_label = 'segclass',
                         beersh_dict = None):
    '''
    The function performs the element (by segclass) and cloud labelling for a bunch of events.
    
    Args:
        labelled_voxels: DATAFRAME or TUPLE/ARRAY/LIST of TUPLES/ARRAYS/LISTS in any combination
    It should contain per event its dataset_id and the positions, energy and segclass of the voxels. 
    If we don't use a DataFrame, the input must have the structure (dataset_id, x, y, z, energy, segclass).
    Its shape will be (6, N).
    
        max_distance: FLOAT
    Indicates the maximum distance between nodes to be connected.
        
        coords: LIST
    Contains the names of the coordinate colums for the dataframe, usually ['x', 'y', 'z']
    
        identifyer: STR
    Name for the column of the event identifyer.
    
        ene_label: STR
    Name for the energy column.
    
        seg_label: STR
    Name for the segclass column.
    
        beersh_dict: DICT or None
    Has the correspondance between classes, so we can group neighbour classes with original classes
    by asigning them the same number (i.e. the dict can have class 1 - other and class 4 - other neighbour marked
    with the number 1, so they are grouped together by this algorythm). If None, the 
    
    RETURNS:
        output_df: DATAFRAME
    Contains the same information as de DF/set of values of the labelled_voxels input, with the new columns
    for the element label, the element energy, the cloud label and the cloud energy.
    '''
    
    labelled_voxels = transform_into_df(labelled_voxels, 
                                        coords = coords, 
                                        identifyer = identifyer, 
                                        ene_label = ene_label, 
                                        seg_label = seg_label)
    #If we have a correspondance dictionary, we create a new column that renames the neighbour segclass as
    #the original segclass, and change the segclass label to the name of this new column. At the end we can delete
    #it if we want
    if beersh_dict != None:
        labelled_voxels = labelled_voxels.assign(group_segclass = labelled_voxels[seg_label].map(beersh_dict))
        seg_label = 'group_segclass'
        
    output_df = pd.DataFrame()
    for idx, df_event in labelled_voxels.groupby(identifyer):
        
        df_event  = label_ordered_elements(df_event, max_distance, coords, ene_label, seg_label = seg_label)
        df_event  = label_ordered_clouds(df_event, max_distance, coords, ene_label)
        
        output_df = output_df.append(df_event, ignore_index=True)
    
    #We drop this auxiliary column
    if beersh_dict != None:
        output_df = output_df.drop(seg_label, axis = 1)
        
    return output_df
