import itertools
import numpy    as np
import networkx as nx

def create_graph_with_attributes(df, max_distance, coords, ener_label = 'ener', add_distant_edges = False):
    '''
    Takes a dataframe and creates a graph with the coordinates as nodes, which are connected by edges 
    if they are separated less than certain distance. Is very similar to the other one, but here attributes
    are defined in a different way
    
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
    
    voxel = [tuple(x) for x in df[coords].to_numpy()]
    eners = df[ener_label].values
    segcl = df['segclass'].values
    if np.isin('nhits', df.columns):
        nhits = df['nhits'].values
    else:
        nhits = np.zeros(len(df))
    
    graph = nx.Graph()
    graph.add_nodes_from(range(len(voxel)))

    attributes = {i:{'voxel':v, 'ener':e, 'segclass':s, 'nhits':n} for i, (v, e, s, n) in enumerate(zip(voxel, eners, segcl, nhits))}
    nx.set_node_attributes(graph, attributes)
    
    #Ahora hacemos los edges para contar las componentes conexas
    for va, vb in itertools.combinations(graph.nodes(data=True), 2):
        va_arr, vb_arr = np.array(va[1]['voxel']), np.array(vb[1]['voxel'])
        dis = np.linalg.norm(va_arr-vb_arr)
        if dis <= max_distance:
            graph.add_edge(va[0], vb[0], distance = dis, adj = 1)
        #new addition to add all edges, with a label to differ them from close edges
        elif add_distant_edges:
            graph.add_edge(va[0], vb[0], distance = dis, adj = 0)
    return graph

def create_graph_dict(voxels, max_distance, coords = ['x', 'y', 'z'], ener_label = 'ener', add_distant_edges = False):
    '''
    Creates a dict with all the graphs for a voxelized file, joining two voxels if they are at most
    the max_distance separation.
    '''
    graphs = {}
    for i, event in voxels.groupby('dataset_id'):
        g = create_graph_with_attributes(event, max_distance, coords, ener_label = ener_label, add_distant_edges=add_distant_edges)
        graphs[i] = g
    return graphs