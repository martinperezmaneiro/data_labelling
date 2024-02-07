import sys
import pandas as pd
import networkx as nx

from utils.graph_utils import create_graph_dict

from invisible_cities.io.dst_io            import load_dst
from invisible_cities.core.configure       import configure


if __name__ == "__main__":
    config   = configure(sys.argv).as_namespace

    #picks the output of the labelling
    infile  = config.file_out
    outfile = infile.replace('.h5', '_graph.h5')

    voxels = load_dst(infile, 'DATASET', config.voxel_tablename)

    graphs = create_graph_dict(voxels, config.max_distance, coords = config.coords, ener_label = config.ener_label)

    nx.write_gpickle(graphs, outfile)