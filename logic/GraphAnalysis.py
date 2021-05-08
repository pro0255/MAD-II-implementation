import networkx as nx
from enum import Enum
import numpy as np
import time
import pandas as pd


class GraphProperties(Enum):
    N = "Number of verticies"
    M = "Number of edges"
    MAX_D = 'Max degree'
    AVG_D =  "Average degree"
    AVG_C = 'Average clustering coefficient'
    LEN_CC = 'Number of components >2'
    LEN_ISOLATED = 'Number of isolated verticies'
    MAX_CC = 'Max component'



class GraphAnalysis:
    def __init__(self):
        self.properties = {}


    def make_calculation(self, G):
        start = time.time()
        n = len(G.nodes())
        m  = len(G.edges())

        nodes, degress = zip(*G.degree())

        max_d = np.max(degress)
        avg_d = (2*m)/m


        avg_c = nx.average_clustering(G)

        cc = nx.connected_components(G)
        len_cc = np.array([len(c) for c in cc])

        len_cc_2 = len(np.argwhere(len_cc >= 2))
        isolated = len(np.argwhere(len_cc < 2))
        max_cc = np.max(len_cc)


        self.properties[GraphProperties.N.value] = n
        self.properties[GraphProperties.M.value] = m
        self.properties[GraphProperties.MAX_D.value] =  max_d
        self.properties[GraphProperties.AVG_D.value] =  avg_d
        self.properties[GraphProperties.AVG_C.value] =  avg_c
        self.properties[GraphProperties.LEN_CC.value] =  len_cc_2
        self.properties[GraphProperties.LEN_ISOLATED.value] =  isolated
        self.properties[GraphProperties.MAX_CC.value] =  max_cc

        end = time.time()

        self.properties['Time'] = end - start


    def get_df(self):
        dic = {}
        for k, v in self.properties.items():
            dic[k] = v
        df = pd.DataFrame.from_dict(dic, orient='index', columns=['Analysis properties'])
        return df



    def get_analysis_dictionary(self):
        return self.properties