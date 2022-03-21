from plotting.plots import *

import math
from sklearn import preprocessing
import networkx as nx
from scipy import spatial

travel_times = ['Bike', 'Public Transport']

def flatten(x):
    x = x.flatten()
    x = x[~np.isnan(x)]
    return x


def reject_outliers(data, m):
    # set values who differ more than m times the standart deviation
    nancount = 0
    for each in data:
        if np.isnan(each):
            nancount += 1
    outliers = abs(data - np.nanmean(data)) > m * np.nanstd(data)
    data[outliers] = np.nanmax(data[~outliers])
    return data


class DataHandling:

    def __init__(self, new=False):
        print("Initializing " + self.__class__.__name__)
        self.new = new
        self.load_data()
        #self.mix_otp_bike()

    def load_data(self):
        print("Loading data")
        if self.new:
            self.neighborhood_se = pd.read_csv(filepath_or_buffer=os.path.join(path_experiments, file_neighborhood_se),
                                               sep=',', index_col=0)
        else:
            self.neighborhood_se = pd.read_csv(filepath_or_buffer=path_neighborhood_se,
                                               sep=';', index_col=0)
        self.flows = pd.read_csv(filepath_or_buffer=path_flows, sep=';', index_col=0)
        self.bike = pd.read_csv(filepath_or_buffer=path_bike_matrix, sep=';', index_col=0)
        #self.pt = pd.read_csv(filepath_or_buffer=path_pt_matrix, sep=';', index_col=0)
        self.otp = pd.read_csv(filepath_or_buffer=path_otp_matrix, sep=';', index_col=0)
        self.euclid = pd.read_csv(filepath_or_buffer=path_euclid_matrix, sep=';', index_col=0)

    def matrices(self):
        # flows are generated distinct for journey to and from
        #self.flows = self.reduce_matrix(self.flows)
        self.bike = self.reduce_matrix(self.bike)
        self.otp = self.reduce_matrix(self.otp)
        self.euclid = self.reduce_matrix(self.euclid)
        #self.pt = self.reduce_matrix(self.pt)

    def edu_score(self):
        edu_ = self.neighborhood_se[census_variables[:3]].values
        scores = []
        for row in edu_:
            score = 0
            missing = np.isnan(row).sum()
            if missing == 0:
                score = row[0]*1 + row[1] * 2 + row[2] * 3
            elif missing == 3:
                score = np.nan
            else:
                for i, val in enumerate(row):
                    if np.isnan(val):
                        score += (100.0 - np.nansum(row))/missing * (i+1)
                    else:
                        score += val * (i+1)
            scores.append(score)
        self.neighborhood_se['edu_score'] = scores

    def stat_prep(self, vars):
        # for var in model_variables:
        #     self.neighborhood_se[var] = reject_outliers(self.neighborhood_se[var].to_numpy(),
        #                                                 m=3.0)

        x = self.neighborhood_se[vars].values.reshape(-len(vars), len(vars))
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        x_scaled = min_max_scaler.fit_transform(x)
        self.model_ = pd.DataFrame(data=x_scaled,
                                   columns=vars,
                                   index=self.neighborhood_se.index)

    def reduce_matrix(self, frame):
        matrix = np.triu(m=frame.to_numpy(), k=0)
        matrix[matrix == 0] = np.nan
        return matrix

    def get_q_ij(self):
        # reciprocal of the route factor - multiplied to result in km/h
        self.otp_qij = self.euclid / self.otp * 3.6
        self.bike_qij = self.euclid / self.bike * 3.6

    def get_q(self):
        self.neighborhood_se['otp_q'] = (np.nan_to_num(self.otp_qij).sum(axis=1) +
                                         np.nan_to_num(self.otp_qij).sum(axis=0)) / (len(self.otp_qij[0]) - 1)
        self.neighborhood_se['bike_q'] = (np.nan_to_num(self.bike_qij).sum(axis=1) +
                                          np.nan_to_num(self.bike_qij).sum(axis=0)) / (len(self.bike_qij[0]) - 1)

    def initiate_graph(self):
        #neighborhoods = self.neighborhood_se.index
        self.graph = nx.Graph()
        for neighborhood in self.neighborhood_se[column_names['geo_id_col']]:
            self.graph.add_node(neighborhood)

    def choose_mode(self, mode):
        if mode == 'pt':
            return self.otp
        elif mode == 'bike':
            return self.bike
        else:
            print('PICKED MODE IS NOT PREDEFINED - PLEASE SPECIFY \'pt\' or \'bike\'')

    def add_edges(self, mode):
        matrix = self.choose_mode(mode=mode)
        matrix = np.nan_to_num(matrix)
        # create edges between all nodes and populate them with travel times as weights
        for i, row in enumerate(matrix):
            for j, value in enumerate(row[i+1:]):
                if value != 0.0:
                    self.graph.add_edge(list(self.neighborhood_se[column_names['geo_id_col']])[i],
                                             list(self.neighborhood_se[column_names['geo_id_col']])[j + i+1],
                                             weight=value)


class Custom_Adjacency:

    def __init__(self, geo_frame):
        print("Initializing " + self.__class__.__name__)
        # areas into geopandas DataFrame
        self.geo_df = geo_frame

        self.pos = {}
        self.adj_g = []
        self.geo_pos()
        print('Get custom adjacency graph')
        self.adjacency_graph()

    def geo_pos(self):
        # get positions of nodes to make the graph spatial
        for count, elem in enumerate(np.array(self.geo_df.centroid)):
            self.pos[self.geo_df.index[count]] = (elem.x, elem.y)

    def remove_islands(self):
        # connect spatially disconnected subgraphs
        # find connected components of the graph and create subgraphs
        S = [self.adj_g.subgraph(c).copy() for c in nx.connected_components(self.adj_g)]
        # only connect if there are disconnected subgraphs
        while len(S) != 1:
            # get index of largest connected component
            largest_cc = np.argmax([len(graph.nodes()) for graph in S])
            # iterate over subgraphs except the largest component
            for subgraph in S[:largest_cc] + S[largest_cc + 1:]:
                subgraph_dist = []
                # declare space of possible connection by considering all nodes outside the current subgraph
                candidate_space = self.adj_g.copy()
                candidate_space.remove_nodes_from(subgraph.nodes())
                # determine number of connections by fraction of subgraph size
                no_connections = math.ceil(len(subgraph.nodes()) / 3)
                for node in subgraph.nodes():
                    # get list of dictionaries with the connected point outside the subgraph as key and distance as value
                    node_dist_dicts = [{dest_point: Point(self.pos[dest_point]).distance(Point(self.pos[node]))}
                                       for dest_point in candidate_space.nodes()]
                    # flatten value list
                    dist_list = [list(dict.values()) for dict in node_dist_dicts]
                    dist_list = np.array([item for sublist in dist_list for item in sublist])
                    # get the determined number of shortest possible connections
                    min_dist = np.argsort(dist_list)[:no_connections]
                    for dist_ind in min_dist:
                        subgraph_dist.append([node,
                                              np.fromiter(node_dist_dicts[dist_ind].keys(), dtype=int),
                                              np.fromiter(node_dist_dicts[dist_ind].values(), dtype=float)])
                min_dist_ind = np.argsort(np.array(subgraph_dist, dtype=object)[:, 2])[:no_connections]
                # add edge to connect disconnected subgraphs
                for ind in min_dist_ind:
                    self.adj_g.add_edge(u_of_edge=subgraph_dist[ind][0],
                                   v_of_edge=subgraph_dist[ind][1][0],
                                   cost=subgraph_dist[ind][2][0])
            S = [self.adj_g.subgraph(c).copy() for c in nx.connected_components(self.adj_g)]

    def adjacency_graph(self):
        # check for invalid polygons and apply simple fix according to https://stackoverflow.com/questions/20833344/fix-invalid-polygon-in-shapely
        for i, pol in enumerate(self.geo_df.geometry):
            if not pol.is_valid:
                self.geo_df.geometry[i] = pol.buffer(0)

        # create adjacency matrix for all areas
        mat = np.invert(np.array([self.geo_df.geometry.disjoint(pol) for pol in self.geo_df.geometry]))
        # correct for self-loops
        np.fill_diagonal(a=mat, val=False)
        # boolean matrix into networkx graph
        self.adj_g = nx.convert_matrix.from_numpy_array(A=mat)
        # dict index and name of areas
        area_dict = {}
        for i, area in enumerate(np.array(self.geo_df.index)):
            area_dict[i] = area
        # relabel nodes to area identifier
        self.adj_g = nx.relabel.relabel_nodes(G=self.adj_g, mapping=area_dict)
        self.remove_islands()



