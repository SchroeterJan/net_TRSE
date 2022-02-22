from plotting.plots import *

import math
from sklearn import preprocessing

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

    def mix_otp_bike(self):
        short = np.asarray(self.euclid < short_trip).nonzero()
        self.mixed = self.otp
        for i, j in enumerate(short[0]):
            self.mixed[j, short[1][i]] = self.bike[j, short[1][i]]

    def edu_score(self):
        edu_ = self.neighborhood_se[census_variables[:3]].values
        scores = []
        for row in edu_:
            score = 0
            if np.nansum(row) == 100.0:
                score = row[0]*3 + row[1] * 2 + row[2] * 1
            else:
                for i, val in enumerate(row):
                    if np.isnan(val):
                        continue
                    else:
                        score += val * (i+1)
                score += 100.0 - np.nansum(row)
            scores.append(score)
        self.neighborhood_se['edu_score'] = scores





    def stat_prep(self, model_variables):
        for var in model_variables:
            self.neighborhood_se[var] = reject_outliers(self.neighborhood_se[var].to_numpy(),
                                                        m=5.0)

        self.model_ = self.neighborhood_se[model_variables]
        x = self.model_.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.model_ = pd.DataFrame(data=x_scaled,
                                         columns=self.model_.columns,
                                         index=self.model_.index)
        #
        # max_val = max(self.neighborhood_se[var])
        # if scale:
        #     self.neighborhood_se[var] = [100.0*(each/max_val) for each in self.neighborhood_se[var]]

    def reduce_matrix(self, frame):
        matrix = np.triu(m=frame.to_numpy(), k=0)
        matrix[matrix == 0] = np.nan
        return matrix

    def get_q_ij(self):
        # reciprocal of the route factor - multiplied to result in km/h
        self.otp_qij = self.euclid / self.otp * 3.6
        self.bike_qij = self.euclid / self.bike * 3.6
        #self.pt_qij = self.euclid / self.pt * 3.6
        #self.mixed_q = self.euclid / self.mixed

    def get_q(self):
        self.neighborhood_se['otp_q'] = (np.nan_to_num(self.otp_qij).sum(axis=1) +
                                         np.nan_to_num(self.otp_qij).sum(axis=0)) / (len(self.otp_qij[0]) - 1)
        self.neighborhood_se['bike_q'] = (np.nan_to_num(self.bike_qij).sum(axis=1) +
                                          np.nan_to_num(self.bike_qij).sum(axis=0)) / (len(self.bike_qij[0]) - 1)
        #self.neighborhood_se['mixed_q'] = (np.nan_to_num(self.mixed_q).sum(axis=1) +
        #                                 np.nan_to_num(self.mixed_q).sum(axis=0)) / (len(self.mixed_q[0]) - 1)


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


class Skater:

    def __init__(self, model, geo_frame):
        print("Initializing " + self.__class__.__name__)
        # areas into geopandas DataFrame
        self.geo_df = geo_frame
        # self.geo_df['lon'] = self.geo_df['centroid'].apply(lambda p: p.x)
        # self.geo_df['lat'] = self.geo_df['centroid'].apply(lambda p: p.y)

        self.pos = {}
        self.adj_g = []
        self.geo_pos()

        # create DataFrame containing relevant socio-economic variables for the model
        self.model_df = model

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
                                              np.fromiter(node_dist_dicts[dist_ind].keys(), dtype='U4'),
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


    # Minimal Spanning Tree Clustering according to https://doi.org/10.1080/13658810600665111
    def mst(self):
        print('Create MST')
        # get adjacency graph
        self.adjacency_graph()

        # iterate over all graph edges to assign a cost
        for u, v in self.adj_g.edges():
            # euclidean distance between attribute vectors
            dist = np.nansum([(self.model_df.loc[u][col] - self.model_df.loc[v][col])**2 for col in range(len(self.model_df.columns.values))])
            self.adj_g[u][v]['cost'] = dist

        mst = nx.algorithms.tree.mst.minimum_spanning_tree(G=self.adj_g, weight='cost', algorithm='prim')
        """    
        # MST generation
        v_1 = np.random.randint(low=0, high=len(model_df))
        mst = nx.Graph()
        mst.add_node(node_for_adding=np.array(model_df.index)[v_1])
        while len(mst.nodes()) <= len(model_df):
            # identify potential candidates by checking for all edges of all vertices currently in the MST
            candidates = np.array(list(graph.edges(mst.nodes(), data=True)))
            # remove edges leading to vertices already in the MST
            candidates = candidates[np.invert(np.in1d(candidates[:, 1], np.array(list(mst.nodes()))))]
            cost_list = [candidate['cost'] for candidate in candidates[:, 2]]
            min_ind = np.argmin(cost_list)
            mst.add_node(node_for_adding=candidates[min_ind][1])
            mst.add_edge(u_of_edge=candidates[min_ind][0], v_of_edge=candidates[min_ind][1])
            print(len(mst.nodes()))
        """
        return mst

    # calculating the intracluster square deviation ("sum of square deviations", SSD)
    def ssd(self, k, x):
        # initialize the SSD
        ssd_k = 0.0
        # get average for all attributes
        attributes_av = [x[col].mean() for col in x.columns]
        # iterate over nodes in tree k
        for i in list(k.nodes()):
            # add SD of each node in tree k to SSD
            ssd_k += np.nansum([(x.loc[i][j] - attributes_av[j]) ** 2 for j in range(len(x.columns))])
        return ssd_k

    # objective function 1 and balancing function
    def objective_functions(self, ssd_t, t_a, t_b):
        ssd_t_a = self.ssd(k=t_a, x=self.model_df.loc[list(t_a.nodes())])
        ssd_t_b = self.ssd(k=t_b, x=self.model_df.loc[list(t_b.nodes())])
        f = ssd_t-(ssd_t_a+ssd_t_b)
        f_2 = min((ssd_t - ssd_t_a), (ssd_t - ssd_t_b))
        return f, f_2

    # solution creation by removing an edge l
    def potential_solution(self, edges, graph):
        s_p = []
        for l in edges:
            mst_copy = graph.copy()
            mst_copy.remove_edge(l[0], l[1])
            split = [graph.subgraph(c).copy() for c in nx.connected_components(mst_copy)]
            s_p.append(split)
        return s_p

    def plot_tree(self, c):
        fig, ax = plt.subplots(figsize=(20, 15))
        geo_plot(frame=self.geo_df, column='clust', axis=ax, cmap='tab20')
        ax.set_title('SKATER clustering', fontsize=40)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_skater, 'c_' + c))
        plt.close(fig)

    def plot_clust_init(self):
        colors = {1: 'tab:blue',
                  2: 'lightsteelblue',
                  3: 'tab:orange',
                  4: 'moccasin',
                  5: 'tab:green',
                  6: 'mediumseagreen',
                  7: 'tab:red',
                  8: 'salmon',
                  9: 'tab:purple',
                  10: 'plum',
                  11: 'tab:brown',
                  12: 'sandybrown',
                  13: 'tab:pink',
                  14: 'palevioletred',
                  15: 'tab:gray',
                  16: 'lightgray',
                  17: 'tab:olive',
                  18: 'gold',
                  19: 'tab:cyan',
                  20: 'aquamarine'}

        self.fig, self.ax = plt.subplots(figsize=(20, 15))
        self.ax.set_aspect('equal')
        # self.geo_df.boundary.plot(ax=self.ax, edgecolor='black')
        self.ax.set_axis_off()

        return colors


    def tree_patitioning(self, c, plot=False):
        print('Start tree partitioning')
        components = [self.mst()]
        sc = 25
        if plot:
            colors = self.plot_clust_init()
            self.geo_df['clust'] = 0
            self.geo_df.plot(ax=self.ax, color=colors[1])
            plt.tight_layout()
            plt.savefig(fname=os.path.join(path_skater, 'c_0'))

        for clust in range(c):
            best_edges = []
            best_edges_values = []
            print('forming cluster ' + str(clust))
            for t in components:
                print('Test component for split')
                # get all possible solutions for finding the starting vertice by iteration over all edges of the MST
                s_p_1 = self.potential_solution(edges=t.edges(), graph=t)
                # calculate difference in number of vertices between two subtrees of a possible split
                split_dif = [(abs(len(split[0].nodes()) - len(split[1].nodes()))) for split in s_p_1]
                # edge/vertex that best splits the MST into two subtrees of similar size
                if len(split_dif) == 0:
                    v_c = list(t.nodes())[0]
                else:
                    split_edge = list(t.edges())[np.argmin(split_dif)]
                    v_c = split_edge[0]

                # Step 1
                # get possible solutions s_p for all edges incident to v_c
                s_p_edges = list(t.edges(v_c))
                n = 0
                n_star = 0
                f_s_star = 0

                list_l = {}
                edges_expanded = []
                while n - n_star <= sc:
                    # Step 2
                    if len(s_p_edges) == 0:
                        break
                    f1_list = []
                    s_p = self.potential_solution(edges=s_p_edges, graph=t)
                    ssd_t = self.ssd(k=t, x=self.model_df.loc[list(t.nodes())])
                    for comp, s in enumerate(s_p):
                        f, f_2 = self.objective_functions(ssd_t=ssd_t, t_a=s[0], t_b=s[1])
                        list_l[s_p_edges[comp]] = f_2
                        f1_list.append(f)
                    f1_s_j = max(f1_list)

                    # Step 3
                    if f1_s_j > f_s_star:
                        s_star = s_p_edges[np.argmax(f1_list)]
                        f_s_star = f1_s_j
                        n_star = n
                        print('new best solution at n= ' + str(n))
                    n += 1

                    # Step 4
                    next = list(list_l.keys())[np.argmin(list(list_l.values()))]
                    if next in set(edges_expanded):
                        print('something is weird')
                    edges_expanded.append(next)
                    del list_l[next]
                    s_p_edges = list(set(list(t.edges(next[0])) + list(t.edges(next[1]))) - set(edges_expanded))

                best_edges.append(s_star)
                best_edges_values.append(f_s_star)
            best_ind = np.argmax(best_edges_values)
            best_edge = best_edges[best_ind]
            components[best_ind].remove_edge(best_edge[0], best_edge[1])
            components.extend([components[best_ind].subgraph(c).copy() for c in nx.connected_components(components[best_ind])])
            del components[best_ind]
            if plot:
                for node in list(components[-1].nodes()):
                    self.geo_df.at[node, 'clust'] = clust + 1
                comp_geo = self.geo_df.loc[list(components[-1].nodes())]
                comp_geo.plot(ax=self.ax, color=colors[clust+2])
                # plt.text(s='2',
                #          x=np.array(comp_geo.dissolve().representative_point()[0].coords.xy)[0],
                #          y=np.array(comp_geo.dissolve().representative_point()[0].coords.xy)[1],
                #          horizontalalignment='center',
                #          fontsize=22,
                #          color='k')
                self.ax.set_axis_off()
                self.ax.set_title('SKATER clustering', fontsize=40)
                plt.tight_layout()
                plt.savefig(fname=os.path.join(path_skater, 'c_' + str(clust + 1)))


        return components


# hist_scaled_se()
# clusters = get_cluster()
# hist_cluster()

# plot_adj_mat()