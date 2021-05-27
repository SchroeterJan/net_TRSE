from Classes import *
from processors import *
from config import *

# Class objects
handler = DataHandling()
analyzer = Analysis(handler=handler)


# declare new path for experiment data
if not os.path.isdir(path_experiments):
    os.mkdir(path_experiments)


def reduce_se_variables():
    census_variables.remove('BEVTOTAAL')
    census_variables.remove('SKSES_GEM')


# get total in- and outbound passenger flow for each stop
def stop_flows():
    path_stopflows = os.path.join(path_experiments, 'stop_flows.csv')
    # follow the data preparation steps up to the assignment algorithm
    flow_prep = PassengerCounts()
    flow_prep.area_stop_matching()
    flow_prep.filter_connections()

    # cast flows to integer, "coerce" sets non integer values to nan
    flow_prep.pass_data[column_names['pass_vol']] = \
        pd.to_numeric(flow_prep.pass_data[column_names['pass_vol']], errors='coerce')
    # group by stop and sum the flows for both origin and destination
    or_sum = flow_prep.pass_data.groupby([column_names['pass_or']]).sum()
    dest_sum = flow_prep.pass_data.groupby([column_names['pass_dest']]).sum()
    # join sums to stop list
    flow_prep.stops = flow_prep.stops.set_index(keys=column_names['stop_name'], drop=True)
    flow_prep.stops = flow_prep.stops.join(or_sum)
    flow_prep.stops = flow_prep.stops.join(dest_sum, rsuffix='dest_flows')
    # write stop list to disk
    flow_prep.stops.columns = [column_names['stop_lat'], column_names['stop_lng'], 'or_flows', 'dest_flows']
    flow_prep.stops.to_csv(path_or_buf=path_stopflows, sep=';', index=True)


# calculate all differences between entries of a given vector
def difference_vector(vector):
    edges = []
    for i, value_i in enumerate(vector):
        for j, value_j in enumerate(vector[i + 1:]):
            edges.append(np.absolute(value_i - value_j))
    return np.array(edges)


# form difference matrices for socio-economic variables
def se_matrices():
    builder = TransportPrep()
    for variable in census_variables:
        matrix = difference_vector(handler.neighborhood_se[variable])
        matrix = builder.build_matrix(length=len(handler.neighborhood_se[variable]), data_list=list(matrix))
        matrix = pd.DataFrame(data=matrix,
                              index=handler.neighborhood_se[column_names['geo_id_col']],
                              columns=handler.neighborhood_se[column_names['geo_id_col']])
        matrix.to_csv(path_or_buf=os.path.join(path_experiments, 'matrix_' + variable), sep=';', index=True)


# CLUSTERING
def cluster_all():
    clusters['pt_all'] = analyzer.clustering(matrix=handler.pt.to_numpy())
    clusters['bike_all'] = analyzer.clustering(matrix=handler.bike.to_numpy())
    clusters['flows_all'] = analyzer.clustering(matrix=handler.flows.to_numpy())


def cluster_rel():
    unused = np.where(np.isnan(handler.flows.to_numpy()))
    clust_pt = np.array(handler.pt)
    clust_pt[unused[0], unused[1]] = 0.0
    clust_bike = np.array(handler.bike)
    clust_bike[unused[0], unused[1]] = 0.0
    clusters['pt_rel'] = analyzer.clustering(matrix=clust_pt)
    clusters['bike_rel'] = analyzer.clustering(matrix=clust_bike)


def get_cluster():
    if os.path.isfile(path_clustercoeff):
        clusters = pd.read_csv(filepath_or_buffer=path_clustercoeff, sep=';')
    else:
        clusters = pd.DataFrame(index=handler.neighborhood_se.index)
        cluster_all()
        cluster_rel()
        clusters.to_csv(path_or_buf=path_clustercoeff, sep=';', index=False)
    return clusters


def se_kmean():
    reduce_se_variables()
    se = np.array(handler.neighborhood_se[census_variables])
    se_kmean = KMeans(n_clusters=6, random_state=0).fit(se)
    se_clust = se_kmean.labels_
    geo_frame = geopandas.GeoDataFrame(crs="EPSG:4326",
                                       geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometrie))
    geo_frame['clust'] = se_clust
    return geo_frame


class Skater:

    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        # areas into geopandas DataFrame
        self.geo_df = geopandas.GeoDataFrame(crs="EPSG:4326",
                                             geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometrie))
        self.geo_df['centroid'] = geopandas.GeoSeries.from_wkt(handler.neighborhood_se['centroid'])
        self.geo_df['lon'] = self.geo_df['centroid'].apply(lambda p: p.x)
        self.geo_df['lat'] = self.geo_df['centroid'].apply(lambda p: p.y)

        self.pos = {}
        self.geo_pos()

        # create DataFrame containing relevant socio-economic variables for the model
        self.model_df = handler.neighborhood_se[model_variables]

    def geo_pos(self):
        # get positions of nodes to make the graph spatial
        for count, elem in enumerate(np.array(self.geo_df[['lon', 'lat']])):
            self.pos[self.geo_df.index[count]] = (elem[0], elem[1])

    def adjacency_graph(self):
        # create adjacency matrix for all areas
        mat = np.invert(np.array([self.geo_df.geometry.disjoint(pol) for pol in self.geo_df.geometry]))
        # correct for self-loops
        np.fill_diagonal(a=mat, val=False)
        # boolean matrix into networkx graph
        adj_g = nx.convert_matrix.from_numpy_array(A=mat)
        # dict index and name of areas
        area_dict = {}
        for i, area in enumerate(np.array(self.geo_df.index)):
            area_dict[i] = area
        # relabel nodes to area identifier
        adj_g = nx.relabel.relabel_nodes(G=adj_g, mapping=area_dict)

        # connect spatially disconnected subgraphs
        # find connected components of the graph and create subgraphs
        S = [adj_g.subgraph(c).copy() for c in nx.connected_components(adj_g)]
        # only connect if there are disconnected subgraphs
        if len(S) != 1:
            # get index of largest connected component
            largest_cc = np.argmax([len(graph.nodes()) for graph in S])
            # iterate over subgraphs except the largest component
            for subgraph in S[:largest_cc] + S[largest_cc+1:]:
                subgraph_dist = []
                # declare space of possible connection by considering all nodes outside the current subgraph
                candidate_space = adj_g.copy()
                candidate_space.remove_nodes_from(subgraph.nodes())
                # determine number of connections by fraction of subgraph size
                no_connections = math.ceil(len(subgraph.nodes())/5)
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
                for ind in min_dist_ind:
                    adj_g.add_edge(u_of_edge=subgraph_dist[ind][0],
                               v_of_edge=subgraph_dist[ind][1][0],
                               cost=subgraph_dist[ind][2][0])
        return adj_g

    # Minimal Spanning Tree Clustering according to https://doi.org/10.1080/13658810600665111
    def mst(self):
        print('Create MST')
        # get adjacency graph
        graph = self.adjacency_graph()

        # iterate over all graph edges to assign a cost
        for u, v in graph.edges():
            # euclidean distance between attribute vectors
            dist = sum([(np.array(self.model_df.loc[u])[col] - np.array(self.model_df.loc[v])[col])**2
                        for col in range(len(self.model_df.columns.values))])
            graph[u][v]['cost'] = dist

        mst = nx.algorithms.tree.mst.minimum_spanning_tree(G=graph, weight='cost', algorithm='prim')
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
            ssd_k += sum([(x.loc[i][j] - attributes_av[j]) ** 2 for j in range(len(x.columns))])
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

    def tree_patitioning(self):
        print('Start tree partitioning')
        components = [self.mst()]
        sc = 20

        for clust in range(15):
            best_edges = []
            best_edges_values = []
            print('Form new cluster')
            for t in components:
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
                    for i, s in enumerate(s_p):
                        f, f_2 = self.objective_functions(ssd_t=ssd_t, t_a=s[0], t_b=s[1])
                        list_l[s_p_edges[i]] = f_2
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

        return components


# stop_flows()
# se_matrices()
# clusters = get_cluster()
