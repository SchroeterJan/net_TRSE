from exp_resources import *


def exec_skater():
    mst_plot(data=Skater(variables=model_variables))
    skat = Skater()
    c = 15

    skat.tree_patitioning(c=c, plot=True)
    animate_skater(c)


def accessibility_barth():
    handler.mix_otp_bike()
    handler.get_q_ij()
    handler.get_q()

    #geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
    #                                   geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    #geo_plot(frame=geo_frame, column='q_otp')

    hist_acc_barth(data=handler.neighborhood_se, mode='bike_q')
    hist_acc_barth(data=handler.neighborhood_se, mode='otp_q')
    #hist_acc_barth(data=handler.neighborhood_se, mode='mixed_q')


def flatten(x):
    x = x.flatten()
    x = x[~np.isnan(x)]
    return x


def velocity():
    handler.get_q_ij()

    heatscatter(x=flatten(handler.bike_q),
                y=flatten(handler.euclid),
                xlabel='Overcome distance per second',
                ylabel='Euclidean distance',
                title='Heatscatter OTP',
                av=True)


def clust():
    handler.initiate_graph()
    handler.add_edges(mode='pt')

    cluster_dict = nx.clustering(handler.graph, weight='weight')
    handler.neighborhood_se['otp_clust'] = np.array(list(cluster_dict.values()))

    handler.initiate_graph()
    handler.add_edges(mode='bike')
    cluster_dict = nx.clustering(handler.graph, weight='weight')
    handler.neighborhood_se['bike_clust'] = np.array(list(cluster_dict.values()))

    hist_cluster(handler=handler, mode='otp_clust')
    hist_cluster(handler=handler, mode='bike_clust')


# handler = DataHandling()
# handler.matrices()
# accessibility_barth()
# velocity()
# clust()
# handler.neighborhood_se.to_csv(os.path.join(path_experiments, file_neighborhood_se))
handler = DataHandling(new=True)
handler.matrices()

handler.otp[handler.otp > 3600.0] = np.nan
a = 1




a = 10
#clust()


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
    builder = ODPrep()
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








# stop_flows()
# se_matrices()
# clusters = get_cluster()
