from exp_resources import *

#plot_mst(data=Skater())

# Class objects
handler = DataHandling()

def get_q(matrix, mode):
    q_ij_mode = handler.euclid/ matrix
    hist_qij(q_ij_mode, mode)
    return (np.nan_to_num(q_ij_mode).sum(axis=1) + np.nan_to_num(q_ij_mode).sum(axis=0)) / (len(q_ij_mode[0]) -1)

handler.matrices()
handler.mix_otp_bike()
q_ = {'q_otp': handler.otp,
      'q_bike': handler.bike,
      'q_mixed': handler.mixed}

for mode, data in q_.items():
    handler.neighborhood_se[mode] = get_q(matrix=data, mode=mode)


geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
                                   geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
geo_plot (frame=geo_frame, column='q_otp')

hist_acc_barth(handler=handler, q_list=list(q_.keys()))






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
