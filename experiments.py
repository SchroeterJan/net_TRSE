from Classes import *
from processors import *
from config import *

# Class objects
handler = DataHandling()
analyzer = Analysis(handler=handler)

path_experiments = os.path.join(path_repo, 'experiment_data')
if not os.path.isdir(path_experiments):
    os.mkdir(path_experiments)

path_stopflows = os.path.join(path_experiments, 'stop_flows.csv')

# get total in- and outbound passenger flow for each stop
def stop_flows():
    # follow the data preparation steps up to the assignment algorithm
    flow_prep = PassengerCounts()
    #flow_prep.area_stop_matching()
    flow_prep.filter_connections()

    # cast flows to integer, "coerce" sets non integer values to nan
    flow_prep.pass_data[column_names['pass_vol']] = pd.to_numeric(flow_prep.pass_data[column_names['pass_vol']],
                                                                  errors='coerce')
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


def difference_matrix(vector):
    edges = []
    for i, Buurt_score in enumerate(vector):
        for j, Buurt_score2 in enumerate(vector[i + 1:]):
            edges.append(np.absolute(Buurt_score - Buurt_score2))
    return np.array(edges)


def se_matrices():
    builder = TransportPrep()
    for variable in census_variables:
        matrix = difference_matrix(handler.neighborhood_se[variable])
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
    clusters.to_csv(path_or_buf=path_clustercoeff, sep=';')


def cluster_rel():
    unused = np.where(np.isnan(handler.flows.to_numpy()))
    clust_pt = np.array(handler.pt)
    clust_pt[unused[0], unused[1]] = 0.0
    clust_bike = np.array(handler.bike)
    clust_bike[unused[0], unused[1]] = 0.0
    clusters['pt_rel'] = analyzer.clustering(matrix=clust_pt)
    clusters['bike_rel'] = analyzer.clustering(matrix=clust_bike)
    clusters.to_csv(path_or_buf=path_clustercoeff, sep=';', index=False)


path_clustercoeff = os.path.join(path_generated, 'neighborhood_clustercoeff.csv')


def get_cluster():
    if os.path.isfile(path_clustercoeff):
        clusters = pd.read_csv(filepath_or_buffer=path_clustercoeff, sep=';')
    else:
        clusters = pd.DataFrame(index=handler.neighborhood_se.index)
        cluster_all()
        cluster_rel()
    return clusters

# stop_flows()
se_matrices()
clusters = get_cluster()













# plotter.Scatter(df=Buurten_cluster,y='PT_cluster',x='Bike_cluster',
#                 xlabel= 'Clustering coefficient for journeys by bike',
#                 ylabel= 'Clustering coefficient for journeys by \nPublic Transport',
#                 title = 'Bike vs. Public Transport')



# Income_thresholded = handler.Thresholding(variable=np.array(df_nodes['Income'], dtype=float), largest_extend=20)
# unemployment_thresholded = handler.Thresholding(variable=np.array(df_nodes['no Job'], dtype=float), largest_extend=20)

# plotter.Correlation_Heatmap(df_nodes)
# plotter.MultiScatter(suptitle='Bike vs. Public Transport colored by socioeconomic variables',
#                   c=[None, df_nodes['Edu_low'], df_nodes['Edu_mid'], df_nodes['Edu_high'], Income_thresholded, unemployment_thresholded],
#                   xtext='Clustering coefficient for journeys by bike', ytext='Clustering coefficient for journeys by \nPublic Transport (maximum 45 Minutes)',
#                   x=df_nodes['bike_cluster_new'], y=df_nodes['PT_cluster_new_limit'],
#                   axt=['Regular','Low Education', 'Average Education', 'High Education', 'Income', 'Unemployment'], shape=(3,2))





"""
#### Thresholding speeds
df_edges['PT_speed_new'] = handler.MatrixThresholding(variable = df_edges['PT_speed'], extend=500, length=len(np.array(df_nodes['Pop']))).flatten()
df_edges['bike_speed_new'] = handler.MatrixThresholding(variable=df_edges['bike_speed'], extend=500, length=len(np.array(df_nodes['Pop']))).flatten()


Income_difference = handler.DifferenceMatrix(vector=np.array(df_nodes['Income']))
df_edges['Income_difference'] = Income_difference
SES_difference = handler.DifferenceMatrix(vector=np.array(df_nodes['SES']))
df_edges['SES_difference'] = SES_difference
No_Job_difference = handler.DifferenceMatrix(vector=np.array(df_nodes['no Job']))
df_edges['No_Job_difference'] = No_Job_difference

df_edges['euclid'] = (df_edges['euclid']*1000)


plotter.Heatscatter(x=df_edges['PT_speed_new'], y=df_edges['Edu_difference'], log=False,av=True,
                       xlabel='Overcome distance in Meter per Minute by Public Transport',
                       ylabel='Absolute difference in Education score',
                       title='')


#plotter.Scatter(df_edges, y='Income_difference', x='Bike_speed_new')

plotter.MultiHeatScatter(suptitle='Differences in social variables vs. journey speed by Bike',
                  xtext='Overcome Meters per Minute by Bike', ytext='Absolute difference in socio-economic variable',
                  x=np.array(df_edges['bike_speed_new']), y=[Income_difference, SES_difference, np.array(df_edges['Edu_difference']), No_Job_difference],
                  axt=['Average Income', 'SES', 'Education', 'No Job'], shape=(2,2))


plotter.Heatscatter(x=np.array(df_edges['PT_speed_new']), y=np.array(df_edges['euclid']),xlabel='PT speed',ylabel='euclid')
"""