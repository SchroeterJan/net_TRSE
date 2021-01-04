from Classes import *

### Variables relating to the nodes
node_vectors = {'bike_clust' :'bike_network_cluster_values',
                'bike_clust_max':'bike_network_cluster_values_max30',
                'PT_clust': 'PT_network_cluster_values',
                'PT_clust_max': 'PT_network_cluster_values_max45',
                'car_clust': 'car_network_cluster_values',
                'car_clust_max': 'car_network_cluster_values_max30',
                'bike_speed_clust': 'bike_speed_cluster_values',
                'euclid_clust': 'euclid_network_cluster_values',
                'Edu_low': 'BEVOPLLAAG_P_2017',
                'Edu_mid': 'BEVOPLMID_P_2017',
                'Edu_high': 'BEVOPLHOOG_P_2017',
                'Pop': 'BEVTOTAAL_2017',
                'SES': 'SKSES_GEM_2017',
                'Income': 'IHHINK_GEM_2017',
                'no Job' : 'PREGWERKL_P_2017',
                'Buurten_size' : 'Buurten_size'
                }


### Variables relating to the edges
edge_vectors = {'Edu_difference': 'Education_difference',
                  'euclid': 'euclidean_distance',
                  'bike_time': 'bike_time_in_seconds',
                  'car_time': 'car_times_in_second',
                  'PT_time': 'minadvice_PT',
                  'bike_speed': 'bike_speed',
                  'PT_speed': 'PT_speed',
                  }


### Class objects
handler = DataHandling()
plotter = Plotting()
analyzer = Analysis()


### Import data to pd DataFrames
df_nodes = handler.Load_Data(node_vectors, isNode=True)
df_edges = handler.Load_Data(edge_vectors)
df_edges_delete = np.array(pickle.load(open('old_scripts/edge_delete_list_popdens.p', 'rb')))

df_edges = df_edges.drop(labels=df_edges_delete)

Buurten = handler.Load_Buurten()
Pop_dens = []
low_dens = []
low_index =[]

for i, each in enumerate(np.array(df_nodes['Pop'])):
    Pop_dens_i = (each/(np.array(df_nodes['Buurten_size'])[i]/1000000))
    Pop_dens.append(Pop_dens_i)
    if Pop_dens_i <= 100:
        low_dens.append(np.array(Buurten['Buurt_code'])[i])
        low_index.append(i)

low_dens= np.array(low_dens)
Pop_dens = np.array(Pop_dens)
df_nodes['Pop_dens'] = Pop_dens

df_nodes = df_nodes.drop(labels=low_dens)
Buurten = Buurten.drop(labels=low_index)



"""
PT_matrix = handler.Build_Matrix(length=len(np.array(df_nodes['Pop'])),data_list=np.array(df_edges['bike_time']))
analyzer.Clustering(matrix=PT_matrix,Buurten=np.array(Buurten['Buurt_code']),name='bike_clustering_new_limit')
"""


df_nodes['bike_cluster_new'] = pickle.load(open('old_scripts/bike_newcluster_values.p', 'rb')).astype(float)
df_nodes['bike_cluster_max_new'] = pickle.load(open('old_scripts/bike_new_maxcluster_values.p', 'rb')).astype(float)
df_nodes['bike_cluster_inverse'] = pickle.load(open('old_scripts/Bike_clustering_new_inversecluster_values.p', 'rb')).astype(float)
df_nodes['bike_cluster_inverse_max'] = pickle.load(open('old_scripts/bike_clustering_iverse_maxcluster_values.p', 'rb')).astype(float)
df_nodes['bike_cluster_new_limit'] = pickle.load(open('old_scripts/bike_clustering_new_limitcluster_values.p', 'rb')).astype(float)

PT_clust_new = pickle.load(open('old_scripts/PT_newcluster_values.p', 'rb')).astype(float)
PT_clust_new = np.where(PT_clust_new == 0.0, np.nan, PT_clust_new)
PT_clust_max_new = pickle.load(open('old_scripts/PT_new_maxcluster_values.p', 'rb')).astype(float)
PT_clust_max_new = np.where(PT_clust_max_new== 0.0, np.nan, PT_clust_max_new)
PT_clust_max_new_limit = pickle.load(open('old_scripts/PT_new_max_newcluster_values.p', 'rb')).astype(float)
PT_clust_max_new_limit = np.where(PT_clust_max_new_limit==0.0, np.nan, PT_clust_max_new_limit)
PT_clust_new_inverse = pickle.load(open('old_scripts/PT_clustering_new_inversecluster_values.p', 'rb')).astype(float)
PT_clust_new_inverse = np.where(PT_clust_new_inverse==0.0, np.nan, PT_clust_new_inverse)
PT_clust_inverse_max = pickle.load(open('old_scripts/PT_clustering_iverse_maxcluster_values.p', 'rb')).astype(float)
PT_clust_inverse_max = np.where(PT_clust_inverse_max==0.0, np.nan, PT_clust_inverse_max)

df_nodes['PT_cluster_max_new'] = PT_clust_max_new
df_nodes['PT_cluster_new'] = PT_clust_new
df_nodes['PT_cluster_new_limit'] = PT_clust_max_new_limit
df_nodes['PT_cluster_inverse'] = PT_clust_new_inverse
df_nodes['PT_cluster_inverse_max'] = PT_clust_inverse_max


### Create CSV for QGIS
"""
Buurten_info_extended = pd.merge(Buurten, df_nodes, on='Buurt_code')
Buurten_info_extended.to_csv(path_or_buf='Buurten_info_inverse.csv', sep=';')
"""
"""
largest_extend= 5
largest = np.argsort(np.array(df_nodes['PT_cluster_new']))[-largest_extend:]
plotter.Scatter(df=df_nodes,y='PT_cluster_new_limit',x='bike_cluster_new',
                xlabel= 'Clustering coefficient for journeys by bike',
                ylabel= 'Clustering coefficient for journeys by \nPublic Transport (maximum 45 Minutes)',
                title = 'Bike vs. Public Transport')

"""
Income_thresholded = handler.Thresholding(variable=np.array(df_nodes['Income'], dtype=float), largest_extend=20)
unemployment_thresholded = handler.Thresholding(variable=np.array(df_nodes['no Job'], dtype=float), largest_extend=20)
"""
#plotter.Correlation_Heatmap(df_nodes)
plotter.MultiScatter(suptitle='Bike vs. Public Transport colored by socioeconomic variables',
                  c=[None, df_nodes['Edu_low'], df_nodes['Edu_mid'], df_nodes['Edu_high'], Income_thresholded, unemployment_thresholded],
                  xtext='Clustering coefficient for journeys by bike', ytext='Clustering coefficient for journeys by \nPublic Transport (maximum 45 Minutes)',
                  x=df_nodes['bike_cluster_new'], y=df_nodes['PT_cluster_new_limit'],
                  axt=['Regular','Low Education', 'Average Education', 'High Education', 'Income', 'Unemployment'], shape=(3,2))
"""


### df_edge delete list popdens
"""
delete_index = []
counter = 0
for row in range(471):
    for column in range(row, 471):
        if row == column:
            continue
        else:
            for index in low_index:
                if column == index:
                    delete_index.append(counter)
                elif row == index:
                    delete_index.append(counter)
            counter += 1

pickle.dump(delete_index, open("edge_delete_list_popdens.p", "wb"))
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

a = 10






####Neglecting Buurts by Population
"""
best = np.array(df_nodes['Pop'])

best_nan = np.isnan(best)
too_small = list(np.where(best_nan == True)[0])

for i, each in enumerate(best):
    if each <= 50.0:
        too_small.append(i)


df_nodes = df_nodes.drop(labels=np.array(too_small))
"""








