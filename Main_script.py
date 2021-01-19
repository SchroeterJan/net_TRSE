from Classes import *

"""
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
"""

### Class objects
handler = DataHandling()
plotter = Plotting()
analyzer = Analysis(handler=handler)

#PT_cluster = analyzer.Clustering(matrix=np.array(handler.PT_times))
#Bike_cluster = analyzer.Clustering(matrix=np.array(handler.Bike_times))

Buurten_cluster = pd.read_csv('Generated_data/Buurten_cluster.csv', sep=';')
Buurten_cluster = Buurten_cluster.replace([0], np.nan)




plotter.Scatter(df=Buurten_cluster,y='PT_cluster',x='Bike_cluster',
                xlabel= 'Clustering coefficient for journeys by bike',
                ylabel= 'Clustering coefficient for journeys by \nPublic Transport (maximum 45 Minutes)',
                title = 'Bike vs. Public Transport')



Income_thresholded = handler.Thresholding(variable=np.array(df_nodes['Income'], dtype=float), largest_extend=20)
#unemployment_thresholded = handler.Thresholding(variable=np.array(df_nodes['no Job'], dtype=float), largest_extend=20)


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














