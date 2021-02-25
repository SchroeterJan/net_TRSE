from Classes import *
from config import *

### Class objects
handler = DataHandling()
plotter = Plotting()
analyzer = Analysis(handler=handler)


path_clustercoeff = os.path.join(path_repo, path_generated, 'neighborhood_clustercoeff.csv')


### CLUSTERING
def cluster_all():
    clusters = pd.DataFrame(index=handler.neighborhood_se.index)
    clusters['pt_all'] = analyzer.Clustering(matrix=handler.pt.to_numpy())
    clusters['bike_all'] = analyzer.Clustering(matrix=handler.bike.to_numpy())
    clusters['flows_all'] = analyzer.Clustering(matrix=handler.flows.to_numpy())
    clusters.to_csv(path_or_buf=path_clustercoeff, sep=';')


def cluster_rel():
    unused = np.where(np.isnan(handler.flows.to_numpy()))
    handler.pt = np.array(handler.pt)
    handler.pt[unused[0], unused[1]] = 0.0
    handler.bike = np.array(handler.bike)
    handler.bike[unused[0], unused[1]] = 0.0
    cluster['pt_rel'] = analyzer.Clustering(matrix=handler.pt)
    cluster['bike_rel'] = analyzer.Clustering(matrix=handler.bike)
    cluster.to_csv(path_or_buf=path_clustercoeff, sep=';', index=False)




if os.path.isfile(path_clustercoeff):
    cluster = pd.read_csv(filepath_or_buffer=path_clustercoeff, sep=';')
else:
    cluster_all()
    cluster_rel()








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