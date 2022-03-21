from resources.exp_resources import *


def velocity():
    handler.get_q_ij()
    handler.bike_qij = reject_outliers(flatten(handler.bike_qij), m=12.)
    handler.otp_qij = flatten(handler.otp_qij)

    heatscatter(x=flatten(handler.bike),
                y=flatten(handler.euclid),
                xlabel='Travel time by Bike in ' + r'$s$',
                ylabel='Euclidean distance in ' + r'$m$',
                title='Bike',
                av=True)

    heatscatter(x=flatten(handler.otp),
                y=flatten(handler.euclid),
                xlabel='Travel time by Public Transport in ' + r'$s$',
                ylabel='Euclidean distance in ' + r'$m$',
                title='Public Transport',
                av=True)

    hist_qij(handler=handler, travel_times=travel_times)


def straightness_centrality():
    print('calculate velocities')
    handler.get_q_ij()
    print('calculate modal efficiency')
    handler.get_q()

    hist_straight(data=handler.neighborhood_se, modes=['bike_q', 'otp_q'])

    geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
                                       geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    straight_map(data=geo_frame, column='otp_q', mode='Public Transport')
    straight_map(data=geo_frame, column='bike_q', mode='Bike')


def clust_coeff(calc=False):
    if calc:
        handler.initiate_graph()
        handler.add_edges(mode='pt')
        print('calculate clustering coefficients for public transport')
        cluster_dict = nx.clustering(handler.graph, weight='weight')
        handler.neighborhood_se['otp_clust'] = np.array(list(cluster_dict.values()))

        handler.initiate_graph()
        handler.add_edges(mode='bike')
        print('calculate clustering coefficients for bike')
        cluster_dict = nx.clustering(handler.graph, weight='weight')
        handler.neighborhood_se['bike_clust'] = np.array(list(cluster_dict.values()))
    else:
        hist_clust(data=handler.neighborhood_se, modes=['bike_clust', 'otp_clust'])

        geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
                                           geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
        clust_map(data=geo_frame, column='otp_clust', mode='Public Transport')
        clust_map(data=geo_frame, column='bike_clust', mode='Bike')


handler = DataHandling()
handler.matrices()

# velocity()
straightness_centrality()
clust_coeff(calc=True)
handler.neighborhood_se.to_csv(os.path.join(path_experiments, file_neighborhood_se))
# handler = DataHandling(new=True)
# handler.matrices()
#
# clust()
a = 1

# handler.bike[handler.bike > 2400.0] = np.nan
# handler.otp[handler.otp > 2400.0] = np.nan
# straightness_centrality()