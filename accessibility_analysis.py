from resources.exp_resources import *


def modal_efficiency():
    handler.get_q_ij()
    handler.bike_qij = reject_outliers(handler.bike_qij.flatten()[~np.isnan(handler.bike_qij.flatten())], m=12.)
    handler.otp_qij = handler.otp_qij.flatten()[~np.isnan(handler.otp_qij.flatten())]

    hist_qij(handler=handler, travel_times=travel_times)

    heatscatter(x=handler.bike.flatten()[~np.isnan(handler.bike.flatten())],
                y=handler.euclid.flatten()[~np.isnan(handler.euclid.flatten())],
                xlabel='Travel time by Bike in ' + r'$s$',
                ylabel='Euclidean distance in ' + r'$m$',
                title='Bike',
                av=True)

    heatscatter(x=handler.otp.flatten()[~np.isnan(handler.otp.flatten())],
                y=handler.euclid.flatten()[~np.isnan(handler.euclid.flatten())],
                xlabel='Travel time by Public Transport in ' + r'$s$',
                ylabel='Euclidean distance in ' + r'$m$',
                title='Public Transport',
                av=True)


def sme():
    print('calculate velocities')
    handler.get_q_ij()
    print('calculate modal efficiency')
    handler.get_q()

    geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
                                       geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    sme_map(data=geo_frame, column='otp_q', mode='Public Transport')
    sme_map(data=geo_frame, column='bike_q', mode='Bike')


def clust_coeff():
    handler.initiate_graph()
    handler.pt
    print('calculate clustering coefficients for public transport')
    cluster_dict = nx.clustering(handler.graph, weight='weight')
    handler.neighborhood_se[clust_pt] = np.array(list(cluster_dict.values()))

    handler.initiate_graph()
    handler.bike
    print('calculate clustering coefficients for bike')
    cluster_dict = nx.clustering(handler.graph, weight='weight')
    handler.neighborhood_se[clust_bike] = np.array(list(cluster_dict.values()))


# handler = DataHandling()
# handler.matrices()

# modal_efficiency()

# handler.bike[handler.bike > 2400.0] = np.nan
# handler.otp[handler.otp > 2700.0] = np.nan

# sme()
# clust_coeff()
# handler.neighborhood_se.to_csv(os.path.join(path_experiments, file_neighborhood_se))
handler = DataHandling(new=True)
handler.matrices()

hist_acc(data=handler.neighborhood_se,
        modes=[sme_bike, sme_pt],
        mode_names=[sme_bike_name, sme_pt_name],
        title='Distributions of Summed Modal Efficiencies',
        file='sme')
hist_acc(data=handler.neighborhood_se,
         modes=[clust_bike, clust_pt],
         mode_names=[clust_bike_name, clust_pt_name],
         title='Distributions of Weighted Clustering Coefficients',
         file='clust')



geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
                                   geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
clust_map(data=geo_frame, column=clust_pt, mode='Public Transport')
clust_map(data=geo_frame, column=clust_bike, mode='Bike')
clust_map(data=geo_frame, column=clust_bike, mode='Bike', circles=True)


