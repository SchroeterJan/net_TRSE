from exp_resources import *

from scipy.optimize import curve_fit

from scipy.stats import gamma


def exec_skater(variables, handler):
    # mst_plot(data=Skater(variables=variables))
    skat = Skater(variables=variables, handler=handler)
    c = 15

    skat.tree_patitioning(c=c, plot=True)
    animate_skater(c)


def velocity():
    handler.get_q_ij()
    handler.bike_qij = reject_outliers(flatten(handler.bike_qij), m=12., nan_or_zero='zero')
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
    #handler.mix_otp_bike()
    handler.get_q_ij()

    # handler.bike_qij[handler.bike > 2400.0] = 0.0
    # handler.otp_qij[handler.otp > 3600.0] = 0.0
    handler.get_q()

    hist_straight(data=handler.neighborhood_se, modes=['bike_q', 'otp_q'])

    geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
                                       geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    straight_map(data=geo_frame, column='otp_q', mode='Public Transport')
    straight_map(data=geo_frame, column='bike_q', mode='Bike')


def clust(calc=False):
    if calc:
        handler.initiate_graph()
        handler.add_edges(mode='pt')

        cluster_dict = nx.clustering(handler.graph, weight='weight')
        handler.neighborhood_se['otp_clust'] = np.array(list(cluster_dict.values()))

        handler.initiate_graph()
        handler.add_edges(mode='bike')
        cluster_dict = nx.clustering(handler.graph, weight='weight')
        handler.neighborhood_se['bike_clust'] = np.array(list(cluster_dict.values()))
    else:
        hist_clust(data=handler.neighborhood_se, modes=['bike_clust', 'otp_clust'])

        geo_frame = geopandas.GeoDataFrame(data=handler.neighborhood_se,
                                           geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
        clust_map(data=geo_frame, column='otp_clust', mode='Public Transport')
        clust_map(data=geo_frame, column='bike_clust', mode='Bike')


def investigate_flows():
    bin_heights, bin_borders, _ = plt.hist(flatten(handler.euclid), bins='auto')
    bin_widths = np.diff(bin_borders)
    plt.close()
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    y1 = bin_heights / np.trapz(bin_centers, bin_heights)

    fit_alpha, fit_loc, fit_beta = gamma.fit(flatten(handler.euclid))
    x = np.linspace(gamma.ppf(0.01, fit_alpha), gamma.ppf(0.99, fit_alpha), 100)
    rv = gamma(fit_alpha)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    #plt.bar(bin_centers, -y1, width=bin_widths, label='histogram')
    #ax.hist(flatten(handler.euclid), density=True, histtype='stepfilled', alpha=0.2)
    plt.show()


    a = 1.99
    mean, var, skew, kurt = gamma.stats(a, moments='mvsk')
    x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), 100)


    vals = gamma.ppf([0.001, 0.5, 0.999], a)
    print(np.allclose([0.001, 0.5, 0.999], gamma.cdf(vals, a)))
    plt.bar(bin_centers, -y1, width=bin_widths, label='histogram')
    plt.show()
    a = 10







    params, _ = curve_fit(lognorm, bin_centers, -y1)
    #x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 20000)
    plt.figure()
    plt.bar(bin_centers, -y1, width=bin_widths, label='histogram')
    #plt.plot(bin_centers, -y1, 'o')
    plt.plot(bin_centers, lognorm(bin_centers, params[0], params[1]), label='fit', c='red')
    plt.legend()
    plt.show()
    a = 10


    total_flows = np.triu(np.triu(handler.flows.values, k=0) + np.tril(handler.flows.values, k=0).T)
    reached = total_flows > 0.0
    reached[handler.bike < 600] = True
    dist_flows = handler.euclid[reached]
    a = 10


def skat_all():
    for var in scaling_variables:
        handler.scale(var)
    handler.scale('otp_clust')
    handler.scale('bike_clust')
    handler.scale('bike_q')
    handler.scale('otp_q')

    model_variables.extend(['otp_clust_scaled',
                           'bike_clust_scaled',
                           'bike_q_scaled',
                           'otp_q_scaled'])




# handler = DataHandling()
# handler.matrices()
# velocity()
# straightness_centrality()
# clust(calc=True)
# handler.neighborhood_se.to_csv(os.path.join(path_experiments, file_neighborhood_se))
# handler = DataHandling(new=True)
# handler.matrices()
#
# clust()

handler = DataHandling(new=True)
handler.matrices()

for var in scaling_variables:
    handler.scale(var)

exec_skater(variables=model_variables, handler=handler)

handler.bike[handler.bike > 2400.0] = np.nan
handler.otp[handler.otp > 2400.0] = np.nan
straightness_centrality()
# clust()

skat_all()
exec_skater(variables=model_variables, handler=handler)

a = 0


investigate_flows()


