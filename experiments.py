from exp_resources import *

from libpysal import weights
from sklearn.metrics import pairwise as skm
from spopt.region import skater as skat_lib
from scipy import spatial
import warnings


import pickle


def investigate(self, data):

    frame = pd.DataFrame(data,
                         columns=handler.neighborhood_se.index,
                         index=handler.neighborhood_se.index)
    ij_mat = []
    for i, n_i in enumerate(self.comps):
        i_times = frame.loc[list(n_i.nodes())]
        ij_list = []
        for j, n_j in enumerate(self.comps):
            if j == i:
                ij_list.append(np.nan)
            else:
                ij_list.append(i_times.loc[:, list(n_j.nodes())].mean().mean())
        ij_mat.append(ij_list)
        a = 1


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


def skater_clust():
    skat_help = Skater_Helper(geo_frame=geo_df, variables=model)
    skat_help.adjacency_graph()

    skater_w = weights.Queen.from_networkx(graph=skat_help.adj_g)

    spanning_forest_kwds = dict(dissimilarity=diss,
                                affinity=None,
                                reduction=np.nansum,
                                center=np.nanmean
                                )

    skat_calc = skat_lib.Skater(gdf=geo_df,
                                w=skater_w,
                                attrs_name=model,
                                n_clusters=no_compartments,
                                floor=1,
                                trace=False,
                                islands="increase",
                                spanning_forest_kwds=spanning_forest_kwds)

    # I expect to see RuntimeWarnings in this block for mean of empty slice np.nanmean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skat_calc.solve()

    skat_stats(skat_result=skat_calc)
    skat_plot(data=skat_calc)


def skat_stats(skat_result):
    skat_stat_index = list(range(len(np.unique(skat_result.labels_))))
    skat_stat_index.append('overall')
    skat_stat = pd.DataFrame(columns=['Compartment',
                                      '#Vertices',
                                      'SSD',
                                      'SSD/Vertice',
                                      census_variables[0] + '_av',
                                      census_variables[0] + '_std',
                                      census_variables[1] + '_av',
                                      census_variables[1] + '_std',
                                      census_variables[2] + '_av',
                                      census_variables[2] + '_std',
                                      census_variables[3] + '_av',
                                      census_variables[3] + '_std',
                                      census_variables[4] + '_av',
                                      census_variables[4] + '_std',
                                      # 'otp_clust' + '_av',
                                      # 'otp_clust' + '_std',
                                      # 'bike_clust' + '_av',
                                      # 'bike_clust' + '_std',
                                      # 'bike_q' + '_av',
                                      # 'bike_q' + '_std',
                                      # 'otp_q' + '_av',
                                      # 'otp_q' + '_std'
                                      ],
                             index=skat_stat_index)

    geo_df['skater_new'] = skat_result.labels_
    geo_df['number'] = 1
    skat_stat['#Vertices'] = geo_df[['skater_new', 'number']].groupby(by='skater_new').count()

    for comp_no in range(len(np.unique(skat_result.labels_))):
        comp_data = geo_df[model][skat_result.labels_ == comp_no]
        or_data = handler.neighborhood_se[census_variables].reset_index(drop=True)[skat_result.labels_ == comp_no]
        skat_stat.at[comp_no, 'Compartment'] = comp_no + 1
        spanning = skat_lib.SpanningForest(**skat_result.spanning_forest_kwds)
        skat_stat.at[comp_no, 'SSD'] = spanning.score(data=comp_data, labels=np.zeros(len(comp_data)))
        skat_stat.at[comp_no, 'SSD/Vertice'] = round(skat_stat['SSD'][comp_no] / skat_stat['#Vertices'][comp_no], 5)
        for c_var in reversed(census_variables):
            skat_stat.at[comp_no, c_var + '_av'] = round(or_data[c_var].mean(), 2)
            skat_stat.at[comp_no, c_var + '_std'] = round(or_data[c_var].std(), 2)

    skat_stat.loc['overall'] = round(skat_stat.mean(axis=0), 2)
    print(skat_stat.to_latex(index=False))


def diss(X, Y=None):
    if Y is None:
        return spatial.distance.squareform(spatial.distance.pdist(X))
    else:
        return (X - Y) ** 2


handler = DataHandling(new=True)
handler.matrices()

# velocity()
# straightness_centrality()
# clust(calc=True)
# handler.neighborhood_se.to_csv(os.path.join(path_experiments, file_neighborhood_se))
# handler = DataHandling(new=True)
# handler.matrices()
#
# clust()


# handler.bike[handler.bike > 2400.0] = np.nan
# handler.otp[handler.otp > 2400.0] = np.nan
# straightness_centrality()

# skat_all()

handler.edu_score()
model = census_variables[3:]
model.append('edu_score')
handler.stat_prep(model_variables=model)


#comps_se = pickle.load(open(os.path.join(path_experiments, 'comps_se.p'), "rb"))
no_compartments = 25
geo_df = geopandas.GeoDataFrame(data=handler.model_,
                                crs=crs_proj,
                                geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))

geo_df.reset_index(inplace=True, drop=True)

skater_clust()
