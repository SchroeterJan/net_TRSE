from exp_resources import *

import pickle


def exec_skater():
    # mst_plot(data=Skater(variables=variables, handler=handler))
    geo_df = geopandas.GeoDataFrame(crs=crs_proj,
                                    geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    skat = Skater(model=handler.model_, geo_frame=geo_df)
    c = 19

    comps = skat.tree_patitioning(c=c, plot=True)
    pickle.dump(comps, open(os.path.join(path_experiments, 'comps_se.p'), "wb"))

    animate_skater(c)


def plot_skater():
    geo_df = geopandas.GeoDataFrame(crs=crs_proj,
                                    geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    skat = Skater(model=handler.model_, geo_frame=geo_df)
    comps = pickle.load(open(os.path.join(path_experiments, 'comps_se.p'), "rb"))
    comp_stat_index = list(range(1, 20))
    comp_stat_index.append('overall')
    comp_stat = pd.DataFrame(columns=['Compartment',
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
    #                                 'otp_clust' + '_av',
    #                                 'otp_clust' + '_std',
    #                                 'bike_clust' + '_av',
    #                                 'bike_clust' + '_std',
    #                                 'bike_q' + '_av',
    #                                 'bike_q' + '_std',
    #                                 'otp_q' + '_av',
    #                                 'otp_q' + '_std'
                                      ],
                             index=comp_stat_index)

    # comp_stat = pd.DataFrame(columns=['Compartment',
    #                                   '#Vertices',
    #                                   'SSD',
    #                                   'otp_clust' + '_av',
    #                                   'otp_clust' + '_std',
    #                                   'bike_clust' + '_av',
    #                                   'bike_clust' + '_std',
    #                                   'bike_q' + '_av',
    #                                   'bike_q' + '_std',
    #                                   'otp_q' + '_av',
    #                                   'otp_q' + '_std'],
    #                          index=list(range(1, 20)))

    colors = skat.plot_clust_init()
    # skat.ax.set_title('Regionalization of Network Properties for Accessibility',
    #                   fontsize=40)
    skat.ax.set_title('Regionalization by Socioeconomic Variables',
                      fontsize=40)
    for i, comp in enumerate(comps):
        stat_list = [str(i +1)]
        stat_list.append(len(comp.nodes()))
        stat_df = handler.neighborhood_se.loc[list(comp.nodes())]
        comp_df = handler.model_.loc[list(comp.nodes())]
        stat_list.append(round(skat.ssd(k=comp, x=comp_df), 5))
        stat_list.append(round(stat_list[-1]/stat_list[-2], 5))
        for var in reversed(census_variables):
            stat_list.append(round(stat_df[var].mean(axis=0), 2))
            stat_list.append(round(stat_df[var].std(axis=0), 2))

        comp_stat.loc[i+1] = stat_list
        for node in list(comp.nodes()):
            skat.geo_df.at[node, 'clust'] = i + 1
        comp_geo = skat.geo_df.loc[list(comp.nodes())]
        # comp_geo.dissolve().plot(ax=skat.ax, color=colors[i+2])
        comp_geo.plot(ax=skat.ax, color=colors[i +1])
        plt.text(s=str(i+1),
                 x=np.array(comp_geo.dissolve().representative_point()[0].coords.xy)[0],
                 y=np.array(comp_geo.dissolve().representative_point()[0].coords.xy)[1],
                 horizontalalignment='center',
                 fontsize=26,
                 color='k')


    # city_parts = geopandas.read_file(os.path.join(dir_raw, 'stadsdelen all 2022-02-17 11.31.06.geojson'))
    # city_parts.boundary.plot(ax=skat.ax)

    skat.ax.set_axis_off()
    plt.tight_layout()


    # plt.savefig(fname=os.path.join(path_skater, 'se', 'SKATER_se_part_comp'))
    plt.savefig(fname=os.path.join(path_skater, 'se', 'SKATER_se'))
    # plt.savefig(fname=os.path.join(path_skater, 'se', 'SKATER_all'))
    comp_all = ['overall']
    comp_all.extend(round(comp_stat.mean(axis=0), 2))
    comp_stat.loc['mean'] = comp_all
    print(comp_stat.to_latex(index=False))
    a =1


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

# handler.bike[handler.bike > 2400.0] = np.nan
# handler.otp[handler.otp > 2400.0] = np.nan
# straightness_centrality()

# skat_all()

handler.edu_score()
model = census_variables[3:]
model.append('edu_score')
handler.stat_prep(model_variables=model)

# plot_skater(model_variables)
# plot_skater(variables=skat_transp())
plot_skater()


# clust()


