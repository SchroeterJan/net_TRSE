from plotting.plots import *


travel_times = ['Bike', 'Public Transport']


def se_year():
    year_list = range(2015, 2021, 1)
    table = np.empty(shape=(len(year_list), len(census_variables)))

    for i, year in enumerate(year_list):
        se_prep = SENeighborhoods()
        se_prep.crop_se(year=year)
        se_prep.geo_data = se_prep.geo_data.set_index(keys=column_names['geo_id_col'], drop=False)

        se_prep.filter_areas()

        # keep only relevant socio-economic variables
        for variable in census_variables:
            se_prep.extract_var(var=variable)

        missing = pd.DataFrame(se_prep.geo_data)
        missing = missing.filter(items=census_variables)
        missing = missing.apply(pd.to_numeric)
        table[i] = missing.isna().sum()

    frame = pd.DataFrame(data=table, index=year_list, columns=census_variables)
    frame['Total'] = frame.sum(axis=1)
    print(frame.to_latex(index=True))
    # se_year_miss(data=full_years, height=empty_list, labels=full_years)


def filter_shorttrips():
    walked = handler.reduce_matrix(frame=handler.euclid)
    walked = np.where(walked < short_trip)
    walked_graph = nx.Graph()
    index_list = list(handler.neighborhood_se.index)
    reduced_otp = handler.reduce_matrix(frame=handler.otp)
    reduced_pt = handler.reduce_matrix(frame=handler.pt)
    reduced_bike = handler.reduce_matrix(frame=handler.bike)
    for i, or_ind in enumerate(walked[0]):
        dest_ind = walked[1][i]
        walked_graph.add_edge(u_of_edge=index_list[or_ind], v_of_edge=index_list[dest_ind])
        reduced_pt[or_ind][dest_ind] = np.nan
        reduced_otp[or_ind][dest_ind] = np.nan
        reduced_bike[or_ind][dest_ind] = np.nan

    reduced_otp = np.floor(reduced_otp/60.)*60.
    time_frame = pd.concat([pd.DataFrame(reduced_otp[~np.isnan(reduced_otp)], columns=['otp']),
                            pd.DataFrame(reduced_pt[~np.isnan(reduced_pt)], columns=['pt'])], axis=1)

    comp_hist(frame=time_frame, colors=['red', 'blue'])
    plt.hist(reduced_pt[~np.isnan(reduced_pt)], bins=50, fc=(0, 0, 1, 0.5))
    # floor_diff = abs(reduced_otp-reduced_pt)
    # floor_diff_pt = floor_diff/reduced_pt
    # floor_diff_pt = floor_diff_pt[~np.isnan(floor_diff_pt)]
    # floor_diff_otp = floor_diff/reduced_otp
    # floor_diff_otp = floor_diff_otp[~np.isnan(floor_diff_otp)]
    # plt.hist(floor_diff_pt, bins=70, range=(0, 1))
    # plt.tight_layout()
    # plt.savefig(fname=os.path.join(path_explore, 'pt_diff3'))
    # plt.close()
    # plt.hist(floor_diff_otp, bins=70, range=(0, 1))
    # plt.tight_layout()
    # plt.savefig(fname=os.path.join(path_explore, 'pt_diff4'))
    # plt.close()

    a = 10

    geo_df = geopandas.GeoDataFrame(crs=crs_proj,
                                    geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    geo_net_plot(geo_frame=geo_df, graph=walked_graph)
    a=1


def hist_cluster():
    cluster_list = {'pt_all': 'blue',

                    'pt_rel': 'red'
                    }

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    clusters = pd.read_csv(filepath_or_buffer=path_clustercoeff, sep=';')
    for cluster in cluster_list:
        sns.histplot(data=clusters, x=cluster, color=cluster_list[cluster], binwidth=0.025, label=cluster, alpha=0.6)
    ax.set_title('Clustering coefficient for Public Transport')
    plt.xlabel('Clustering coefficient')
    ax.margins(x=0)
    plt.tight_layout()
    plt.legend()
    plt.savefig(fname=os.path.join(path_hists, 'cluster_hist'))
    plt.close(f)


def geo_plot():
    geo_frame = geopandas.GeoDataFrame(crs="EPSG:4326",
                                       geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))

    fig, axes = multi_plot(shape=[2, 3])
    axes = axes.flatten()
    reduce_se_variables()
    for i, each in enumerate(census_variables):
        if each == 'IHHINK_GEM':
            handler.neighborhood_se[handler.neighborhood_se[each] > 100000.0] = 100000.0
        elif each == 'PREGWERKL_P':
            handler.neighborhood_se[handler.neighborhood_se[each] > 30.0] = 30.0

        geo_frame[each] = handler.neighborhood_se[each]
        geo_frame.plot(ax=axes[i], column=each, legend=True, cmap='OrRd')
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_title(label=each)
    plt.savefig(fname=os.path.join(path_maps, 'se_variables_maps'))


def plot_se_kmean():
    geo_frame = se_kmean()

    fig, ax = plt.subplots(figsize=(20, 15))
    geo_frame.plot(column='clust', legend=True, cmap='viridis_r', ax=ax)
    ax.set_title('KMean Cluster for Socio-economic data')
    plt.savefig(fname=os.path.join(path_maps, 'kmean_cluster'))


def plot_adj_mat():
    skat = Skater()
    comp = skat.tree_patitioning()

    fig, ax = plt.subplots(figsize=(20, 15))
    skat.geo_df.plot(ax=ax)
    # nx.drawing.nx_pylab.draw_networkx_edges(G=comp, pos=skat.pos, ax=ax)

    for component in comp:
        nx.drawing.nx_pylab.draw_networkx_edges(G=component, pos=skat.pos, ax=ax)
    plt.show()
    plt.close(fig=fig)
    for i in range(16):
        a = list(comp[i].nodes())
        for node in a:
            skat.geo_df.at[node, 'clust'] = i
    fig, ax = plt.subplots(figsize=(20, 15))
    skat.geo_df.plot(ax=ax, column='clust')
    plt.show()
    a =10


# se_year()
filter_shorttrips()




# hist_modes(travel_times)
# hist_flows()
# hist_se()
# hist_scaled_se()

# clusters = get_cluster()
# hist_cluster()

# geo_plot()
# plot_se_kmean()
plot_adj_mat()


def test():
    geo_frame = geopandas.GeoDataFrame(crs="EPSG:4326",
                                       geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    geo_frame['pop'] = handler.neighborhood_se['BEVTOTAAL']
    geo_frame['empty'] = pd.isnull(geo_frame['pop'])
    geo_frame.plot(aspect=1, column='empty')
    a=1


#test()
