from Classes import *
from config import *
from experiments import *


plt.style.use('seaborn')  # pretty matplotlib plots
plt.rc('font', size=24)
sns.set_theme(style="ticks")

# handler = DataHandling()
# plotter = Plotting()

path_plot = os.path.join(path_repo, 'plots')
if not os.path.isdir(path_plot):
    os.mkdir(path_plot)
path_hists = os.path.join(path_plot, 'hists')
if not os.path.isdir(path_hists):
    os.mkdir(path_hists)
path_explore = os.path.join(path_plot, 'explore')
if not os.path.isdir(path_explore):
    os.mkdir(path_explore)
path_maps = os.path.join(path_plot, 'maps')
if not os.path.isdir(path_maps):
    os.mkdir(path_maps)

travel_times = ['Bike', 'Public Transport']


def se_year():
    year_list = range(2015, 2021, 1)
    empty_list = []

    for year in year_list:
        se_prep = SENeighborhoods()
        se_prep.crop_se(year=year)
        se_prep.geo_data = se_prep.geo_data.set_index(keys=column_names['geo_id_col'], drop=False)

        # keep only relevant socio-economic variables
        for variable in census_variables:
            se_prep.extract_var(var=variable)

        se_prep.filter_areas()
        a = se_prep.geo_data.filter(items=census_variables)
        b = a.isna().sum().sum()
        empty_list.append(b)

    plt.figure()
    plt.bar(x=year_list, height=empty_list)
    plt.title('Missing Census Data Points per Year')
    plt.xlabel('Year')
    plt.ylabel('Total Missing Data Points among Variables')
    plt.savefig(fname=os.path.join(path_explore, 'missing_data'))
    plt.close()


def hist_modes():
    edges_flat = pd.DataFrame(columns=travel_times)
    edges_flat[travel_times[0]] = handler.bike.values.flatten()
    edges_flat[travel_times[1]] = handler.pt.values.flatten()

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    colors = {'Bike': 'red', 'Public Transport': 'blue'}

    for mode in travel_times:
        sns.histplot(data=edges_flat, x=mode, binwidth=60, color=colors[mode], label=mode, alpha=0.5)
    ax.set_title('Travel time histogram')
    ax.margins(x=0)
    plt.tight_layout()
    plt.legend()

    plt.savefig(fname=os.path.join(path_hists, 'travel_times'))
    plt.close(f)


def hist_flows():
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    sns.histplot(data=pd.DataFrame(data=handler.flows.values.flatten(), columns=['values']), log_scale=True)
    ax.set_title('Passenger flow histogram')
    ax.margins(x=0)
    plt.tight_layout()
    plt.xlabel('Travellers on between two areas')
    plt.savefig(fname=os.path.join(path_hists, 'flow'))
    plt.close(f)


def hist_se():
    for variable in census_variables:
        f, ax = plt.subplots(figsize=(15, 9))
        sns.despine(f)

        handler.neighborhood_se[variable] = handler.neighborhood_se[variable].replace(to_replace=0.0, value=np.nan)
        sns.histplot(data=handler.neighborhood_se, x=variable)
        ax.set_title('Histogram of ' + variable)
        ax.margins(x=0)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hists, variable + '_hist'))
        plt.close(f)


def hist_scaled_se():
    for variable in scaling_variables:
        f, ax = plt.subplots(figsize=(15, 9))
        sns.despine(f)

        handler.neighborhood_se[variable + '_scaled'] = handler.neighborhood_se[variable + '_scaled'].replace(
            to_replace=0.0, value=np.nan)
        sns.histplot(data=handler.neighborhood_se, x=variable)
        ax.set_title('Histogram of ' + variable + '_scaled')
        ax.margins(x=0)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hists, variable + '_scaled_hist'))
        plt.close(f)


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
                                       geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometrie))

    fig, axes = plotter.multi_plot(shape=[2, 3])
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
    skat.geo_df.plot(ax=ax, column='clust')
    # nx.drawing.nx_pylab.draw_networkx_edges(G=mst_graph, pos=skat.pos, ax=ax)

    for component in comp:
        nx.drawing.nx_pylab.draw_networkx_edges(G=component, pos=skat.pos, ax=ax)
    plt.show()
    for i in range(10):
        a = list(comp[i].nodes())
        for node in a:
            skat.geo_df.at[node, 'clust'] = i


    a = 10



# hist_modes()
# hist_flows()
# hist_scaled_se()

# clusters = get_cluster()
# hist_cluster()

# geo_plot()
# plot_se_kmean()
plot_adj_mat()
