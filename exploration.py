from Classes import *
from config import *
from experiments import *


plt.style.use('seaborn')  # pretty matplotlib plots
plt.rc('font', size=24)
sns.set_theme(style="ticks")


def multi_plot(self, shape, suptitle='', ytext='', xtext=''):
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(30, 15))
    # fig.suptitle(suptitle)
    fig.text(0.5, 0.04, xtext, ha='center', va='center')
    fig.text(0.03, 0.5, ytext, ha='center', va='center', rotation='vertical')
    return fig, axes


def se_year():
    year_list = range(2015, 2021, 1)
    empty_list = []
    full_years = []

    for year in year_list:
        se_prep = SENeighborhoods()
        se_prep.crop_se(year=year)
        se_prep.geo_data = se_prep.geo_data.set_index(keys=column_names['geo_id_col'], drop=False)

        se_prep.filter_areas()

        # keep only relevant socio-economic variables
        for variable in census_variables:
            se_prep.extract_var(var=variable)

        a = pd.DataFrame(se_prep.geo_data)
        a = a.filter(items=census_variables)
        a = a.apply(pd.to_numeric)
        b = a.isna().sum().sum()
        if len(a.columns) < 5:
            print('missing variable')
        else:
            full_years.append(int(year))
            empty_list.append(b)

    fig, ax = plt.subplots()
    bars = ax.bar(x=full_years, height=empty_list, align='center', tick_label=full_years)
    ax.set_title('Missing Census Data Points per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Missing Data Points among Variables')
    ax.bar_label(bars, label_type='center')
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
        f, ax = plt.subplots(figsize=(7, 5))
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
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)
        sns.histplot(data=handler.neighborhood_se, x=variable + '_scaled')
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


se_year()
# hist_modes()
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
