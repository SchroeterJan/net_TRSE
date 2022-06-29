from plotting.plot_functions import *
import matplotlib.image as mgimg
from matplotlib import animation
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import matplotlib as mpl


plt.style.use('seaborn')  # pretty matplotlib plots
plt.rc('font', size=24)
sns.set_theme(style="ticks")


def hist_modes(handler, travel_times):
    time_frame = pd.DataFrame(columns=travel_times)
    time_frame[travel_times[0]] = handler.bike.flatten()
    time_frame[travel_times[1]] = handler.otp.flatten()

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    colors = ['red', 'blue']

    comp_hist(frame=time_frame, colors=colors, binw=60, ax=ax)

    ax.set_title('Travel time histogram')
    ax.set_xlabel('Time in seconds')
    ax.margins(x=0)
    plt.tight_layout()
    plt.legend()

    plt.savefig(fname=os.path.join(path_hists, 'travel_times'))
    plt.close(f)


def hist_flows(handler):
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    data = pd.DataFrame(data=handler.flows.flatten(),
                        columns=['values'])
    sns.histplot(data, log_scale=True, legend=False)
    meanline(data=data, variable='values', ax=ax)
    ax.set_title('Passenger flow histogram')
    ax.margins(x=0)
    plt.xlabel('Passengers between two areas')
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_hists, 'flow'))
    plt.close(f)


def explore_se(data):
    geo_frame = geopandas.GeoDataFrame(data=data,
                                       geometry=geopandas.GeoSeries.from_wkt(data.geometry))
    for i, variable in enumerate(census_variables):
        f, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [2, 1]})
        sns.despine(f)
        data[variable] = data[variable].replace(to_replace=0.0, value=np.nan)
        # ax[i].hist(x=handler.neighborhood_se[variable])

        vmax = np.nanmean(geo_frame[variable].values) + 3 * np.nanstd(geo_frame[variable].values)
        if len(geo_frame[variable].values[geo_frame[variable].values > vmax]) == 0:
            vmax=None

        geo_plot(frame=geo_frame,
                 axis=ax[0],
                 column=variable,
                 vmax=vmax
                 )
        ax[0].set_title('Spatial Arrangement of ' + census_names[i])
        sns.histplot(data=data, x=variable, ax=ax[1])
        meanline(data=data, variable=variable, x=i + 1, ax=ax[1])
        ax[1].set_xlabel(census_names[i] + ' in %')
        # ax.margins(x=0)
        ax[1].set_title('Distribution of ' + census_names[i])
        plt.xlabel(census_names[i] + ' in %')
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hist_se, variable))
        plt.close(f)


# def skat_comp(data, mode, title, filename):
#     f, ax = plt.subplots(nrows=len(census_variables), ncols=1, figsize=(7, len(census_variables) * 5))
#     sns.despine(f)
#     for i, variable in enumerate(census_variables):
#         # data[variable] = data[variable].replace(to_replace=0.0, value=np.nan)
#         # ax[i].hist(x=handler.neighborhood_se[variable])
#         ax[i].scatter(data[variable + '_av'], data[mode + '_av'])
#         # meanline(data=data, variable=variable, x=i + 1, ax=ax[i])
#         ax[i].set_xlabel(census_names[i])
#         ax[i].set_xlim(xmin=0)
#         ax[i].set_ylim(ymin=0)
#         for j, txt in enumerate(data['Compartment']):
#             ax[i].annotate(txt, (data[variable + '_av'].values[j], data[mode + '_av'].values[j]))
#     f.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(fname=os.path.join(path_comp, filename))
#     plt.close(f)





def skat_comp(data, modes, mode_names, title, file, size_factor=None, annotate=False):
    f, ax = plt.subplots(nrows=len(census_variables),
                         ncols=len(modes),
                         sharex='col',
                         sharey='row',
                         figsize=(5, 7))
    for j, variable in enumerate(census_variables):
        colors = ['r', 'b']
        i = 0
        if annotate:
            ax[j][i].set_ylabel('$\overline{' + census_names[j][1:-1] + '}$')
        else:
            s = mpl.rcParams['lines.markersize'] ** 2 * 0.5
            ax[j][i].set_ylabel(census_names[j])
        for mode in modes:
            if annotate:
                s = data[mode + '_std'].values * size_factor * mpl.rcParams['lines.markersize'] ** 2
            miss_se = np.isnan(data[variable].values)
            corr = np.corrcoef(data[variable].values[np.invert(miss_se)],
                               data[mode].values[np.invert(miss_se)])[0, 1]
            ax[j][i].scatter(x=data[mode].values,
                             y=data[variable].values,
                             s=s,
                             # s=0.5 * mpl.rcParams['lines.markersize'] ** 2,
                             marker='.',
                             color=colors[i],
                             label=round(corr, 4)
                             )
            if annotate:
                texts = [ax[j][i].annotate(text=txt,
                                           xy=(data[mode].values[k], data[variable].values[k]),
                                           fontsize=8) for k, txt in enumerate(data['Compartment'].values)]
            leg = ax[j][i].legend(loc="best", title="$\\rho$")
            for item in leg.legendHandles:
                item.set_visible(False)
            ax[j][i].set_ylim(bottom=0.0)
            ax[j][i].spines['top'].set_visible(False)
            ax[j][i].spines['right'].set_visible(False)
            i += 1
    if annotate:
        ax[j][1].set_xlabel('$\overline{' + mode_names[1][1:-1] + '}$')
        ax[j][0].set_xlabel('$\overline{' + mode_names[0][1:-1] + '}$')
    else:
        ax[j][1].set_xlabel(mode_names[1])
        ax[j][0].set_xlabel(mode_names[0])
    f.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_comp, 'scat_comp_' + file))
    plt.close(f)


def sme_map(data, column, mode):

    f, ax = plt.subplots(figsize=(8, 8))
    sns.despine(f)


    geo_plot(frame=data,
             column=column,
             axis=ax,
             legend='Summed Modal Efficiency')

    ax.set_title('Spatial distribution of ' + r'$SME$ for ' + mode, fontdict=dict(fontsize=18))
    ax.margins(x=0)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_maps, 'q_map_' + mode))
    plt.close(f)


def skat_acc_map(total_geo, mode, dissolved_geo):
    f, ax = plt.subplots(figsize=(8, 8))
    sns.despine(f)

    geo_plot(frame=total_geo,
             column=mode,
             axis=ax,
             legend='Summed Modal Efficiency')

    dissolved_geo.boundary.plot(ax=ax, edgecolor='black', linewidth=3)

    ax.set_title('Spatial distribution of ' + r'$SME$ for ' + mode, fontdict=dict(fontsize=18))
    ax.margins(x=0)
    plt.tight_layout()
    # plt.savefig(fname=os.path.join(path_maps, 'q_map_' + mode))
    # plt.close(f)


def hist_qij(handler, travel_times):
    time_frame = pd.concat([pd.DataFrame(handler.bike_qij), pd.DataFrame(handler.otp_qij)],
                           axis=1)
    time_frame = time_frame.set_axis(travel_times, axis=1, inplace=False)

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    colors = ['red', 'blue']

    comp_hist(frame=time_frame, colors=colors, binw=0.2, ax=ax)

    ax.set_title('Histogram for ' + r'$1/qt_{ij}$')
    ax.set_xlabel('Modal Efficiency in ' + r'$km/h$')
    ax.margins(x=0)
    plt.tight_layout()
    plt.legend()
    plt.savefig(fname=os.path.join(path_q, 'v_comp'))
    plt.close(f)


def hist_acc(data, modes, mode_names, title, file):
    f, ax = plt.subplots(ncols=2, figsize=(7, 5))

    sns.despine(f)
    # data = data[modes]
    # data.set_axis(['Bike', 'Public Transport'], axis=1, inplace=True)

    colors = ['red', 'blue']
    for i, mode in enumerate(modes):
        ax[i].hist(data[mode].values, color=colors[i], bins=40)
        ax[i].set_xlabel(mode_names[i])
        ax[i].set_ylabel('Count')

    # comp_hist(frame=data, colors=colors, binw=0.2, ax=ax)

    f.suptitle(title)
    # ax.margins(x=0)
    plt.tight_layout()
    # plt.legend()
    plt.savefig(fname=os.path.join(path_q, file))
    plt.close(f)


def mst_plot(g, pos, geo_df):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax = geo_df.plot()
    ax.set_title('MST for SKATER', fontsize=40)
    nx.drawing.nx_pylab.draw_networkx_edges(G=g, pos=pos, ax=ax, width=2.0)
    plt.tight_layout()
    # plt.savefig(fname=os.path.join(path_maps, 'mst'))
    plt.close(fig)


def skat_plot(geo_df, labels, title, labels_or=None):
    label_col = 'label'
    geo_df[label_col] = labels

    colors1 = plt.cm.tab20b(np.linspace(0., 1, 128))
    colors2 = plt.cm.tab20c(np.linspace(0, 1, 128))

    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    ax = geo_df.plot(column=label_col,
                     categorical=True,
                     figsize=(12, 8),
                     cmap=mymap,
                     linewidth=0.0
                     )
    geo_df.dissolve(by=label_col).boundary.plot(ax=ax, edgecolor='w')
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    if labels_or is None:
        labels_or = labels

    plot_texts = [plt.text(s=str(label + 1),
                           x=np.array(geo_df[labels_or == label].dissolve().representative_point()[0].coords.xy)[0],
                           y=np.array(geo_df[labels_or == label].dissolve().representative_point()[0].coords.xy)[1],
                           color='k') for label in range(len(np.unique(labels_or)))]

    # adjust_text(texts)
    plt.tight_layout()


def plot_mat(mat, title, file, sort):
    mat = np.round(mat, 2)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)


    cax = ax1.matshow(mat, cmap='magma_r')
    cbar = fig.colorbar(mappable=cax, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('$d_{GA}$')

    # for i in range(len(mat)):
    #     for j in range(len(mat)):
    #         if i > j:
    #             c = mat[j, i]
    #             if not np.isnan(c):
    #                 ax1.text(i, j, str(c), va='center', ha='center', fontsize=7)

    ax1.set_xticks(list(range(len(mat))), sort)
    ax1.set_yticks(list(range(len(mat))), sort)
    fig.suptitle('Dissimilarity Matrix for ' + title, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(path_comp, 'Matrix_' + file))
    plt.close()


def plot_clust_val():
    df = pd.read_csv(filepath_or_buffer=os.path.join(path_experiments, 'clust_val_scores.csv'))

    fig, ax1 = plt.subplots()
    x_ = list(range(2, len(df['ch'].values[:58]) + 2))
    ax1.plot(x_, df['ch'].values[:58], 'g-')
    # fig.subplots_adjust(right=0.75)
    ax2 = ax1.twinx()
    ax2.plot(x_, df['db'].values[:58], 'b-')
    # ax3 = ax1.twinx()
    # ax3.plot(x_, df['sil'].values[:58], 'r-')

    # ax3.spines.right.set_position(("axes", 1.2))

    ax1.set_title('Clustering Validation of SKATER Results')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Caliski-Harabasz Score', color='g')
    ax2.set_ylabel('Davies-Bouldin Score', color='b')
    # ax3.set_ylabel('Silhouette Score', color='r')

    ymin, ymax = plt.ylim()
    a = 18
    plt.vlines(x=a, ymax=ymax, ymin=ymin, color='b')
    plt.text(a-1, 0.5, str(a))
    # b = 14
    # plt.vlines(x=b, ymax=ymax, ymin=ymin, color='r')
    # plt.text(b, 0.5, str(b))
    c = 27
    plt.vlines(x=c, ymax=ymax, ymin=ymin, color='g')
    plt.text(c-1, 0.5, str(c))

    # x_ticks = [0, 10, a, c, 40, 50]
    # ax1.set_xticks(x_ticks)
    # x_tick_label = list(np.array(x_ticks) + 2)
    # ax1.set_xticklabels(x_tick_label)

    plt.savefig(os.path.join(path_explore, 'skater_validation'))
    plt.close()


def animate_skater(c):
    fig = plt.figure()

    # initiate an empty  list of "plotted" images
    myimages = []

    # loops through available png:s
    for p in range(0, c+1):
        ## Read in picture
        fname = "c_%01d.png" % p
        img = mgimg.imread(os.path.join(path_skater, fname))
        imgplot = plt.imshow(img)

        # append AxesImage object to the list
        myimages.append([imgplot])

    ## create an instance of animation
    my_anim = animation.ArtistAnimation(fig, myimages, interval=1000, blit=True, repeat_delay=1000)

    ## NB: The 'save' method here belongs to the object you created above
    my_anim.save(os.path.join(path_skater, "animation.mp4"))

    ## Showtime!
    #plt.show()


def heatscatter(x, y, xlabel='', ylabel='', title='', log=False, multi=False, av=False, multiax=False):
    ### Compile Test plot to find xmax
    fig_test, ax_test = plt.subplots()
    hb_test = ax_test.hexbin(x=x, y=y, gridsize=50, mincnt=2)
    verts_test = hb_test.get_offsets()
    xmax =max(verts_test[:, 0])
    xmin = min(verts_test[:, 0])
    ymax = max(verts_test[:, 1])
    plt.close(fig=fig_test)

    x_ = x[~np.isnan(np.array(x))]
    y_ = y[~np.isnan(np.array(y))]
    #xmin = np.amin(x_)
    ymin = 0.0
    #ymax = np.amax(y_)
    ### Get Plot
    if multi:
        hb = multiax.hexbin(x=x, y=y, gridsize=50, cmap='cubehelix', mincnt=2,
                             extent=[xmin, xmax, ymin, ymax])
    else:
        fig, axs = plt.subplots(ncols=1, sharey=True, figsize=(9, 6))
        if log == True:
            hb = axs.hexbin(x=x, y=y, gridsize=50, cmap='cubehelix', mincnt=1, bins='log',
                             extent=[xmin, xmax, ymin,ymax])
        else:
            hb = axs.hexbin(x=x, y=y, gridsize=50, cmap='cubehelix', mincnt=1, extent=[xmin, xmax, ymin, ymax])
        cb = fig.colorbar(hb, ax=axs)
        cb.set_label('counts')
        axs.set_xlabel(xlabel=xlabel)
        axs.set_ylabel(ylabel=ylabel)
        axs.set_title(label='Heatscatter of Modal Efficiency ' + r'$1/qt_{ij}$' +' for ' + title)
        #axs.margins(x=0)


    if av:
        # Note that mincnt=1 adds 1 to each count
        counts = hb.get_array()
        #ncnts = np.count_nonzero(np.power(10, counts))
        verts = hb.get_offsets()
        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(verts[:, 1])
        # sorts records array so all unique elements are together
        sorted_array = verts[:,1][idx_sort]
        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_array, return_counts=True, return_index=True)
        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])
        for y_hexagons in res:
            sum_x = 0
            sum_counts = 0
            for hexa in y_hexagons:
                sum_x += counts[hexa]* verts[hexa][0]
                sum_counts += counts[hexa]
            average_x = sum_x/sum_counts
            binx, biny = average_x , verts[y_hexagons[0]][1]
            if multi:
                multiax.plot(binx, biny, 'r.', zorder=100, markersize=6)
            else:
                axs.plot(binx, biny, 'r.', zorder=100, markersize=6)
    if multi:
        return hb
    else:
        # plt.show()

        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_q, 'heatscatter_' + title))
        plt.close(fig)





def clust_map(data, column, mode, circles=False):
    f, ax = plt.subplots(figsize=(8, 8))
    sns.despine(f)

    geo_plot(frame=data,
             column=column,
             axis=ax,
             reverse=True,
             legend='Weighted Clustering Coefficient',
             circles=circles)


    ax.set_title('Spatial distribution of ' + r'$\widetilde{C}$ for ' + mode, fontdict=dict(fontsize=18))
    ax.margins(x=0)
    plt.tight_layout()
    if circles:
        plt.savefig(fname=os.path.join(path_maps, 'clust_map_' + mode + '_circles'))
    else:
        plt.savefig(fname=os.path.join(path_maps, 'clust_map_' + mode))
    plt.close(f)


def trse_box(data, labels, feature, acc):
    # the figure and axes
    fig, ax = plt.subplots()

    for label in range(len(np.unique(labels))):
        if label == 6 or label == 15 or label == 16 or label == 18 or label == 24:
            continue
        x = np.array(data[labels == label][feature])
        y = np.array(data[labels == label][acc])

        # some fake data
        # x = np.random.rand(1000) ** 2
        # y = np.sqrt(np.random.rand(1000))
        # x = np.random.rand(1000)
        # y = np.random.rand(1000)

        # plotting the original data
        # ax1.scatter(x, y, c='r', s=1)

        # doing the box plot
        boxplot_2d(x, y, ax=ax, whis=1)

    a = 10
