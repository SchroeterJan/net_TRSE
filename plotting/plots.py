from plotting.plot_functions import *
import matplotlib.image as mgimg
from matplotlib import animation

plt.style.use('seaborn')  # pretty matplotlib plots
plt.rc('font', size=24)
sns.set_theme(style="ticks")




def hist_modes(handler, travel_times):
    time_frame = pd.DataFrame(columns=travel_times)
    time_frame[travel_times[0]] = handler.bike.values.flatten()
    time_frame[travel_times[1]] = handler.otp.values.flatten()

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    colors = ['red', 'blue']

    comp_hist(frame=time_frame, colors=colors)

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
    data = pd.DataFrame(data=handler.flows.values.flatten(),
                        columns=['values'])
    sns.histplot(data, log_scale=True, legend=False)
    meanline(data=data, variable='values')
    ax.set_title('Passenger flow histogram')
    ax.margins(x=0)
    plt.xlabel('Passengers between two areas')
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_hists, 'flow'))
    plt.close(f)


def hist_se(handler):
    for i, variable in enumerate(census_variables):
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)

        handler.neighborhood_se[variable] = handler.neighborhood_se[variable].replace(to_replace=0.0, value=np.nan)
        sns.histplot(data=handler.neighborhood_se, x=variable)
        meanline(data=handler.neighborhood_se, variable=variable, x=i + 1)
        ax.set_title('Histogram of ' + variable)
        ax.margins(x=0)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hist_se, variable + '_hist'))
        plt.close(f)


def hist_scaled_se(handler):
    for variable in scaling_variables:
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)
        sns.histplot(data=handler.neighborhood_se, x=variable + '_scaled')
        ax.set_title('Histogram of ' + variable + '_scaled')
        ax.margins(x=0)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hists, variable + '_scaled_hist'))
        plt.close(f)


def se_maps(handler):
    geo_frame = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    fig, axes = multi_plot(shape=[2, 3])
    axes = axes.flatten()
    for i, each in enumerate(census_variables):
        if each == 'IHHINK_GEM':
            handler.neighborhood_se[handler.neighborhood_se[each] > 100000.0] = 100000.0
        elif each == 'PREGWERKL_P':
            handler.neighborhood_se[handler.neighborhood_se[each] > 30.0] = 30.0

        geo_frame[each] = handler.neighborhood_se[each]
        geo_plot(frame=geo_frame, axis=axes[i], column=each, cmap='OrRd')
        axes[i].set_title(label=each)
    plt.savefig(fname=os.path.join(path_maps, 'se_variables_maps'))


def hist_acc_barth(data, mode):
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    sns.histplot(data=data, x=mode)
    ax.set_title('Histogram of ' + mode)
    ax.margins(x=0)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_hists, mode))
    plt.close(f)


def hist_qij(matrix, mode):
    f, ax = plt.subplots(figsize=(7, 5))
    x = matrix.flatten()
    x = x[~np.isnan(x)]
    sns.despine(f)
    sns.histplot(data=x)
    ax.set_title('Histogram of inverse velocity for ' + mode)
    ax.margins(x=0)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_q, 'v_' + mode))
    plt.close(f)


def mst_plot(data):
    fig, ax = plt.subplots(figsize=(20, 15))
    geo_plot(frame=data.geo_df, axis=ax, cmap='tab20')
    ax.set_title('MST for SKATER', fontsize=40)
    nx.drawing.nx_pylab.draw_networkx_edges(G=data.mst(), pos=data.pos, ax=ax, width=2.0)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_maps, 'mst'))
    plt.close(fig)


# def plot_skat_clust(skat, components):
#     colors = {1: 'tab:blue',
#               2: 'lightsteelblue',
#               3: 'tab:orange',
#               4: 'moccasin',
#               5: 'tab:green',
#               6: 'mediumseagreen',
#               7: 'tab:red',
#               8: 'salmon',
#               9: 'tab:purple',
#               10: 'plum',
#               11: 'tab:brown',
#               12: 'sandybrown',
#               13: 'tab:pink',
#               14: 'palevioletred',
#               15: 'tab:gray',
#               16: 'lightgray',
#               17: 'tab:olive',
#               18: 'gold',
#               19: 'tab:cyan',
#               20: 'aquamarine'}
#
#     skat.geo_df['clust'] = 0
#     fig, ax = plt.subplots(figsize=(20, 15))
#     ax.set_aspect('equal')
#     ax.set_title('SKATER clustering', fontsize=40)
#     skat.geo_df.boundary.plot(ax=ax, edgecolor='black')
#     ax.set_axis_off()
#
#     skat.geo_df.plot(ax=ax, color=colors[1])
#     plt.tight_layout()
#     plt.savefig(fname=os.path.join(path_skater, 'c_0'))
#
#
#     for i, comp in enumerate(components):
#         for node in list(comp.nodes()):
#             skat.geo_df.at[node, 'clust'] = i + 1
#         comp_geo = skat.geo_df.loc[list(comp.nodes())]
#         comp_geo.plot(ax=ax, color=colors[i + 2])
#         ax.set_axis_off()
#         plt.tight_layout()
#         plt.savefig(fname=os.path.join(path_skater, 'c_' + str(i + 1)))


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
        hb = multiax.hexbin(x=x, y=y, gridsize=50, cmap='cubehelix', mincnt=20,
                             extent=[xmin, xmax, ymin, ymax])
    else:
        fig, axs = plt.subplots(ncols=1, sharey=True, figsize=(7, 4))
        if log == True:
            hb = axs.hexbin(x=x, y=y, gridsize=50, cmap='cubehelix', mincnt=1, bins='log',
                             extent=[xmin, xmax, ymin,ymax])
        else:
            hb = axs.hexbin(x=x, y=y, gridsize=50, cmap='cubehelix', mincnt=1, extent=[xmin, xmax, ymin, ymax])
        cb = fig.colorbar(hb, ax=axs)
        cb.set_label('counts')
        axs.set_xlabel(xlabel=xlabel)
        axs.set_ylabel(ylabel=ylabel)
        #axs.set_title(label=title)


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
                multiax.plot(binx, biny, 'r.', zorder=100, markersize=12)
            else:
                axs.plot(binx, biny, 'r.', zorder=100, markersize=12)
    if multi:
        return hb
    else:
        plt.show()