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


def hist_acc_barth(handler, q_list):
    for mode in q_list:
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)
        sns.histplot(data=handler.neighborhood_se, x=mode)
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



    # fig, ax = plt.subplots(figsize=(20, 15))
    # skat.geo_df.plot(ax=ax)
    # for component in comp:
    #     nx.drawing.nx_pylab.draw_networkx_edges(G=component, pos=skat.pos, ax=ax)

