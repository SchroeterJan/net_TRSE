from resources.prep_resources import *

from matplotlib import pyplot as plt
import seaborn as sns


def geo_plot(frame, column=None, axis=None, legend=None):
    if axis != None:
        axis.set_axis_off()
    frame.plot(ax=axis, column=column,
               cmap='Set1',
               scheme='quantiles',
               legend=True,
               legend_kwds=dict(loc='lower left',
                                fontsize='medium',
                                title=legend,
                                frameon=False))


def meanline(data, variable=None, x=1):
    if isinstance(data, pd.DataFrame):
        plt.axvline(np.nanmean(data[variable].to_numpy()), color='k', linestyle='dashed', linewidth=1)
    elif isinstance(data, np.ndarray):
        plt.axvline(np.nanmean(data.flatten()), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(data[variable].mean() * 1.1,
             max_ylim * (1 - 0.1 * x),
             'Mean: {:.2f}'.format(data[variable].mean()))


# # plot a graph (network) on a map represented by a geopandas data frame
# def geo_net_plot(geo_frame, graph):
#     # initiate a dictionary storing the position of all nodes
#     pos = {}
#     # loop over centroids of the areas in the geopandas data frame
#     for count, elem in enumerate(np.array(geo_frame.centroid)):
#         # add latitude (y) and longitude (x) of each area centroid to the position dictionary
#         pos[geo_frame.index[count]] = (elem.x, elem.y)
#     fig, ax = plt.subplots(figsize=(20, 15))
#     # plot the geopandas data frame
#     geo_frame.plot(ax=ax)
#     # plot the graph on the map using networkx
#     nx.drawing.nx_pylab.draw_networkx_edges(G=graph, pos=pos, ax=ax)


# plot a histogram comparing columns of a data frame


def comp_hist(frame, colors, binw):
    for i, col in enumerate(frame.columns):
        sns.histplot(data=frame, x=col, binwidth=binw, color=colors[i], label=col, alpha=0.5)
        meanline(data=frame, variable=col, x=i + 5)


def multi_plot(shape, suptitle='', ytext='', xtext=''):
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(30, 15))
    # fig.suptitle(suptitle)
    fig.text(0.5, 0.04, xtext, ha='center', va='center')
    fig.text(0.03, 0.5, ytext, ha='center', va='center', rotation='vertical')
    return fig, axes