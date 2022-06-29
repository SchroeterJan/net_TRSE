from resources.prep_resources import *

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import cartopy.crs as ccrs
import networkx as nx




# geographic plotting using pysal and mapclassify
def geo_plot(frame, reverse=False, vmax=None, column=None, axis=None, legend=None, circles=False):

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("bottom", size="3%", pad=0.1)
    cax.tick_params(axis='both', which='major', labelsize=15)
    cax.set_title(label=legend, fontdict=dict(fontsize=15))
    if reverse:
        frame.plot(ax=axis,
                   column=column,
                   cmap='magma_r',
                   #cmap='Set1',
                   #scheme='equalinterval',
                   legend=True,
                   cax=cax,
                   legend_kwds=dict(
                       orientation='horizontal',
                       # label=legend,
                       # fontsize='medium'
                       # loc='lower left',
                       # #fontsize='medium',
                       # #title=legend,
                       # #frameon=False
                       ),
                   linewidth=0.0,
                   )
    else:
        frame.plot(ax=axis,
                   column=column,
                   cmap='magma',
                   # cmap='Set1',
                   # scheme='equalinterval',
                   legend=True,
                   cax=cax,
                   legend_kwds=dict(
                       orientation='horizontal',
                       # label=legend,
                       # fontsize='medium'
                       # loc='lower left',
                       # #frameon=False
                       ),
                   linewidth=0.0,
                   vmax=vmax
                   )
    if circles:
        map = frame.dissolve()
        for dist in range(1, 5):
            circle = map.centroid.buffer(distance=dist*2*1000).boundary
            circle.plot(ax=axis, color='g', linewidth=2.0)
            axis.text(x=circle.bounds['maxx'],
                      y=circle.bounds['miny'] + (circle.bounds['maxy'] - circle.bounds['miny'])/2,
                      s=(str(dist * 2) + 'km'),
                      zorder=10*dist,
                      color='g')

    axis.set_axis_off()


# plot a meanline into a histogram
def meanline(data, ax, variable=None, x=1):
    min_ylim, max_ylim = ax.get_ylim()
    if isinstance(data, pd.DataFrame):
        ax.vlines(x=np.nanmean(data[variable].to_numpy()),
                  ymin=min_ylim,
                  ymax=max_ylim,
                  color='k',
                  linestyle='dashed',
                  linewidth=1)
    elif isinstance(data, np.ndarray):
        ax.vlines(x=np.nanmean(data.flatten()),
                  ymin=min_ylim,
                  ymax=max_ylim,
                  color='k',
                  linestyle='dashed',
                  linewidth=1)
    ax.text(x=data[variable].mean() * 1.1,
            y=max_ylim * (1 - 0.1 * x),
            s='Mean: {:.2f}'.format(data[variable].mean()))


# compare the distributions of multiple variables in one plot
def comp_hist(frame, colors, binw, ax):
    for i, col in enumerate(frame.columns):
        sns.histplot(data=frame, x=col, binwidth=binw, color=colors[i], label=col, alpha=0.5)
        meanline(data=frame, variable=col, x=i + 5, ax=ax)


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