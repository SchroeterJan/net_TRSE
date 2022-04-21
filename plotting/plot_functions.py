from resources.prep_resources import *

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import cartopy.crs as ccrs
import networkx as nx


def geo_plot(frame,reverse=False, column=None, axis=None, legend=None, circles=False):

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
                   linewidth=0,
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
                       # #fontsize='medium',
                       # #title=legend,
                       # #frameon=False
                   )
                   )
    #plt.colorbar(axis, fraction=0.046, pad=0.04)
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


def comp_hist(frame, colors, binw, ax):
    for i, col in enumerate(frame.columns):
        sns.histplot(data=frame, x=col, binwidth=binw, color=colors[i], label=col, alpha=0.5)
        meanline(data=frame, variable=col, x=i + 5, ax=ax)


def multi_plot(shape, suptitle='', ytext='', xtext=''):
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(30, 15))
    # fig.suptitle(suptitle)
    fig.text(0.5, 0.04, xtext, ha='center', va='center')
    fig.text(0.03, 0.5, ytext, ha='center', va='center', rotation='vertical')
    return fig, axes



def boxplot_2d(x,y, ax, whis=1.5):
    xlimits = [np.nanpercentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.nanpercentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k'
    )

