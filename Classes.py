from config import *


class DataHandling:

    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        self.Load_Data()


    def Load_Data(self):
        print("Loading data")
        self.neighborhood_se = pd.read_csv(filepath_or_buffer=path_neighborhood_se, sep=';', index_col=0)
        self.flows = pd.read_csv(filepath_or_buffer=path_flows, sep=';', index_col=0)
        self.bike = pd.read_csv(filepath_or_buffer=os.path.join(path_repo, path_generated,'Bike_times_GH.csv'),
                                sep=';', index_col=0)
        self.pt = pd.read_csv(filepath_or_buffer=os.path.join(path_repo, path_generated, 'PT_times.csv'),
                              sep=';', index_col=0)


    def Build_speed_vector(self, variable, euclid, name):
        speed = []
        for i, each in enumerate(euclid):
            speed = (each*1000.0)/(variable[i]/60.0)
            speed.append(speed)

        pickle.dump(np.array(speed), open(name + "_speed.p", "wb"))



    def Thresholding(self, variable, largest_extend):
        variable = np.nan_to_num(x=variable, nan=0.0)
        largest = np.argsort(variable)[-largest_extend:]
        for each in largest:
            if variable[each] == 'nan':
                continue
            else:
                variable[each] = variable[largest[0]]
        print(' thresholded to ' + str(variable[largest[0]]))
        return variable

    def MatrixThresholding(self, variable, extend, length):
        matrix = self.Build_Matrix(length=length,
                                               data_list=np.array(variable))
        matrix2 = np.triu(matrix, k=0)
        fastest, fastest_index = Analysis().find_max(matrix2, extend)
        bins_index = np.bincount(fastest_index).argsort()[-4:][::-1]
        for each in bins_index:
            matrix[:, each] = np.nan
            matrix[each, :] = np.nan
        matrix = np.triu(matrix, k=1)
        matrix = matrix.flatten()
        matrix = np.delete(arr=matrix, obj=np.where(matrix == 0.0))
        return matrix

    def DifferenceMatrix(self, vector):

        edges = []
        for i, Buurt_score in enumerate(vector):
            for j, Buurt_score2 in enumerate(vector[i + 1:]):
                edges.append(np.absolute(Buurt_score - Buurt_score2))

        return np.array(edges)


class Plotting:
    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rc('font', size=24)

    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        sns.set(rc={"figure.figsize": (14, 18)}, font_scale=2.0)

    def Correlation_Heatmap(self, df):
        print("Plotting Correlation Heatmap of given DataFrame")
        corrmat = df.corr()
        f, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(corrmat, ax=ax, cmap="RdYlBu", linewidths=0.1)
        plt.show()

    def Heatscatter(self, x, y, xlabel='', ylabel='', title='', log=False, multi=False, av=False, multiax=False):
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



    def MultiHeatScatter(self, x, y, axt, shape, suptitle='', ytext='', xtext=''):
        fig, axes = self.MultiPlot(shape=shape, suptitle=suptitle, ytext=ytext, xtext=xtext)
        axes_count = 0
        row_count = 0
        for row_axes in axes:
            for axis in row_axes:
                y_ = y[axes_count]
                hb = self.Heatscatter(x=x, y=y_, log=False, multi=True, multiax=axis, av=True)
                if row_count != (len(axes) - 1):
                    axis.get_xaxis().set_visible(False)
                cb = fig.colorbar(hb, ax=axis)
                cb.set_label('counts')
                #axis.xaxis.grid()
                #axis.yaxis.grid()
                axis.set_title(axt[axes_count])
                axes_count += 1
            row_count += 1

        plt.show()

    def MultiPlot(self, shape, suptitle='', ytext='', xtext= ''):
        fig, axes = plt.subplots(shape[0], shape[1])
        #fig.suptitle(suptitle)
        fig.text(0.5, 0.04, xtext, ha='center', va='center')
        fig.text(0.03, 0.5, ytext, ha='center', va='center', rotation='vertical')
        return fig, axes

    def MultiScatter(self, x, y, c, axt, shape, suptitle='', ytext='', xtext=''):
        fig, axes = self.MultiPlot(shape=shape, suptitle=suptitle, ytext=ytext, xtext=xtext)
        cmap = sns.cubehelix_palette(as_cmap=True)

        axes_count = 0
        row_count = 0
        for row_axes in axes:
            for axis in row_axes:
                scatter = axis.scatter(x=x, y=y, c=c[axes_count], cmap=cmap, s=28.0)
                if axes_count % 2 != 0:
                    axis.get_yaxis().set_visible(False)
                if row_count != (len(axes)-1):
                    axis.get_xaxis().set_visible(False)
                axis.invert_yaxis()
                axis.invert_xaxis()
                axis.xaxis.grid()
                axis.yaxis.grid()
                fig.colorbar(scatter, ax=axis)
                axis.set_title(axt[axes_count])
                axes_count +=1
            row_count +=1
        plt.savefig('Bike_vs_PT_socio.png',dpi=fig.dpi, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
        plt.show()

    def Hist(self, series, title, bins):
        a = sns.histplot(series, x=series, bins=bins)
        a.set_title(title)
        plt.savefig(fname=title)
        plt.show()




    def Scatter(self, df, x, y, xlabel=None, ylabel=None, title=''):
        #sns.set_theme(style="ticks")
        #df = df.melt('Average Income 2017', var_name='category',  value_name='y')
        #u, c = np.unique(df["category"], return_inverse=True)
        fig, ax = plt.subplots()
        ax.scatter(x=df[x], y=df[y])
        #scmap = lambda i: plt.plot([],[], marker="o",ls="none", c=plt.cm.tab10(i))[0]
        #plt.gca().invert_yaxis()
        if xlabel==None:
            ax.set_xlabel(x)
        else:
            ax.set_xlabel(xlabel=xlabel)

        if ylabel==None:
            ax.set_ylabel(y)
        else:
            ax.set_ylabel(ylabel=ylabel)

        ax.set_title(label=title)
        ax.invert_yaxis()
        ax.invert_xaxis()
        #plt.legend(handles=[scmap(i) for i in range(len(u))],
                   #labels=list(u))
        #g = sns.catplot(x="Average Income 2017", data=df)
        plt.show()


class Analysis:


    def __init__(self, handler):
        self.handler = handler
        print("Initializing " + self.__class__.__name__)


    def find_max(self, matrix2, max_extend):
        print("Getting the " + str(max_extend) + " fastest trips")
        temp = matrix2
        trips = []
        trips_index = []
        for i in range(max_extend):
            max_val = np.unravel_index(np.nanargmax(temp), matrix2.shape)
            trips.append(temp[max_val[0]][max_val[1]])
            trips_index.append(max_val[0])
            trips_index.append(max_val[1])
            temp[max_val[0]][max_val[1]] = 0.0
        return np.array(trips), np.array(trips_index)


    def Clustering(self, matrix):
        neighborhoods = self.handler.neighborhood_se.index
        Graph = nx.Graph()
        matrix = np.nan_to_num(x=matrix, nan=0.0)
        # create a node for each of these unique locations
        for neighborhood in neighborhoods:
            Graph.add_node(neighborhood)

        # create edges between all nodes and populate them with travel times as weights
        for i, row in enumerate(matrix):
            for j, value in enumerate(row[i:]):
                if value != 0.0:
                    Graph.add_edge(neighborhoods[i], neighborhoods[j + i], weight=value)

        cluster_dict = nx.clustering(Graph, weight='weight')
        cluster_values = np.array(list(cluster_dict.values()))
        return cluster_values



