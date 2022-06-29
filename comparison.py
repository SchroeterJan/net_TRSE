from resources.exp_resources import *

from scipy.cluster import hierarchy
from scipy.stats import rankdata


class Hierarchical:

    def __init__(self, vars, init_labels, model_data):
        print("Initializing " + self.__class__.__name__)
        self.vars = vars
        self.labeling = init_labels
        self.model_ = model_data

    def form_hierarchy(self):
        new = max(self.labeling)
        z = []
        size = {}
        cur_labeling = np.unique(self.labeling)
        while len(np.unique(self.labeling)) > 1:
            new += 1
            print('getting resemblance matrice for ' + str(len(np.unique(self.labeling))) + ' clusters')
            r = np.triu(self.upgma())
            r[np.tril_indices(r.shape[0], 0)] = np.nan
            merge_ind = np.unravel_index(np.nanargmin(r), r.shape)

            x = np.unique(self.labeling)[merge_ind[0]]
            y = np.unique(self.labeling)[merge_ind[1]]
            cur_labeling[np.where(cur_labeling == x)] = new
            cur_labeling[np.where(cur_labeling == y)] = new
            self.labeling[self.labeling == x] = new
            self.labeling[self.labeling == y] = new

            unique, counts = np.unique(cur_labeling, return_counts=True)
            counts = dict(zip(unique, counts))
            size[new] = counts[new]
            z.append([x, y, np.nanmin(r), size[new]])
        return z



    # unweighted pair-group method using arithmetic averages
    def upgma(self):
        # initialize resemblance matrix
        r_mat = np.zeros(shape=(len(np.unique(self.labeling)), len(np.unique(self.labeling))))
        for i_ind, i in enumerate(np.unique(self.labeling)):
            for j_ind, j in enumerate(np.unique(self.labeling)):
                if i == j:
                    r_mat[i_ind][j_ind] = 0.0
                else:
                    x = self.model_[self.labeling == i][self.vars]
                    y = self.model_[self.labeling == j][self.vars]
                    # xy = pd.concat([x, y])

                    # standardized Euclidean distance takes the standard deviation between the compartments into account
                    e_mat = metrics.pairwise.nan_euclidean_distances(X=x, Y=y)
                    r_mat[i_ind][j_ind] = np.nanmean(e_mat.flatten())
        return r_mat


def best_cut():
    d_ga = 0.01

    ch_values = []
    db_values = []
    no_clust = []
    nan_rows = pd.isnull(handler.model_).any(axis=1)
    ch_matrix = handler.model_[~nan_rows]

    while d_ga < 3.0:
        d_labels = np.array(list(skat_labels))
        d_cut = hierarchy.cut_tree(Z=z, height=d_ga)
        for i, clust in enumerate(d_cut):
            d_labels[np.where(init_labels == i)] = clust
        no_clust.append(len(np.unique(d_labels)))
        d_labels = d_labels[~nan_rows]
        ch_values.append(metrics.calinski_harabasz_score(X=ch_matrix, labels=d_labels))
        db_values.append(metrics.davies_bouldin_score(X=ch_matrix, labels=d_labels))
        d_ga += 0.01


def example_dist():
    tw = np.where(skat_labels == 11)
    tw1 = np.where(skat_labels == 20)
    ex = np.append(tw[0], tw1[0])
    geo_df = geopandas.GeoDataFrame(data=handler.neighborhood_se[model],
                                    crs=crs_proj,
                                    geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
    ex_geo = geo_df.loc[list(ex)]
    ex_geo = ex_geo.sort_index()
    df = np.ones(shape=(len(tw[0]), len(tw1[0])))
    # df *= np.tri(*df.shape, k=-1)
    df = pd.DataFrame(df, index=tw[0], columns=tw1[0])
    df = df.combine_first(df.T).fillna(0.0)
    df = np.triu(df.values)
    g = nx.from_numpy_array(A=df)
    w_ex = weights.Queen.from_networkx(g)
    ex_geo.reset_index(inplace=True, drop=True)
    ax = ex_geo.plot(edgecolor='grey', facecolor='w')
    f, ax = w_ex.plot(ex_geo, ax=ax,
                      edge_kws=dict(color='r', linestyle=':', linewidth=1),
                      node_kws=dict(marker=''))
    ax.set_axis_off()


def mat_operations(vars, title, file):
    # create resemblance matrix
    handler.stat_prep(vars=vars)
    h_se = Hierarchical(vars=vars,
                        init_labels=init_labels,
                        model_data=handler.model_)

    r_mat = h_se.upgma()

    # order matrix
    ind = list(np.array(clust_ind) + 1)
    df = pd.DataFrame(r_mat, index=ind, columns=ind)
    df = df.reindex(sorted_, axis=0)
    df = df.reindex(sorted_, axis=1)

    # trim matrix
    r_mat = np.triu(df.values)
    r_mat[np.tril_indices(r_mat.shape[0], 0)] = np.nan

    plot_mat(mat=r_mat,
             title=title,
             file=file,
             sort=[str(item) for item in sorted_])






handler = DataHandling(new=True)
handler.matrices()

acc_all = [clust_bike, clust_pt, sme_bike, sme_pt]


# load labels found by regionalization
no_compartments = 27
skat_labels = np.load(file=os.path.join(path_experiments, 'reg_result_' + str(no_compartments) + '.npy'))
# get unique labels and the count of neighborhoods assigned to each
unique, counts = np.unique(skat_labels, return_counts=True)
# get all 'islands' - agglomerations only comprised of a single neighborhood
islands = np.where(counts < 3)[0]
# islands = np.append(islands, [0])
# get row indices in the original size array of the islands
island_rows = [list(np.where(skat_labels == island)[0]) for island in islands]
island_rows = [item for sublist in island_rows for item in sublist]
# neglect islands from label array
skat_labels = np.delete(arr=skat_labels, obj=island_rows)
# store label array as initial labels
init_labels = np.array(skat_labels)


# skat_comp(data=handler.neighborhood_se,
#           modes=[clust_bike, clust_pt],
#           mode_names=[clust_bike_name, clust_pt_name],
#           title='Weighted Clustering Coefficients vs. \n Socioeconomic Variables',
#           file='clust')
#
#
# skat_comp(data=handler.neighborhood_se,
#           modes=[sme_bike, sme_pt],
#           mode_names=[sme_bike_name, sme_pt_name],
#           title='Summed Modal Efficiencies vs. \n Socioeconomic Variables',
#           file='sme')


# drop islands from neighborhood data
handler.neighborhood_se.drop(labels=list(handler.neighborhood_se.index[island_rows]), inplace=True)
handler.neighborhood_se.reset_index(inplace=True, drop=True)

skat_geo = geopandas.GeoDataFrame(data=handler.neighborhood_se[acc_all],
                                  crs=crs_proj,
                                  geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))

skat_geo['label'] = skat_labels
center = list(np.array([1, 2, 4, 9, 13, 25, 26, 27]) -1)
center_geo = skat_geo[[item in center for item in list(skat_geo['label'])]]
outter_geo = skat_geo[[item not in center for item in list(skat_geo['label'])]]



a = 10

clust_ind = [clust for clust in list(unique) if clust not in list(islands)]
skat_stat = skat_stats(geo_df=skat_geo,
                       model=acc_all,
                       skat_labels=skat_labels,
                       or_data=handler.neighborhood_se[census_variables],
                       index_col=clust_ind)

skat_stat.drop('overall', inplace=True)
av_census = [variable + '_av' for variable in census_variables]
skat_stat.rename(columns=dict(zip(av_census, census_variables)), inplace=True)


# skat_comp(data=skat_stat,
#           modes=[clust_bike, clust_pt],
#           mode_names=[clust_bike_name, clust_pt_name],
#           size_factor=80,
#           title='Averages of Weighted Clustering Coefficients vs. \n Averages of Socioeconomic Variables \n for Clusters produced by Regionalization',
#           file='clust_skat',
#           annotate=True)
#
#
# skat_comp(data=skat_stat,
#           modes=[sme_bike, sme_pt],
#           mode_names=[sme_bike_name, sme_pt_name],
#           size_factor=4,
#           title='Averages of Summed Modal Efficiencies vs. \n Averages Socioeconomic Variables \n for Clusters produced by Regionalization',
#           file='sme_skat',
#           annotate=True)


# sort everything by income
sort_by = census_variables[2]
sorted_ = list(skat_stat.sort_values(by=sort_by)['Compartment'])


# get resemblance matrices for public transport, bike and socioeconomic status

mat_operations(vars=list(census_variables),
               title='\n Socioeconomic Variables',
               file='se')

mat_operations(vars=[clust_pt],
               title=clust_pt_name,
               file='clust_pt')
mat_operations(vars=[clust_bike],
               title=clust_bike_name,
               file='clust_bike')
mat_operations(vars=[sme_pt],
               title=sme_pt_name,
               file='sme_pt')
mat_operations(vars=[sme_bike],
               title=sme_bike_name,
               file='sme_bike')

mat_operations(vars=[clust_pt, clust_bike],
               title='\n Combining ' + clust_pt_name + ' and ' + clust_bike_name,
               file='clust_both')
mat_operations(vars=[sme_pt, sme_bike],
               title='\n Combining ' + sme_pt_name + ' and ' + sme_bike_name,
               file='sme_both')








def acc_skat():
    skat_geo['labels'] = skat_labels
    aggl_geo = skat_geo.dissolve(by='labels')
    f, ax = plt.subplots(figsize=(8, 8))
    sns.despine(f)

    geo_plot(frame=skat_geo,
             column=pt_comp,
             axis=ax,
             legend='Summed Modal Efficiency')
    aggl_geo.boundary.plot(ax=ax)




# acc_skat()






# def scale_mat(mat):
#     scaler = preprocessing.MinMaxScaler()
#     x_scaled = scaler.fit_transform(X=mat.flatten().reshape(-1, 1))
#     return np.reshape(a=x_scaled[:, 0], newshape=mat.shape)
#
# r_mat_otp = scale_mat(r_mat_otp)
# r_mat_bike = scale_mat(r_mat_bike)
# r_mat_se = scale_mat(r_mat_se)


def comp(r_mat_acc, color, title, file):
    r_se = r_mat_se.flatten()[~np.isnan(r_mat_se.flatten())]
    r_acc = r_mat_acc.flatten()[~np.isnan(r_mat_acc.flatten())]
    color = color.flatten()[~np.isnan(color.flatten())]
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.scatter(x=r_se,
                y=r_acc,
                # c=color,
                # marker='.',
                # cmap='Reds'
                )
    fig.suptitle('Dissimilarities by Socioeconomic Variables vs. \n by ' + title, fontsize=18)
    ax.set_xlabel('Dissimilarity by Socioeconomic Variables')
    ax.set_ylabel('Dissimilarity by ' + title)
    plt.tight_layout()
    plt.savefig(os.path.join(path_comp, 'comp_' + file))
    plt.close()
    print(np.corrcoef(r_se, r_acc)[0,1])



comp(r_mat_acc=r_mat_otp,
     color=trim_mat(np.subtract.outer(skat_stat[sort_by].values, skat_stat[sort_by].values)),
     title='Public Transport',
     file ='pt_new')
comp(r_mat_acc=r_mat_bike,
     color=trim_mat(np.subtract.outer(skat_stat[sort_by].values, skat_stat[sort_by].values)),
     title='Bike',
     file='bike_new')













# model = ['otp_q']
# dendro_title = 'Dendrogram of UPGMA on regionalization clusters \n for $St_{Transport}$ and $\widetilde{C}_{Bike}$'
# dendro_path = 'dendrogram_acc'
# cutoff = 1.29
# upgma_title = 'UPGMA on regionalization clusters for $St_{Transport}$ and $\widetilde{C}_{Bike}$ \n Tree cut-off at $d_{GA}=1.29$'
# upgma_path = 'upgma_map_acc'
# r_mat_file = 'r_mat_otp'


# handler.edu_score()
# model = list(census_variables)
# model.append('edu_score')
# model = model[3:]
# dendro_title = 'Dendrogram of UPGMA on regionalization clusters \n for socioeconomic variables'
# dendro_path = 'dendrogram_se'
# cutoff = 1.49
# upgma_title = 'UPGMA on regionalization clusters for socioeconomic variables \n Tree cut-off at $d_{GA}=1.37$'
# upgma_path = 'upgma_map_se'
# r_mat_file = 'r_mat_se'











cluster_no = np.array([clust for clust in list(unique) if clust not in list(islands)]) + 1
r_mat = pd.DataFrame(data=r_mat, index=cluster_no, columns=cluster_no)

# r_mat.to_csv(path_or_buf=os.path.join(path_experiments, r_mat_file))



skat_geo = geopandas.GeoDataFrame(data=skat_labels,
                                  crs=crs_proj,
                                  geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry)).dissolve(by=0)

# geo_df = geopandas.GeoDataFrame(data=handler.neighborhood_se[model],
#                                 crs=crs_proj,
#                                 geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))


# stats = skat_stats(geo_df=geo_df,
#                    model=model,
#                    skat_labels=skat_labels,
#                    or_data=handler.neighborhood_se[census_variables],
#                    index_col=[clust for clust in list(unique) if clust not in list(islands)]
#                    )


# skat_acc_map(total_geo=geo_df, mode=model[0], dissolved_geo=skat_geo)

# example_dist()

# w_queen = weights.Queen.from_dataframe(df=skat_geo, geom_col='geometry')
# cust_adj = Adj_Islands(geo_frame=skat_geo, g_init=w_queen.to_networkx())
# skat_adj = np.triu(nx.to_numpy_array(cust_adj.adj_g))

r_acc = np.triu(pd.read_csv(filepath_or_buffer=os.path.join(path_experiments, 'r_mat_acc'), index_col=0))
r_se = np.triu(pd.read_csv(filepath_or_buffer=os.path.join(path_experiments, 'r_mat_se'), index_col=0))

r_acc[np.tril_indices(r_acc.shape[0], 0)] = np.nan
r_se[np.tril_indices(r_se.shape[0], 0)] = np.nan




# plot_mat(mat=np.round(r_se, 2))

# r_acc[skat_adj == 1] = 0.0
# r_se[skat_adj ==1] = 0.0

r_acc = r_acc.flatten()
r_acc = r_acc[r_acc != 0.0]

r_se = r_se.flatten()
r_se = r_se[r_se != 0.0]

small = r_se < np.nanquantile(r_se, 0.25)


def plot_small():
    plt.scatter(x=r_se[small], y=r_acc[small])
    plt.ylim(0, 2.5)
    plt.xlim(0, 2)
    plt.title('Similarities by Socioeconomic Variables \n vs. by Accessibility measures')
    plt.ylabel('Similarity by Accessibility measures')
    plt.xlabel('Similarity by Socioeconomic variables')



def plot_all():
    plt.scatter(x=r_se, y=r_acc)
    plt.ylim(0, 4)
    plt.xlim(0, 8)
    plt.title('Similarities by Socioeconomic Variables \n vs. by Accessibility measures')
    plt.ylabel('Similarity by Accessibility measures')
    plt.xlabel('Similarity by Socioeconomic variables')


def network_control():
    skat_geo.reset_index(inplace=True, drop=True)
    diff = np.abs(r_se - r_acc)
    diff_small = np.where(small, diff, np.nan)
    # smaller differences should result in larger edges (to illustrate strength)
    diff_small = 1/diff_small
    diff_small *= (3.0 / np.nanmax(diff_small))
    diff_small = np.nan_to_num(diff_small)
    pos = geo_pos(geo_df=skat_geo)
    g = nx.from_numpy_array(diff_small)

    fig, ax = plt.subplots(figsize=(20, 15))
    ax = skat_geo.boundary.plot()
    # ax.set_title('MST for SKATER', fontsize=40)
    edges = g.edges()
    weights = [g[u][v]['weight'] for u, v in edges]
    nx.drawing.nx_pylab.draw_networkx_edges(G=g, pos=pos, ax=ax, width=weights)
    plt.tight_layout()
    # plt.savefig(fname=os.path.join(path_maps, 'mst'))
    plt.close(fig)


plot_small()
network_control()


# real_labels = np.unique(skat_labels)
# def relabeling():
#     for mask_l, real_l in enumerate(real_labels):
#         skat_labels[skat_labels == real_l] = mask_l
#
# relabeling()
# real_labels = dict(zip(np.unique(skat_labels), real_labels))
#
# h = Hierarchical(vars=model, init_labels=skat_labels)
# z = h.form_hierarchy()

z = np.load(file=os.path.join(path_experiments, 'upgma_se.npy'))


# best_cut()



fig, ax = plt.subplots()
dn = hierarchy.dendrogram(z)
ax.set_xticklabels(list(map(str, np.array(dn['leaves']) + 1)))
ax.set_ylabel('$d_{GA}$')
ax.set_xlabel('Cluster Number')
ax.set_title(dendro_title)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.hlines(y=cutoff, xmax=xmax, xmin=xmin, color='black')
trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
ax.text(0, cutoff, "{:.2f}".format(cutoff), color="black", transform=trans,
        ha="right", va="center")

plt.savefig(os.path.join(path_maps, dendro_path))
plt.close()

cut = hierarchy.cut_tree(Z=z, height=cutoff)

for i, clust in enumerate(cut):
    skat_labels[np.where(init_labels == i)] = clust


skat_plot(geo_df=geo_df, labels=skat_labels, title=upgma_title, labels_or=init_labels)
plt.savefig(os.path.join(path_maps, upgma_path))


a = 1


