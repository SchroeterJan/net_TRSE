from resources.exp_resources import *

import pygeoda
from libpysal import weights
from spopt.region import skater as skat_lib
import warnings


# class cust_tree:
#
#
#     def __init__(self, c):
#         print("Initializing " + self.__class__.__name__)
#         self.c = c
#         self.create()
#
#     def change_c(self, or_, dest_):
#         for or_new in np.append(np.where(self.c[:, or_])[0], np.where(self.c[or_, :])[0]):
#             if or_new < dest_:
#                 self.c[or_new][dest_] = True
#             elif or_new > dest_:
#                 self.c[dest_][or_new] = True
#         for dest_new in np.append(np.where(self.c[:, dest_])[0], np.where(self.c[dest_, :])[0]):
#             if dest_new < or_:
#                 self.c[dest_new][or_] = True
#             elif dest_new > or_:
#                 self.c[or_][dest_new] = True
#
#
#
#     def create(self):
#         maxdist = diss(geo_df[model])
#         self.e = np.triu(maxdist)
#         self.e[np.tril_indices(self.e.shape[0], 0)] = np.nan
#         e_ = np.where(self.c, self.e, 1.0)
#
#         i = 0
#         self.T = []
#
#         esort = np.argsort(self.e.flatten())
#         self.labels = np.array(range(len(self.c)))
#
#         maxdist = pd.DataFrame(maxdist)
#
#         while len(self.T) < (len(self.c)-1):
#             cand = esort[i]
#             cand_u, cand_v = np.unravel_index(cand, self.c.shape)
#             if self.c[cand_u, cand_v]:
#                 clust_m = self.labels[cand_u]
#                 clust_l = self.labels[cand_v]
#                 if clust_m != clust_l:
#                     e_cand = self.e[cand_u][cand_v]
#                     if e_cand >= maxdist[clust_m][clust_l]:
#                         print(len(self.T))
#                         nodes_m = np.where(self.labels == clust_m)[0]
#                         nodes_l = np.where(self.labels == clust_l)[0]
#                         e_star_cand = {}
#                         for m_ in nodes_m:
#                             for l_ in nodes_l:
#                                 self.change_c(or_=m_, dest_=l_)
#                                 if m_ < l_:
#                                     e_star_cand[e_[m_][l_]] = [m_, l_]
#                                 elif m_ > l_:
#                                     e_star_cand[e_[l_][m_]] = [l_, m_]
#                         e_star_min = np.nanmin(list(e_star_cand.keys()))
#                         self.T.append(e_star_cand[e_star_min])
#
#                         self.labels[nodes_m] = clust_l
#
#                         for c in np.unique(self.labels):
#                             if c == clust_l:
#                                 continue
#                             dist_max = np.nanmax([maxdist[c][clust_l], maxdist[c][clust_m]])
#                             maxdist[c][clust_l] = dist_max
#                             maxdist[clust_l][c] = dist_max
#                         maxdist.drop(labels=clust_m, axis=0, inplace=True)
#                         maxdist.drop(labels=clust_m, axis=1, inplace=True)
#                     elif np.isnan(e_cand):
#                         print(len(self.T))
#                         nodes_m = np.where(self.labels == clust_m)[0]
#                         nodes_l = np.where(self.labels == clust_l)[0]
#                         e_star_cand = {}
#                         for m_ in nodes_m:
#                             for l_ in nodes_l:
#                                 self.change_c(or_=m_, dest_=l_)
#                                 if m_ < l_:
#                                     e_star_cand[e_[m_][l_]] = [m_, l_]
#                                 elif m_ > l_:
#                                     e_star_cand[e_[l_][m_]] = [l_, m_]
#                         del e_star_cand[1.0]
#                         e_star_min = np.min(list(e_star_cand.keys()))
#                         if np.isnan(e_star_min):
#                             self.T.append(list(e_star_cand.values())[0])
#                         else:
#                             self.T.append(e_star_cand[e_star_min])
#                         self.labels[nodes_m] = clust_l
#
#                         for c in np.unique(self.labels):
#                             if c == clust_l:
#                                 continue
#                             dist_max = np.nanmax([maxdist[c][clust_l], maxdist[c][clust_m]])
#                             maxdist[c][clust_l] = dist_max
#                             maxdist[clust_l][c] = dist_max
#                         maxdist.drop(labels=clust_m, axis=0, inplace=True)
#                         maxdist.drop(labels=clust_m, axis=1, inplace=True)
#             i += 1
#         a = 1


def skater_clust(figtitle, c, adj, store=False):

    skater_w = weights.Queen.from_networkx(graph=adj)

    # g = nx.read_gpickle('custom_tree')
    # a = diss(geo_df[model])
    # for i, j in list(g.edges):
    #     g[i][j]['weight'] = True
    # tree_w = weights.Queen.from_networkx(graph=g)
    # mst_plot(g=g ,pos=cust_adj.pos, geo_df=geo_df)
    # cust_tree(c=np.triu(nx.to_numpy_array(G=cust_adj.adj_g).astype(bool)))


    spanning_forest_kwds = dict(dissimilarity=diss,
                                affinity=None,
                                reduction=np.nansum,
                                center=np.nanmean
                                )

    skat_calc = skat_lib.Skater(gdf=geo_df,
                                w=skater_w,
                                attrs_name=model,
                                n_clusters=c,
                                floor=1,
                                trace=False,
                                islands="increase",
                                spanning_forest_kwds=spanning_forest_kwds)

    # RuntimeWarnings are expected in this block for mean of empty slice np.nanmean
    print('Run Regionalization')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skat_calc.solve()

    if store:
        np.save(file=os.path.join(path_experiments, 'reg_result'), arr=skat_calc.labels_)

    stats = skat_stats(geo_df=geo_df,
                       model=model,
                       skat_labels=skat_calc.labels_,
                       or_data=handler.neighborhood_se[census_variables],
                       spanning=skat_lib.SpanningForest(**skat_calc.spanning_forest_kwds),
                       print_latex=True)
    skat_plot(geo_df=skat_calc.gdf, labels=skat_calc.labels_, title=figtitle)
    plt.savefig(fname=os.path.join(path_maps, figtitle))
    plt.close()
    # return stats['#Vertices'].std(), stats['SSD']['overall'], skat_calc.labels_


def clust_no():
    # std_list = []
    # ssd_list = []
    # ch_list = []
    # db_list = []
    # no_clust = 2
    #
    # cust_adj = Custom_Adjacency(geo_frame=geo_df)
    #
    # nan_rows = pd.isnull(handler.model_).any(axis=1)
    # ch_matrix = handler.model_[~nan_rows]
    #
    # while no_clust <= 70:
    #     print(str(no_clust))
    #     std_skat, ssd_skat, labels = skater_clust(figtitle='Regionalization by Socioeconomic Variables',
    #                                               c=no_clust,
    #                                               adj=cust_adj.adj_g)
    #     std_list.append(std_skat)
    #     ssd_list.append(ssd_skat)
    #
    #     db_list.append(metrics.davies_bouldin_score(X=ch_matrix[model].values, labels=labels[~nan_rows]))
    #     # ch_list.append(metrics.calinski_harabasz_score(X=handler.model_[model].values, labels=labels))
    #
    #     no_clust += 1
    #
    # print(std_list)
    # print(ssd_list)

    db_list = [1.310795874851597, 1.0740683006402385, 1.0740683006402385, 1.0740683006402385, 1.0740683006402385, 2.306312783675591, 3.337083261450856, 4.713962417998082, 4.713962417998082, 3.411913443978441, 3.9849153226242127, 3.6676036639023732, 3.580742776315083, 3.6045049481597005, 3.9447575842148854, 3.712519573144579, 3.678167141358887, 3.472271849379491, 3.343449975172678, 3.2621786973890012, 3.3487732608318175, 3.6700702688033844, 3.6127581137033538, 3.6336120088986688, 3.73251435654776, 3.6494022263078945, 3.075043772500411, 3.0199976675035076, 3.408631492385986, 3.201539492198936, 3.2466920268968282, 3.5555206899538616, 3.455206529019691, 3.208040276278221, 3.150656996876756, 2.820764929173691, 2.820764929173691, 2.820764929173691, 2.8618937387917747, 2.8473384361336187, 2.8314415679115266, 2.8314415679115266, 2.8244155239760866, 2.8505660912978943, 2.7903031936964324, 2.9223157084861264, 3.0319908103188387, 3.0319908103188387, 3.0866870433262, 3.0866870433262, 3.2050187074975045, 3.098342055084611, 3.067633501926536, 3.496994337816185, 3.613521978900207, 3.613521978900207, 3.3741377888761632, 3.5824209100600557, 3.7657918829683634, 3.333044834977238, 3.333044834977238, 3.5162245164776356, 3.6633258081448203, 3.485658357808851, 3.485658357808851, 3.4624156892112588, 3.3724404649511235, 3.3724404649511235, 3.3133264023244826]

    ch_list = [49.166556814552244, 58.16862504009793, 58.16862504009793, 58.16862504009793, 58.16862504009793, 55.60032005121745, 55.96951802001429, 61.55161236700713, 61.55161236700713, 58.51709011255213, 53.23498831848449, 49.91977118039292, 46.44387784130733, 44.470118867824475, 43.47342090121672, 42.27973547927433, 40.43759773066444, 39.732936066616766, 38.796958701651036, 38.07252470018797, 37.250517276001986, 37.16490348422594, 36.997726474730726, 36.979376397532704, 37.988271921034894, 36.966608812437, 36.68900590539053, 36.46248874455498, 36.75914342733096, 37.620808317227, 37.01866661788733, 36.7796241093802, 37.095627678838134, 37.18502671164866, 37.72519799593742, 37.793645649016156, 37.793645649016156, 37.793645649016156, 37.57889566946963, 36.41379727870704, 36.2238982571733, 36.2238982571733, 35.73766877767791, 35.629098532152895, 35.6342364428387, 35.29054926968965, 35.19102120664188, 35.19102120664188, 35.43154062879522, 35.43154062879522, 35.444366458400765, 35.4716349096296, 35.386559974882196, 35.179716501786864, 35.19011808353446, 35.19011808353446, 35.39064406469346, 35.49410626751077, 37.151773340169655, 37.32303897425339, 37.32303897425339, 37.432888107799094, 37.51690236807191, 38.84761345022345, 38.84761345022345, 38.9708983539591, 39.00679341951301, 39.00679341951301, 38.92079568243865]
    # std_list = [160.5, 157.12061892592794, 147.26909893117428, 137.17638280695405, 128.28538065764232, 103.10842323045887, 92.94815424621287, 76.0440093629998, 71.99368027820219, 69.5562083312787, 67.01798059533668, 64.11869541442925, 57.63754622846201, 55.705734799779215, 54.14733228994432, 52.097260653963936, 49.448969832989036, 48.37669594198386, 47.14284145021383, 46.01646162255133, 36.1863005491143, 35.11202704625726, 34.202191898570874, 23.19220558722262, 21.81596642102272, 21.481609197873393, 21.260381619363876, 21.027189243245623, 18.048976589102736, 17.767196111409653, 15.80740407166365, 15.163994862702852, 14.952433246799666, 14.85423049141434, 14.68209941538482, 14.533304744600626, 14.287825829585556, 13.972406359935983, 13.484226919247394, 13.221402079145905, 13.087985009380114, 12.987867218028912, 12.861528107208926, 12.68914185492828, 12.596864770058962, 12.32684506171518, 12.166363452563134, 11.994341312060621, 11.533412331136002, 11.281399953775404, 10.862045609971053, 10.7872508024155, 10.690409182292715, 10.520480952908924, 10.27790534816626, 10.222035910170748, 10.150072048757767, 10.042438198552752, 9.992866307980925]
    # ssd_list = [504.74, 304.84, 207.11, 150.71, 115.02, 90.93, 71.87, 56.31, 47.38, 40.75, 36.12, 32.35, 29.14, 26.41, 23.94, 21.9, 20.09, 18.47, 17.12, 15.9, 14.81, 13.73, 12.78, 11.95, 11.01, 10.35, 9.75, 9.18, 8.66, 8.09, 7.62, 7.2, 6.81, 6.45, 6.13, 5.84, 5.57, 5.33, 5.1, 4.87, 4.67, 4.48, 4.3, 4.13, 3.97, 3.82, 3.66, 3.52, 3.4, 3.27, 3.15, 3.04, 2.93, 2.83, 2.73, 2.64, 2.55, 2.46, 2.32]
    fig, ax1 = plt.subplots()
    x_ = list(range(2, len(ch_list)+2))
    ax1.plot(x_, ch_list, 'g-')
    # fig.subplots_adjust(right=0.75)
    # ax1.plot(x_, std_list, 'g-')
    #ax1.set_xlim(x_[0], 0)
    ax2 = ax1.twinx()
    ax2.plot(x_, db_list, 'b-')
    # ax2.plot(x_, ssd_list, 'b-')
    # ax3 = ax1.twinx()
    # ax3.plot(x_, ch_list, 'r-')

    # ax3.spines.right.set_position(("axes", 1.2))

    ax1.set_title('Number of Clusters for Regionalization')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Caliski-Harabasz Score', color='g')
    ax2.set_ylabel('Davies-Bouldin Score', color='b')
    # ax1.set_ylabel('$\sigma$ of Average Number of Neighborhoods per Cluster', color='g')
    ax2.set_ylabel('Average SSD per Cluster', color='b')
    # ax1.set_ylabel('Degree Distribution ', color='b')
    # ax3.set_ylabel('Edges remaining in G', color='r')
    # ax1.legend(loc='lower left')

    # min_ylim, max_ylim = plt.ylim()
    # plt.axvline(x=v_P_k, color='grey')
    # plt.text(v_P_k * 0.95, max_ylim / 2 * 0.9, str(int(int(v_P_k / 60) * 60)))
    # plt.axvline(x=v_nodes, color='black')
    # plt.text(v_nodes * 1.5, max_ylim / 2 * 1.1, str((int(v_nodes / 60) * 60)))

    plt.savefig(os.path.join(path_plot, 'explore', 'no_clust_regio'))




handler = DataHandling(new=True)
handler.matrices()


handler.edu_score()
model = list(census_variables)
model.append('edu_score')
model = model[3:]


handler.stat_prep(vars=model)

# nan_rows = pd.isnull(handler.model_).any(axis=1)
# handler.model_ = handler.model_[~nan_rows]
# handler.neighborhood_se = handler.neighborhood_se[~nan_rows]

geo_df = geopandas.GeoDataFrame(data=handler.model_,
                                crs=crs_proj,
                                geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
geo_df.reset_index(inplace=True, drop=True)


# clust_no()

no_compartments = 37

w_queen = weights.Queen.from_dataframe(df=geo_df, geom_col='geometry')
cust_adj = Adj_Islands(geo_frame=geo_df, g_init=w_queen.to_networkx())

# cust_adj = Custom_Adjacency(geo_frame=geo_df)
skater_clust(figtitle='Regionalization by Socioeconomic Variables', c=no_compartments, adj=cust_adj.adj_g, store=True)
