from resources.exp_resources import *

from libpysal import weights
from spopt.region import skater as skat_lib
from scipy import spatial
import warnings



def skater_clust(figtitle):
    cust_adj = Custom_Adjacency(geo_frame=geo_df)
    skater_w = weights.Queen.from_networkx(graph=cust_adj.adj_g)

    spanning_forest_kwds = dict(dissimilarity=diss,
                                affinity=None,
                                reduction=np.nansum,
                                center=np.nanmean
                                )

    skat_calc = skat_lib.Skater(gdf=geo_df,
                                w=skater_w,
                                attrs_name=model,
                                n_clusters=no_compartments,
                                floor=1,
                                trace=False,
                                islands="increase",
                                spanning_forest_kwds=spanning_forest_kwds)

    # I expect to see RuntimeWarnings in this block for mean of empty slice np.nanmean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skat_calc.solve()

    skat_stats(skat_result=skat_calc)
    skat_plot(data=skat_calc, title=figtitle)


def skat_stats(skat_result):
    skat_stat_index = list(range(len(np.unique(skat_result.labels_))))
    skat_stat_index.append('overall')
    skat_stat = pd.DataFrame(index=skat_stat_index)

    geo_df['skater_new'] = skat_result.labels_
    geo_df['number'] = 1
    skat_stat['#Vertices'] = geo_df[['skater_new', 'number']].groupby(by='skater_new').count()

    for comp_no in range(len(np.unique(skat_result.labels_))):
        comp_data = geo_df[model][skat_result.labels_ == comp_no]
        or_data = handler.neighborhood_se[census_variables].reset_index(drop=True)[skat_result.labels_ == comp_no]
        skat_stat.at[comp_no, 'Compartment'] = comp_no + 1
        spanning = skat_lib.SpanningForest(**skat_result.spanning_forest_kwds)
        skat_stat.at[comp_no, 'SSD'] = spanning.score(data=comp_data, labels=np.zeros(len(comp_data)))
        skat_stat.at[comp_no, 'SSD/Vertice'] = round(skat_stat['SSD'][comp_no] / skat_stat['#Vertices'][comp_no], 5)
        for c_var in reversed(census_variables):
            skat_stat.at[comp_no, c_var + '_av'] = round(or_data[c_var].mean(), 2)
            skat_stat.at[comp_no, c_var + '_std'] = round(or_data[c_var].std(), 2)

    skat_stat.loc['overall'] = round(skat_stat.mean(axis=0), 2)
    print(skat_stat.to_latex(index=False))


def diss(X, Y=None):
    if Y is None:
        return spatial.distance.squareform(spatial.distance.pdist(X))
    else:
        return (X - Y) ** 2


handler = DataHandling(new=True)
handler.matrices()


handler.edu_score()
model = list(census_variables)
model.append('edu_score')
model = model[3:]

handler.stat_prep(model_variables=model)

no_compartments = 25
geo_df = geopandas.GeoDataFrame(data=handler.model_,
                                crs=crs_proj,
                                geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))

handler.neighborhood_se = handler.neighborhood_se[~pd.isnull(geo_df[model]).all(axis=1)]
geo_df.reset_index(inplace=True, drop=True)
geo_df = geo_df[~pd.isnull(geo_df[model]).all(axis=1)]

skater_clust(figtitle='Regionalization by Socioeconomic Variables')
