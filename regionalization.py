from resources.exp_resources import *


handler = DataHandling(new=False)
handler.matrices()

# handler.edu_score()
# model = list(census_variables)
# model.append('edu_score')
# model = model[3:]

handler.stat_prep(vars=census_variables)

geo_df = geopandas.GeoDataFrame(data=handler.model_,
                                crs=crs_proj,
                                geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
geo_df.reset_index(inplace=True, drop=True)

w_queen = weights.Queen.from_dataframe(df=geo_df, geom_col='geometry')
cust_adj = Adj_Islands(geo_frame=geo_df, g_init=w_queen.to_networkx())

no_compartments = 18


figtitle = 'Regionalization by Socioeconomic Variables k = ' + str(no_compartments)

# cust_adj = Custom_Adjacency(geo_frame=geo_df)
skat_res = skater_clust(c=no_compartments,
                        adj=cust_adj.adj_g,
                        geo_df=geo_df,
                        store=True)


stats = skat_stats(geo_df=geo_df,
                   skat_labels=skat_res.labels_,
                   or_data=handler.neighborhood_se[census_variables],
                   model=census_variables,
                   spanning=skat_lib.SpanningForest(**skat_res.spanning_forest_kwds),
                   print_latex=True)

skat_plot(geo_df=skat_res.gdf,
          labels=skat_res.labels_,
          title=figtitle)

plt.savefig(fname=os.path.join(path_maps, figtitle))
plt.close()
