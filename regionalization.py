from resources.exp_resources import *


def reg_all():
    no_compartments_l = [18, 27]

    for no_compartments in no_compartments_l:

        handler = DataHandling(new=False)
        handler.matrices()

        # handler.edu_score()
        # model = list(census_variables)
        # model.append('edu_score')
        # model = model[3:]
        model = census_variables

        handler.stat_prep(vars=model)

        geo_df = geopandas.GeoDataFrame(data=handler.model_,
                                        crs=crs_proj,
                                        geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
        geo_df.reset_index(inplace=True, drop=True)

        w_queen = weights.Queen.from_dataframe(df=geo_df, geom_col='geometry')
        cust_adj = Adj_Islands(geo_frame=geo_df, g_init=w_queen.to_networkx())

        figtitle = 'SKATER Regionalization by \n Socioeconomic Variables with k = ' + str(no_compartments)

        # cust_adj = Custom_Adjacency(geo_frame=geo_df)
        skat_res = skater_clust(c=no_compartments,
                                adj=cust_adj.adj_g,
                                geo_df=geo_df,
                                model=model,
                                store=True)

        stats = skat_stats(geo_df=geo_df,
                           skat_labels=skat_res.labels_,
                           or_data=handler.neighborhood_se[census_variables],
                           model=model,
                           spanning=skat_lib.SpanningForest(**skat_res.spanning_forest_kwds),
                           print_latex=True)


        skat_plot(geo_df=skat_res.gdf,
                  labels=skat_res.labels_,
                  title=figtitle)

        plt.savefig(fname=os.path.join(path_maps, 'skater_se_' + str(no_compartments)))
        plt.close()


def reg_indiv():
    for i, variable in enumerate(census_variables):

        no_compartments = 27

        handler = DataHandling(new=False)
        handler.matrices()

        # handler.edu_score()
        # model = list(census_variables)
        # model.append('edu_score')
        # model = model[3:]
        model = [variable]

        handler.stat_prep(vars=model)
        handler.neighborhood_se = handler.neighborhood_se[~np.isnan(handler.model_).values]
        handler.model_ = handler.model_[~np.isnan(handler.model_).values]

        geo_df = geopandas.GeoDataFrame(data=handler.model_,
                                        crs=crs_proj,
                                        geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))
        geo_df.reset_index(inplace=True, drop=True)

        w_queen = weights.Queen.from_dataframe(df=geo_df, geom_col='geometry')
        cust_adj = Adj_Islands(geo_frame=geo_df, g_init=w_queen.to_networkx())

        figtitle = 'SKATER Regionalization by ' + census_names[i]


        # cust_adj = Custom_Adjacency(geo_frame=geo_df)
        skat_res = skater_clust(c=no_compartments,
                                adj=cust_adj.adj_g,
                                geo_df=geo_df,
                                model=model,
                                store=False)


        stats = skat_stats(geo_df=geo_df,
                           skat_labels=skat_res.labels_,
                           or_data=handler.neighborhood_se[census_variables],
                           model=model,
                           spanning=skat_lib.SpanningForest(**skat_res.spanning_forest_kwds),
                           print_latex=True)

        skat_plot(geo_df=skat_res.gdf,
                  labels=skat_res.labels_,
                  title=figtitle)

        plt.savefig(fname=os.path.join(path_maps, 'skater_' + variable))
        plt.close()


reg_indiv()

