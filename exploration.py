from exp_resources import *

travel_times = ['Bike', 'Public Transport']


def se_year():
    year_list = range(2015, 2021, 1)
    table = np.empty(shape=(len(year_list), len(census_variables)))

    for i, year in enumerate(year_list):
        se_prep = SENeighborhoods()
        se_prep.crop_se(year=year)
        se_prep.geo_data = se_prep.geo_data.set_index(keys=column_names['geo_id_col'], drop=False)

        se_prep.filter_areas()

        # keep only relevant socio-economic variables
        for variable in census_variables:
            se_prep.extract_var(var=variable)

        missing = pd.DataFrame(se_prep.geo_data)
        missing = missing.filter(items=census_variables)
        missing = missing.apply(pd.to_numeric)
        table[i] = missing.isna().sum()

    frame = pd.DataFrame(data=table, index=year_list, columns=census_variables)
    frame['Total'] = frame.sum(axis=1)
    print(frame.to_latex(index=True))


def sna_links(handler):
    flow_prep = PassengerCounts()
    flow_prep.area_stop_matching()

    links = np.asarray(flow_prep.stop_area_association == True).nonzero()
    geo_df = geopandas.GeoDataFrame(crs=crs_proj,
                                    geometry=geopandas.GeoSeries.from_wkt(handler.neighborhood_se.geometry))

    stop_points = [Point(float(lng), float(lat)) for lng, lat in zip(
        flow_prep.stops[column_names['stop_lng']],
        flow_prep.stops[column_names['stop_lat']])]
    stop_points = geopandas.GeoSeries(stop_points, crs='epsg:4326')

    if crs_proj != None:
        stop_points = stop_points.to_crs(crs_proj)
    elif crs_proj == "":
        print("Polygon coordinates given in espg:4326")
    else:
        print('Geographic system is wrongly defined')

    lines = []
    for i in range(len(links[0])):
        lines.append(LineString([stop_points[links[1][i]], geo_df.centroid[links[0][i]]]))
    lines = geopandas.GeoSeries(lines)
    lines.to_file("sna.geojson", driver='GeoJSON')


handler = DataHandling()
handler.matrices()
sna_links(handler)
se_year(handler)
hist_modes(handler, travel_times)
hist_flows(handler)
hist_se(handler)
se_maps(handler)
