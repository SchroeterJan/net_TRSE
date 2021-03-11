from config import *
from Routing import *


class SENeighborhoods:
    # set up DataFrame for socio-economic variables
    neighborhood_se = []

    # Initialize class
    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        self.path_se = os.path.join(dir_data, file_se)
        self.path_geo = os.path.join(dir_data, file_geo)

        # load geographic data set if found
        if os.path.isfile(self.path_geo):
            print('loading geo data of the city')
            self.geo_data = pd.read_csv(filepath_or_buffer=self.path_geo, sep=';')
        else:
            print('ERROR - No geographic data found at: ' + self.path_geo)

        self.size_col = 'size'
        self.area_polygons()

        header = open(file=self.path_se, mode='r').readline()
        header = np.array([i.strip() for i in header.split(sep=';')])
        self.geo_col_ind = np.where(header == column_names['geo_id_se'])[0][0]
        self.year_col_ind = np.where(header == column_names['year_col'])[0][0]
        self.se_var_col_ind = np.where(header == column_names['se_var_col'])[0][0]
        self.se_col_ind = np.where(header == column_names['se_col'])[0][0]
        self.neighborhood_se.append(header.tolist())

    # crop socio-economic data according to geographic data
    def crop_se(self, year):
        print('cropping socio-economic data')
        geo_ids = set(self.geo_data[column_names['geo_id_col']])

        if os.path.isfile(self.path_se):
            with open(file=self.path_se, mode='r') as se:                                    # open socio-economic data set
                for line in se:
                    split = line.split(sep=';')
                    if split[self.geo_col_ind] in geo_ids and split[self.year_col_ind] == str(year):
                        self.neighborhood_se.append(split)
        else:
            print('ERROR - No socio-economic data found at: ' + self.path_se)

    # extract variables of interest from socio-economic data set
    def extract_var(self, var):
        print('Extracting Variable ' + var)
        self.neighborhood_se = np.array(self.neighborhood_se)
        for line in self.neighborhood_se:
            line_array = [i.strip() for i in line]               # strip to get rid of "newline"
            if line_array[self.se_var_col_ind] == var:
                self.geo_data.at[str(line_array[self.geo_col_ind]), var] = line_array[self.se_col_ind]

    def area_polygons(self):
        # form shapely Polygons of all areas
        area_polygons = [shapely.wkt.loads(area_polygon) for area_polygon
                         in self.geo_data[column_names['area_polygon']]]
        area_polygons = geopandas.GeoSeries(area_polygons)
        self.geo_data[self.size_col] = [poly.area for poly in area_polygons]

        if crs_proj != None:
            # transform polygon coordinates to geodetic lon/lat coordinates
            area_polygons_new = []
            crs_out = 'epsg:4326'
            proj = Transformer.from_crs(crs_proj, crs_out)
            for each in area_polygons:
                x, y = each.exterior.coords.xy
                polygon = []
                for each in zip(x, y):
                    lng_, lat_ = proj.transform(each[0], each[1])
                    polygon.append((lat_, lng_))
                area_polygons_new.append(Polygon(polygon))
            area_polygons = geopandas.GeoSeries(area_polygons_new)

        elif crs_proj == "":
            print("Polygon coordinates given in espg:4326")
        else:
            print('Geographic system is wrongly defined')


        # updating area polygons in data set
        self.geo_data[column_names['area_polygon']] = area_polygons
        # adding area centroids to data set
        self.geo_data['centroid'] = [P.centroid for P in area_polygons]

    # filter by population density
    def filter_areas(self):
        print('filter low populated areas')
        self.geo_data = self.geo_data.replace(r'^\s*$', np.nan, regex=True)
        self.geo_data = self.geo_data.fillna(value=0.0)
        self.geo_data = self.geo_data.astype({column_names['pop_col']: int, self.size_col: float})
        self.geo_data['pop_km2'] = self.geo_data[column_names['pop_col']] / (self.geo_data[self.size_col]/1000000.0)
        self.geo_data = self.geo_data[self.geo_data['pop_km2'] > min_popdens]


class PassengerCounts:

    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        path_passcount = os.path.join(dir_data, file_passcount)

        if os.path.isfile(path_passcount):
            print('Loading Flow Data')
            self.pass_data = pd.read_csv(filepath_or_buffer=path_passcount, sep=';')
        else:
            print('ERROR - No passenger count data found in path: ' + path_passcount)

        if os.path.isfile(path_stops):
            print('Loading Stop Information')
            self.stops = pd.read_csv(filepath_or_buffer=path_stops, sep=';')
        else:
            print('ERROR - No stop location data found in path: ' + path_stops)
            self.stops = pd.DataFrame()                                                 #option to populate stops

        if os.path.isfile(path_neighborhood_se):
            print('Loading neighborhood data')
            self.neighborhood_se = pd.read_csv(filepath_or_buffer=path_neighborhood_se, sep=';')
        else:
            print('ERROR - No neighborhood data found in path: ' + path_neighborhood_se)

        header = open(file=path_passcount, mode='r').readline()
        header = np.array([i.strip() for i in header.split(sep=';')])
        self.or_ind = np.where(header == column_names['pass_or'])[0][0]
        self.dest_ind = np.where(header == column_names['pass_dest'])[0][0]
        self.flow_ind = np.where(header == column_names['pass_vol'])[0][0]

    def area_stop_matching(self):
        print('Matching areas and stops according to proximity conditions')
        # drop Stops without assigned location
        self.stops = self.stops.mask(self.stops.eq('None')).dropna()
        self.stops = self.stops.reset_index(drop=True)

        # form shapely Points for all Stops
        stop_points = [Point(float(lng), float(lat)) for lng, lat in zip(
            self.stops[column_names['stop_lng']],
            self.stops[column_names['stop_lat']])]
        stop_points = geopandas.GeoSeries(stop_points)

        # reload areas into shapely/geopandas
        area_polygons = [shapely.wkt.loads(area_polygon) for area_polygon
                         in self.neighborhood_se[column_names['area_polygon']]]
        area_polygons = geopandas.GeoSeries(area_polygons)

        # distance Matrix of all stops (Points) and areas (Polygons)
        stops_area_distance_matrix = \
            stop_points.apply(lambda stop: area_polygons.distance(stop))

        # extract stops inside or proximate to relevant areas
        short_distance, a = np.where(stops_area_distance_matrix <= proximity)
        relevant_stops = np.unique(short_distance)
        self.stops = self.stops.iloc[relevant_stops]

        # exclude stops marked to be excluded
        for exclude in exclude_stops:
            self.stops = self.stops[self.stops[column_names['stop_name']] != exclude]

        # shrink distance matrix to relevant stops and transpose to be area-focussed
        stops_area_distance_matrix = stops_area_distance_matrix.iloc[np.array(self.stops.index)]
        stops_area_distance_matrix = np.array(stops_area_distance_matrix).T

        # run area-stop assignment-scheme
        for i, area_stop_distance in enumerate(stops_area_distance_matrix):
            # check for stops in or proximate to area
            proxi_stops = np.where(area_stop_distance <= proximity)[0].tolist()
            # else check for closest stop and except all stops up to a certain range further than this
            if len(proxi_stops) == 0:
                mindist = np.min(area_stop_distance[np.nonzero(area_stop_distance)])
                extra_mile = mindist * (1.0 + range_factor)
                proxi_stops = np.where(area_stop_distance <= extra_mile)[0].tolist()
            # marking assigned stops
            area_stop_distance[proxi_stops] = True
            stops_area_distance_matrix[i] = area_stop_distance

        # bool all non assigned stops
        stops_area_distance_matrix[np.where(stops_area_distance_matrix != True)] = False
        self.stop_area_association = pd.DataFrame(stops_area_distance_matrix,
                                                  index=self.neighborhood_se[column_names['geo_id_col']],
                                                  columns=self.stops[column_names['stop_name']])

    def filter_passcount(self):
        print('Filter irrelevant connections')
        # set up pure route frame
        connections = pd.DataFrame(data={'or': self.pass_data[column_names['pass_or']],
                                         'dest': self.pass_data[column_names['pass_dest']]})
        # set of relevant Stops
        stops = set(self.stops[column_names['stop_name']])
        # check if connections contain relevant stops
        rel_connections = connections.isin(stops)
        # keep only connections between relevant stops
        rel_connections = rel_connections.all(axis=1)
        # keep only passenger counts for those connections
        self.pass_data = self.pass_data.loc[rel_connections.values]

    def assign_passcounts(self):
        print('Form flows between areas')
        # set up flow matrix
        stops_map = {stop: index for index, stop in enumerate(list(self.stops[column_names['stop_name']]))}
        flow_matrix = np.zeros(shape=[len(self.stops[column_names['stop_name']]),
                                      len(self.stops[column_names['stop_name']])])

        # fill flow matrix
        for route in np.array(self.pass_data):
            flow = route[self.flow_ind]
            or_ = route[self.or_ind]
            dest_ = route[self.dest_ind]
            try:
                flow = int(flow)
                flow_matrix[stops_map[or_]][stops_map[dest_]] = flow
            except:
                continue

        # set up area flow matrix
        area_flow_matrix = np.zeros((len(self.stop_area_association.index),
                                      len(self.stop_area_association.index)))
        # flow assignment scheme
        for origin, flow_row in enumerate(flow_matrix):
            # get associated stops for the origin area
            or_associations = np.where(self.stop_area_association.to_numpy()[:, origin] == 1.0)[0]
            for destination, flow in enumerate(flow_row):
                # get associated stops for the destination area
                dest_associations = np.where(self.stop_area_association.to_numpy()[:, destination] == 1.0)[0]
                # combile possible trip combinations between potential origins and potential destinations
                trip_combinations = np.array(list(itertools.product(or_associations, dest_associations)))
                # avoid self loops
                trip_combinations = trip_combinations[np.invert([np.all(combination == combination[0])
                                                                 for combination in trip_combinations])]
                # assign equal flow fractions among the potential journeys
                for combination in trip_combinations:
                    area_flow_matrix[combination[0], combination[1]] += (flow / len(trip_combinations))
        area_flow_matrix[area_flow_matrix == 0.0] = np.nan
        return(pd.DataFrame(area_flow_matrix,
                            index=self.stop_area_association.index,
                            columns=self.stop_area_association.index))


class TransportPrep:

    def __init__(self, fair=False):
        print("Initializing " + self.__class__.__name__)

        if os.path.isfile(path_neighborhood_se):
            print('Loading neighborhood data')
            self.neighborhood_se = pd.read_csv(filepath_or_buffer=path_neighborhood_se, sep=';')
        else:
            print('ERROR - No neighborhood data found in path: ' + path_neighborhood_se)

        if os.path.isfile(os.path.join(dir_data, file_locations)):
            print('Loading provided locations for travel times')
            self.locations = pd.read_csv(filepath_or_buffer=os.path.join(dir_data, file_locations), sep=';')

    def get_gh_times(self):
        grabber = GH_grabber()
        if os.path.isfile(path_bike_scrape):
            print('removing existing Graphhopper times')
            os.remove(path_bike_scrape)

        with open(file=path_bike_scrape, mode='w') as GHtimes:
            GHtimes.write('ORIGING_LAT,ORIGIN_LNG,DESTINATION_LAT,DESTINATION_LNG,DURATION,DISTANCE,WEIGHT\n')
            for i, or_row in enumerate(self.locations):
                for j, dest_row in enumerate(self.locations[i + 1:]):
                    res = grabber.bike_planner(or_=or_row, dest_=dest_row)
                    GHtimes.write(res)

                print('Finished row ' + str(i))

    def build_matrix(self, length: int, data_list: list):
        matrix = np.zeros([length, length], dtype=float)
        k = 0
        for row in range(length):
            for col in range(length - row - 1):
                value = data_list[k]
                if value == 'None':
                    value = 0.0
                matrix[row, col + row + 1] = value
                matrix[col + row + 1, row] = value
                k += 1
        matrix[matrix == 0.0] = np.nan
        return matrix

    def order_times(self):
        if os.path.isfile(path_bike_scrape):
            print('Loading bike times')
            bike_data = pd.read_csv(filepath_or_buffer=path_bike_scrape, sep=',')
        else:
            print('ERROR - no bike times data found in path: ' + path_bike_scrape)

        times_matrix = self.build_matrix(length=len(self.locations[column_names['geo_id_col']]),
                                         data_list=list(bike_data['DURATION']/1000.0))
        matrix_bike = pd.DataFrame(times_matrix,
                                   index=self.locations[column_names['geo_id_col']],
                                   columns=self.locations[column_names['geo_id_col']])
        matrix_bike = matrix_bike.reindex(index=self.neighborhood_se[column_names['geo_id_col']])
        matrix_bike = matrix_bike.reindex(columns=self.neighborhood_se[column_names['geo_id_col']])
        return matrix_bike

    #
    # def compare9292(self):
    #     self.loadOTPtimes()
    #
    #     Buurts_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, generated, 'Buurten_noParks.p'), "rb"))
    #     BuurtBBGA = set(self.BBGA_Buurt_data.index)  # convert to set for fast lookups
    #     BuurtDiff = [i for i, item in enumerate(Buurts_9292) if item not in BuurtBBGA]
    #
    #     minadvice_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, generated, 'minadvice_PT.p'), "rb"))
    #
    #     OTP_times = np.array(self.OTP_times_data['DURATION'])
    #     """
    #     for i, each in enumerate(OTP_times):
    #         try:
    #             a = float(each)%60
    #             if a <= 30.0:
    #                 OTP_times[i] = float(each)-a
    #             else:
    #                 OTP_times[i] = float(each)+(60.0-a)
    #         except:
    #             continue
    #     """
    #
    #     matrix_OTP = DataHandling().build_matrix(length=len(np.array(self.BBGA_Buurt_data['LNG'])),
    #                                              data_list=OTP_times)
    #     matrix_9292 = DataHandling().build_matrix(length=len(Buurts_9292), data_list=minadvice_9292)
    #     matrix_9292 = np.delete(arr=matrix_9292, obj=BuurtDiff, axis=0)
    #     matrix_9292 = np.delete(arr=matrix_9292, obj=BuurtDiff, axis=1)
    #     Buurts_9292 = np.delete(arr=Buurts_9292, obj=BuurtDiff)
    #     matrix_9292 = pd.DataFrame(matrix_9292, columns=pd.Series(Buurts_9292)).set_index(pd.Series(Buurts_9292))
    #     matrix_9292 = matrix_9292.reindex(index=self.BBGA_Buurt_data.index)
    #     matrix_9292 = matrix_9292.reset_index().drop(columns='Buurt_code')
    #     matrix_9292 = matrix_9292.reindex(self.BBGA_Buurt_data.index, axis=1)
    #
    #
    #     self.BBGA_Buurt_data['clust_9292']= Analysis().Clustering(matrix=matrix_9292,
    #                                                               Buurten=np.array(self.BBGA_Buurt_data.index))
    #     self.BBGA_Buurt_data['clust_OTP'] = Analysis().Clustering(matrix=matrix_OTP,
    #                                                                Buurten=np.array(self.BBGA_Buurt_data.index))
    #
    #     self.BBGA_Buurt_data.to_csv(path_or_buf=os.path.join(self.ROOT_DIR, generated, 'clust_comp.csv'),
    #                                 index=True, index_label='Buurt_code', sep=';')
    #
    #     clust_comp = pd.read_csv(filepath_or_buffer=os.path.join(self.ROOT_DIR, generated, 'clust_comp.csv'),
    #                              sep=';')
    #     temp = np.array(clust_comp['clust_9292']).argsort()
    #     ranks_9292 = np.empty_like(temp)
    #     ranks_9292[temp] = np.arange(len(np.array(clust_comp['clust_9292'])))
    #     temp = np.array(clust_comp['clust_OTP']).argsort()
    #     ranks_OTP = np.empty_like(temp)
    #     ranks_OTP[temp] = np.arange(len(np.array(clust_comp['clust_OTP'])))
    #     clust_comp_kend = kendalltau(ranks_9292, ranks_OTP)
