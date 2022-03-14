import itertools

import geopandas
import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import Point

from resources.config import *


def build_matrix(length: int, data_list: list):
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


class SENeighborhoods:
    # set up DataFrame for socio-economic variables

    # Initialize class
    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        self.neighborhood_se = []
        self.path_se = os.path.join(dir_raw, file_se)
        self.path_geo = os.path.join(dir_raw, file_geo)

        # load geographic data set if found
        if os.path.isfile(self.path_geo):
            print('loading geo data of the city')
            self.geo_data = geopandas.read_file(self.path_geo)
        else:
            print('ERROR - No geographic data found at: ' + self.path_geo)

        # calculate area of geographic units in km^2 (depending on crs unit, here m)
        self.geo_data['area'] = self.geo_data['geometry'].area / 10 ** 6

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
        if os.path.isfile(self.path_se):
            with open(file=self.path_se, mode='r') as se:  # open socio-economic data set
                for line in se:
                    # split columns by seperator and strip "newline"
                    split = [x.strip() for x in line.split(sep=';')]
                    if split[self.year_col_ind] == str(year):
                        self.neighborhood_se.append(split)
        else:
            print('ERROR - No socio-economic data found at: ' + self.path_se)

    # extract variables of interest from socio-economic data set
    def extract_var(self, var):
        print('Extracting Variable ' + var)
        self.neighborhood_se = np.array(self.neighborhood_se)
        geo_ids = set(self.geo_data[column_names['geo_id_col']])
        for line in self.neighborhood_se:
            if line[self.se_var_col_ind] == var and line[self.geo_col_ind] in geo_ids:
                self.geo_data.at[str(line[self.geo_col_ind]), var] = line[self.se_col_ind]

    def filter_areas(self):
        # get population of areas
        self.extract_var(var=column_names['pop_col'])
        # calculate population per square kilometer
        self.geo_data['pop_area'] = pd.to_numeric(self.geo_data['BEVTOTAAL']) / self.geo_data['area']
        # filter for populated areas (over 150 people per square kilometer)
        self.geo_data = self.geo_data[self.geo_data['pop_area'] > min_popdens]


class PassengerCounts:

    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        path_passcount = os.path.join(dir_raw, file_passcount)

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
            self.stops = pd.DataFrame()  # option to populate stops

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
        stop_points = geopandas.GeoSeries(stop_points, crs='epsg:4326')

        # reload areas into shapely/geopandas
        area_polygons = [shapely.wkt.loads(area_polygon) for area_polygon
                         in self.neighborhood_se['geometry']]
        area_polygons = geopandas.GeoSeries(area_polygons, crs=crs_proj)

        if crs_proj != None:
            stop_points = stop_points.to_crs(crs_proj)
        elif crs_proj == "":
            print("Polygon coordinates given in espg:4326")
        else:
            print('Geographic system is wrongly defined')



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


    def filter_connections(self):
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
        return (pd.DataFrame(area_flow_matrix,
                             index=self.stop_area_association.index,
                             columns=self.stop_area_association.index))


class ODPrep:

    def __init__(self, fair=False):
        print("Initializing " + self.__class__.__name__)
        if os.path.isfile(path_neighborhood_se):
            print('Loading neighborhood data')
            self.neighborhood_se = pd.read_csv(filepath_or_buffer=path_neighborhood_se, sep=';')
        else:
            print('ERROR - No neighborhood data found in path: ' + path_neighborhood_se)

        if os.path.isfile(os.path.join(dir_raw, file_locations)):
            print('Loading provided locations for travel times')
            self.locations = pd.read_csv(filepath_or_buffer=os.path.join(dir_raw, file_locations), sep=';')

    def load_data(self, path):
        if os.path.isfile(path):
            print('Loading ' + str(path))
            return pd.read_csv(filepath_or_buffer=path, sep=',')
        else:
            print('ERROR while trying to load ' + str(path))

    def matrix_to_frame(self, matrix):
        od_matrix = pd.DataFrame(matrix,
                                 index=self.neighborhood_se[column_names['geo_id_col']],
                                 columns=self.neighborhood_se[column_names['geo_id_col']])
        return od_matrix

    def calc_euclid(self):
        geo_df = geopandas.GeoDataFrame(crs=crs_proj,
                                        geometry=geopandas.GeoSeries.from_wkt(self.neighborhood_se.geometry))
        # iterate over all centroid locations and use geopandas distance function to calculate the distance to all other centroid locations
        distances = np.array([geo_df.centroid.distance(centroid) for centroid in geo_df.centroid])
        return distances
