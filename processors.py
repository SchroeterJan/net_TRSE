from config import *
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import pickle
from scipy.stats import kendalltau
import geopandas
from shapely.geometry import Point
import shapely.wkt
import itertools



class SE_Neighborhoods:
    # set up DataFrame for socio-economic variables
    neighborhood_se = []

    # Initialize class
    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        self.path_se = os.path.join(dir_data, file_se)
        self.path_geo = os.path.join(dir_data, file_geo)


        self.geo_id_col = column_names['geo_id_col']
        self.pop_col = column_names['pop_col']
        self.size_col = column_names['size_col']
        self.geo_col_se = column_names['geo_id_se']
        self.year_col = column_names['year_col']
        self.se_var_col = column_names['se_var_col']
        self.se_col = column_names['se_col']

        # load geographic data set if found
        if os.path.isfile(self.path_geo):
            print('loading geo data of the city')
            self.geo_data = pd.read_csv(filepath_or_buffer=self.path_geo, sep=';')
        else:
            print('ERROR - No geographic data found at: ' + self.path_geo)

        header = open(file=self.path_se, mode='r').readline()
        header = np.array([i.strip() for i in header.split(sep=';')])
        self.geo_col_ind = np.where(header == self.geo_col_se)[0][0]
        self.year_col_ind = np.where(header == self.year_col)[0][0]
        self.se_var_col_ind = np.where(header == self.se_var_col)[0][0]
        self.se_col_ind = np.where(header == self.se_col)[0][0]
        self.neighborhood_se.append(header.tolist())



    # crop socio-economic data according to geographic data
    def crop_se(self, year):
        print('cropping socio-economic data')
        geo_ids = set(self.geo_data[self.geo_id_col])

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


    # filter by population density
    def filter_areas(self):
        print('filter low populated areas')
        self.geo_data = self.geo_data.replace(r'^\s*$', np.nan, regex=True)
        self.geo_data = self.geo_data.fillna(value=0.0)
        self.geo_data = self.geo_data.astype({self.pop_col: int, self.size_col: float})
        self.geo_data['pop_km2'] = self.geo_data[self.pop_col] / (self.geo_data[self.size_col]/1000000.0)
        self.geo_data = self.geo_data[self.geo_data['pop_km2'] > min_popdens]


class Passenger_Counts:

    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        path_passcount = os.path.join(dir_data, file_passcount)
        path_stops = os.path.join(dir_data, file_stops)

        if os.path.isfile(path_passcount):
            print('Loading Flow Data')
            self.pass_data = pd.read_csv(filepath_or_buffer=path_passcount, sep=';')
        else:
            print('ERROR - No passenger count data found in path: ' + path_passcount)

        if os.path.isfile(path_stops):
            print('Loading Stop Information')
            self.path_stops = path_stops
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

        # form shapely Polygons of all areas
        area_polygons = [shapely.wkt.loads(area_polygon) for area_polygon
                         in self.neighborhood_se[column_names['area_polygon']]]
        area_polygons = geopandas.GeoSeries(area_polygons)

        # distance Matrix of all stops (Points) and areas (Polygons)
        stops_area_distance_matrix = stop_points.apply(lambda stop: area_polygons.distance(stop))

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

        return(pd.DataFrame(area_flow_matrix,
                            index=self.stop_area_association.index,
                            columns=self.stop_area_association.index))






class transportPrep:
    # Find root direction
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PC6Info = 'PC6_VLAKKEN_BAG.csv'
    Buurten_PC6 = 'Buurten_PC6.csv'
    OTP_times = 'OTP_times.csv'
    OTP_times_old = 'OTP_times_old.csv'
    Buurten_PT_times = 'Buurten_PT_times.csv'
    Buurten_bike_times = 'Buurten_bike_times.csv'

    def __init__(self, fair=False):
        print("Initializing " + self.__class__.__name__)
        self.path_PC6Info = os.path.join(self.ROOT_DIR, raw, self.PC6Info)
        self.path_BuurtBBGA = os.path.join(self.ROOT_DIR, generated, BBGA_Buurten)
        self.path_BuurtPC6 = os.path.join(self.ROOT_DIR, generated, self.Buurten_PC6)
        self.path_OTPtimes = os.path.join(self.ROOT_DIR, raw, self.OTP_times)
        self.path_OTPtimes_old = os.path.join(self.ROOT_DIR, raw, self.OTP_times_old)
        self.path_BuurtPTtimes = os.path.join(self.ROOT_DIR, generated, self.Buurten_PT_times)
        self.path_Buurtbiketimes = os.path.join(self.ROOT_DIR, generated, self.Buurten_bike_times)
        self.BBGA_Buurt_data = pd.read_csv(filepath_or_buffer=self.path_BuurtBBGA, sep=';', index_col='Buurt_code')
        self.fair = fair



    def loadOTPtimes(self):
        self.OTP_times_data = pd.read_csv(filepath_or_buffer=self.path_OTPtimes, sep=',')
        self.OTP_times_data_old = pd.read_csv(filepath_or_buffer=self.path_OTPtimes, sep=',')

    def selectPC6(self):
        self.PC6_data = pd.read_csv(filepath_or_buffer=self.path_PC6Info, sep=';')
        if os.path.isfile(self.path_BuurtPC6):
            print('removing existing Buurten PC6 Association')
            os.remove(self.path_BuurtPC6)
        with open(file=self.path_BuurtPC6, mode='w') as BuurtPC6:
            BuurtPC6.write('Buurt_code;PC6_LAT;PC6_LNG\n')
            for Buurt_code, row in self.BBGA_Buurt_data.iterrows():
                PC6_in_Buurt_ind = np.where(np.array(self.PC6_data['Buurtcode']) == Buurt_code)[0]
                distance_list = []
                for ind in PC6_in_Buurt_ind:
                    PC6_long, PC6_lat = self.PC6_data.at[ind, 'LNG'], self.PC6_data.at[ind, 'LAT']
                    Buurt_long, Buurt_lat = row['LNG'], row['LAT']
                    distance_list.append(haversine((Buurt_lat, Buurt_long), (PC6_lat, PC6_long), unit=Unit.KILOMETERS))
                if not distance_list:
                    self.BBGA_Buurt_data.at[Buurt_code, 'PC6_lat'] = None
                    self.BBGA_Buurt_data.at[Buurt_code, 'PC6_lng'] = None
                elif self.fair==False:
                    BuurtPC6.write(Buurt_code +';' + str(self.PC6_data.at[
                        PC6_in_Buurt_ind[distance_list.index(min(distance_list))], 'LAT']) + ';' + str(self.PC6_data.at[
                        PC6_in_Buurt_ind[distance_list.index(min(distance_list))], 'LNG']) +'\n')
                    self.BBGA_Buurt_data.at[Buurt_code, 'PC6_lat'] = self.PC6_data.at[
                        PC6_in_Buurt_ind[distance_list.index(min(distance_list))], 'LAT']
                    self.BBGA_Buurt_data.at[Buurt_code, 'PC6_lng'] = self.PC6_data.at[
                        PC6_in_Buurt_ind[distance_list.index(min(distance_list))], 'LNG']
                else:
                    # to be implemented, option to collect locations fairly depending on area of Buurt
                    continue


    def Build_Matrix(self, length, data_list):
        matrix = np.zeros([length, length], dtype=float)
        k = 0
        for row in range(length):
            for col in range(length - row - 1):
                value= data_list[k]
                if value == 'None':
                    value = 0.0
                matrix[row, col + row + 1] = value
                matrix[col + row + 1, row] = value
                k += 1

        matrix[matrix == 0.0] = np.nan
        return matrix


    def prepareTransport_times(self):
        Buurten_times = pickle.load(open(os.path.join(self.ROOT_DIR, generated, 'Buurten_noParks.p'), "rb"))
        BuurtBBGA = set(self.BBGA_Buurt_data.index)
        BuurtDiff = [i for i, item in enumerate(Buurten_times) if item not in BuurtBBGA]

        minadvice_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, generated, 'minadvice_PT.p'), "rb"))
        bike_times = pickle.load(open(os.path.join(self.ROOT_DIR, generated, 'bike_time_in_seconds.p'), "rb"))

        matrix_9292 = self.Build_Matrix(length=len(Buurten_times), data_list=minadvice_9292)
        matrix_bike = self.Build_Matrix(length=len(Buurten_times), data_list=bike_times)
        matrix_9292 = np.delete(arr=matrix_9292, obj=BuurtDiff, axis=0)
        matrix_9292 = np.delete(arr=matrix_9292, obj=BuurtDiff, axis=1)
        matrix_bike = np.delete(arr=matrix_bike, obj=BuurtDiff, axis=0)
        matrix_bike = np.delete(arr=matrix_bike, obj=BuurtDiff, axis=1)

        Buurten_times = np.delete(arr=Buurten_times, obj=BuurtDiff)

        matrix_9292 = pd.DataFrame(matrix_9292, index=pd.Series(Buurten_times), columns=pd.Series(Buurten_times))
        matrix_bike = pd.DataFrame(matrix_bike, index=pd.Series(Buurten_times), columns=pd.Series(Buurten_times))
        matrix_9292 = matrix_9292.reindex(index=self.BBGA_Buurt_data.index)
        matrix_bike = matrix_bike.reindex(index=self.BBGA_Buurt_data.index)
        matrix_9292 = matrix_9292.reset_index().drop(columns='Buurt_code')
        matrix_bike = matrix_bike.reset_index().drop(columns='Buurt_code')
        matrix_9292 = matrix_9292.reindex(self.BBGA_Buurt_data.index, axis=1)
        matrix_bike = matrix_bike.reindex(self.BBGA_Buurt_data.index, axis=1)
        matrix_bike.to_csv(path_or_buf=self.path_Buurtbiketimes, sep=';', index=False)
        matrix_9292.to_csv(path_or_buf=self.path_BuurtPTtimes, sep=';', index=False)



    def compare9292(self):
        self.loadOTPtimes()

        Buurts_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, generated, 'Buurten_noParks.p'), "rb"))
        BuurtBBGA = set(self.BBGA_Buurt_data.index)  # convert to set for fast lookups
        BuurtDiff = [i for i, item in enumerate(Buurts_9292) if item not in BuurtBBGA]

        minadvice_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, generated, 'minadvice_PT.p'), "rb"))

        OTP_times = np.array(self.OTP_times_data['DURATION'])
        """
        for i, each in enumerate(OTP_times):
            try:
                a = float(each)%60
                if a <= 30.0:
                    OTP_times[i] = float(each)-a
                else:
                    OTP_times[i] = float(each)+(60.0-a)
            except:
                continue
        """

        matrix_OTP = DataHandling().Build_Matrix(length=len(np.array(self.BBGA_Buurt_data['LNG'])),
                                                 data_list=OTP_times)
        matrix_9292 = DataHandling().Build_Matrix(length=len(Buurts_9292), data_list=minadvice_9292)
        matrix_9292 = np.delete(arr=matrix_9292, obj=BuurtDiff, axis=0)
        matrix_9292 = np.delete(arr=matrix_9292, obj=BuurtDiff, axis=1)
        Buurts_9292 = np.delete(arr=Buurts_9292, obj=BuurtDiff)
        matrix_9292 = pd.DataFrame(matrix_9292, columns=pd.Series(Buurts_9292)).set_index(pd.Series(Buurts_9292))
        matrix_9292 = matrix_9292.reindex(index=self.BBGA_Buurt_data.index)
        matrix_9292 = matrix_9292.reset_index().drop(columns='Buurt_code')
        matrix_9292 = matrix_9292.reindex(self.BBGA_Buurt_data.index, axis=1)


        self.BBGA_Buurt_data['clust_9292']= Analysis().Clustering(matrix=matrix_9292,
                                                                  Buurten=np.array(self.BBGA_Buurt_data.index))
        self.BBGA_Buurt_data['clust_OTP'] = Analysis().Clustering(matrix=matrix_OTP,
                                                                   Buurten=np.array(self.BBGA_Buurt_data.index))

        self.BBGA_Buurt_data.to_csv(path_or_buf=os.path.join(self.ROOT_DIR, generated, 'clust_comp.csv'),
                                    index=True, index_label='Buurt_code', sep=';')

        clust_comp = pd.read_csv(filepath_or_buffer=os.path.join(self.ROOT_DIR, generated, 'clust_comp.csv'),
                                 sep=';')
        temp = np.array(clust_comp['clust_9292']).argsort()
        ranks_9292 = np.empty_like(temp)
        ranks_9292[temp] = np.arange(len(np.array(clust_comp['clust_9292'])))
        temp = np.array(clust_comp['clust_OTP']).argsort()
        ranks_OTP = np.empty_like(temp)
        ranks_OTP[temp] = np.arange(len(np.array(clust_comp['clust_OTP'])))
        clust_comp_kend = kendalltau(ranks_9292, ranks_OTP)

