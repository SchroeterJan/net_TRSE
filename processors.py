from config import *
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import pickle
from scipy.stats import kendalltau




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


    # extract variables of interest from Buurten BBGA
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


    # relevant_stop_locations = 'relevant_stop_Locations.csv'
    # relevant_PassCount = 'relevant_PassCount.csv'
    # Buurten_Stops_Associations = 'Buurten_Stops_associations.csv'
    # Buurten_flows = 'Buurten_flows.csv'

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


    def relevantStops(self):
        # Drop Stops without assigned location
        self.stops = self.stops.mask(self.stops.eq('None')).dropna()
        self.stops = self.stops.reset_index(drop=True)

        # Form shapely Points for all Stops
        GVB_Stop_Points = [Point(float(lng), float(lat)) for lng, lat in zip(
            self.stops['LNG'],
            self.stops['LAT'])]
        GVB_Stop_Points = geopandas.GeoSeries(GVB_Stop_Points)

        # Form shapely Polygons of all Buurten
        Buurt_Polygons = [shapely.wkt.loads(Buurt_Polygon) for Buurt_Polygon
                          in BuurtBBGA.BBGA_Buurt_data['WKT_LNG_LAT']]
        Buurt_Polygons = geopandas.GeoSeries(Buurt_Polygons)

        # Distance Matrix of all Stops (Points) and Buurten (Polygons)
        Stops_Buurten_distance_matrix = GVB_Stop_Points.apply(lambda Stop: Buurt_Polygons.distance(Stop))
        Short_Distances, a = np.where(Stops_Buurten_distance_matrix <= proximity_measure)
        relevant_Stops = np.unique(Short_Distances)
        self.stops = self.stops.iloc[relevant_Stops]

        # Exclude stops with connections to regional trains
        for NS_stop in exclude_stops:
            self.stops = self.stops[self.stops.Stop_name != NS_stop]

        # shrink distance matrix to relevant Stops and Transpose to be Buurt-focussed
        Stops_Buurten_distance_matrix = Stops_Buurten_distance_matrix.iloc[np.array(self.stops.index)]
        Stops_Buurten_distance_matrix = np.array(Stops_Buurten_distance_matrix).T

        # run Buurt-Stop-Assignment scheme
        for i, Buurt_Stop_Distances in enumerate(Stops_Buurten_distance_matrix):
            # check for stops in proximity and within the Buurt
            Stops = np.where(Buurt_Stop_Distances <= proximity_measure)[0].tolist()
            # else check for closest Stop and except all Stops up to 10% further than this
            if len(Stops) == 0:
                mindist = np.min(Buurt_Stop_Distances[np.nonzero(Buurt_Stop_Distances)])
                extra_mile = mindist * (1.0 + range_factor)
                Stops = np.where(Buurt_Stop_Distances <= extra_mile)[0].tolist()
            # marking assigned stops
            Buurt_Stop_Distances[Stops] = True
            Stops_Buurten_distance_matrix[i] = Buurt_Stop_Distances

        # bool all non assigned stops
        Stops_Buurten_distance_matrix[np.where(Stops_Buurten_distance_matrix != True)] = False
        Stops_Buurten_distance_matrix = pd.DataFrame(Stops_Buurten_distance_matrix,
                                                     index=BuurtBBGA.BBGA_Buurt_data.index,
                                                     columns=self.stops.Stop_name)

        Stops_Buurten_distance_matrix.to_csv(path_or_buf=self.path_BuurtStopsAss, sep=';')
        self.stops.to_csv(path_or_buf=self.path_relStop_locations, sep=';')

    def filterPassCount(self):
        self.relevantStop_locations = load(path=self.path_relStop_locations, sep=';')

        # drop Counts column to check the connections
        connections = self.PassCount_data.drop(labels='Totaal_reizen', axis='columns')
        # set of relevant Stops
        Stops = set(self.relevantStop_locations['Stop_name'])
        # check if connections contain relevant stops
        rel_connections = connections.isin(Stops)
        # keep only connections between relevant stops
        rel_connections = rel_connections.all(axis=1)
        # keep only passenger counts for those connections
        self.PassCount_data = self.PassCount_data.loc[rel_connections.values]
        self.PassCount_data.to_csv(path_or_buf=self.path_relPassCount, sep=';')

    def assignPassCounts(self):
        self.relevant_PassCount_data = load(path=self.path_relPassCount, sep=';')
        self.Buurten_Stops_Association = load(path=self.path_BuurtStopsAss, sep=';', index_col=0)
        self.relevant_Locations = load(path=self.path_relStop_locations, sep=';')

        # set up flow matrix
        flow_matrix = pd.DataFrame(index=self.relevant_Locations.Stop_name, columns=self.relevant_Locations.Stop_name)

        # fill flow matrix
        for flow in np.array(self.relevant_PassCount_data):
            flow_matrix.at[flow[1], flow[2]] = flow[3]

        # discard all flows below 50
        flow_matrix = flow_matrix.where(flow_matrix != '<50', other=np.nan)
        flow_matrix = flow_matrix.fillna(value=0.0)
        flow_matrix = flow_matrix.astype(float)

        # set up Buurt flow matrix
        Buurt_flow_matrix = np.zeros((len(self.Buurten_Stops_Association.index),
                                      len(self.Buurten_Stops_Association.index)))
        # flow assignment scheme
        for origin, flow_row in enumerate(flow_matrix.to_numpy()):
            # get associated stops for the origin Buurt
            or_associations = np.where(self.Buurten_Stops_Association.to_numpy()[:, origin] == 1.0)[0]
            for destination, flow in enumerate(flow_row):
                # get associated stops for the destination Buurt
                dest_associations = np.where(self.Buurten_Stops_Association.to_numpy()[:, destination] == 1.0)[0]
                # combile possible trip combinations between potential origins and potential destinations
                trip_combinations = np.array(list(itertools.product(or_associations, dest_associations)))
                # avoid self loops
                trip_combinations = trip_combinations[np.invert([np.all(combination == combination[0])
                                                                 for combination in trip_combinations])]
                # assign equal flow fractions among the potential journeys
                for combination in trip_combinations:
                    Buurt_flow_matrix[combination[0], combination[1]] += (flow / len(trip_combinations))

        Buurt_flow_matrix = pd.DataFrame(Buurt_flow_matrix,
                                         index=self.Buurten_Stops_Association.index,
                                         columns=self.Buurten_Stops_Association.index)
        Buurt_flow_matrix.to_csv(path_or_buf=self.path_Buurtflows, sep=';')





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

