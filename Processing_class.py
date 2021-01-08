from Classes import *
import os
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import json
import pickle
from scipy.stats import kendalltau
from shapely.geometry import Point
import shapely.wkt
import geopandas
import itertools


global RawData, GeneratedData, BBGA_Buurten, load
RawData = 'Raw_data'
GeneratedData = 'Generated_data'
BBGA_Buurten = 'BBGA_Buurt.csv'


def load(path, sep, index_col=None):
    return pd.read_csv(filepath_or_buffer=path, sep=sep, index_col=index_col)

class BBGAPrep:
    # Find root direction
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Declare paths to data
    BBGA = 'bbga_latest_and_greatest.csv'
    BuurtenInfo = 'GEBIED_BUURTEN.csv'


    # Initialize class
    def __init__(self, vars, year):
        print("Initializing " + self.__class__.__name__)
        self.path_BBGA = os.path.join(self.ROOT_DIR, RawData, self.BBGA)
        self.path_BuurtBBGA = os.path.join(self.ROOT_DIR, GeneratedData, BBGA_Buurten)
        self.path_BuurtenInfo = os.path.join(self.ROOT_DIR, RawData, self.BuurtenInfo)
        self.df_vars = pd.DataFrame()  # DataFrame for extracted BBGA variables

        ###CROP BBGA TO BUURT LEVEL
        self.buurtcrop()
        ###EXTRACT VARIABLES
        for variable in vars:
            self.extract_var(var=variable, year=year)
        ###FILTER BUURTEN BY POPULATION PER KM2
        self.filterbuurten()
        ###MERGE DATA FRAMES AND STORE
        self.mergestore()

    # Load data about Buurten
    def loadbuurten(self):
        self.Buurten_data = pd.read_csv(filepath_or_buffer=self.path_BuurtenInfo, sep=';')

    # crop BBGA to Buurten level
    def buurtcrop(self):
        if 'Buurten_data' not in globals():
            self.loadbuurten()
        if os.path.isfile(self.path_BuurtBBGA):
            print('removing existing BBGA Buurten')
            os.remove(self.path_BuurtBBGA)
        print('cropping BBGA')
        Buurten_Codes = set(self.Buurten_data['Buurt_code'])
        with open(file=self.path_BBGA, mode='r') as BBGA:                           # open BBGA
            with open(file=self.path_BuurtBBGA, mode='w') as BuurtBBGA:             # create new Buurten BBGA file
                for line in BBGA:
                    if line.split(sep=';')[1] in Buurten_Codes:
                        BuurtBBGA.write(line)

    # extract variables of interest from Buurten BBGA
    def extract_var(self, var, year):
        print('Extracting Variable ' + var)
        with open(file=self.path_BuurtBBGA, mode='r') as Buurt_BBGA:
            for line in Buurt_BBGA:
                line_array = [i.strip() for i in line.split(sep=';')]               # strip to get rid of "newline"
                if line_array[2] == var and int(line_array[0]) == year:
                    self.df_vars.at[str(line_array[1]), var] = line_array[3]

    # filter by population density
    def filterbuurten(self):
        if 'Buurten_data' not in globals():
            self.loadbuurten()
        del_list = []                                                               # declare filter list
        for i, Buurten_code in enumerate(self.Buurten_data['Buurt_code']):
            try:
                Pop_km2 = float(self.df_vars.at[Buurten_code, 'BEVTOTAAL'], ) / (
                        self.Buurten_data.at[i, 'Opp_m2'] / 1000000.0)
                if Pop_km2 <= 100.0:
                    del_list.append(Buurten_code)
            except:
                del_list.append(Buurten_code)
        self.df_vars = self.df_vars.drop(index=del_list, axis=0)                    # apply filter


    def mergestore(self):
        print('merging Buurt and BBGA frame')
        if os.path.isfile(self.path_BuurtBBGA):
            self.loadbuurten()
            self.df_vars = self.df_vars.join(self.Buurten_data.set_index('Buurt_code'))
            print('storing new Buurt BBGA csv')
            self.df_vars.to_csv(path_or_buf=self.path_BuurtBBGA, index=True, index_label='Buurt_code', sep=';')
        else:
            print('ERROR while merging frames')


class PassengerCountPrep:
    # Find root direction
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PassCountfile = 'HBReizen_Schroter.csv'
    GTFS_Stopsfile = 'stops_amsterdam.txt'
    OSM_stops = 'OSM_PT_stations_ams.csv'
    GVB_stop_locations = 'GVB_Stop_Locations.csv'
    relevant_stop_locations = 'relevant_stop_Locations.csv'
    relevant_PassCount = 'relevant_PassCount.csv'
    Buurten_Stops_Associations = 'Buurten_Stops_associations.csv'
    Buurten_flows = 'Buurten_flows.csv'



    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        self.path_PassCount = os.path.join(self.ROOT_DIR, RawData, self.PassCountfile)
        self.path_GTFS_Stops = os.path.join(self.ROOT_DIR, RawData, self.GTFS_Stopsfile)
        self.path_OSMStops = os.path.join(self.ROOT_DIR, RawData, self.OSM_stops)
        self.path_GVBStop_Locations = os.path.join(self.ROOT_DIR, GeneratedData, self.GVB_stop_locations)
        self.path_relStop_locations = os.path.join(self.ROOT_DIR, GeneratedData, self.relevant_stop_locations)
        self.path_relPassCount = os.path.join(self.ROOT_DIR, GeneratedData, self.relevant_PassCount)
        self.path_BuurtStopsAss = os.path.join(self.ROOT_DIR, GeneratedData, self.Buurten_Stops_Associations)
        self.path_Buurtflows = os.path.join(self.ROOT_DIR, GeneratedData, self.Buurten_flows)
        self.PassCount_data = load(path=self.path_PassCount, sep=';')


    def loadStopData(self):

        # Read OSM Stop location extract from OpenStreetMaps
        self.OSMStops_data = pd.read_csv(filepath_or_buffer=self.path_OSMStops, sep=';')
        if os.path.isfile(self.path_GVBStop_Locations):
            self.GVBStop_locations = pd.read_csv(filepath_or_buffer=self.path_GVBStop_Locations, sep=';')
        if os.path.isfile(self.path_relStop_locations):
            self.relevantStop_locations = pd.read_csv(filepath_or_buffer=self.path_relStop_locations, sep=';')


    def checkStopLocations_old(self):
        self.loadStopData()
        self.GTFSStops_data['stop_name'] = self.GTFSStops_data['stop_name'].str.split(', ').str[1]
        GTFSStop_names = set(self.GTFSStops_data['stop_name'])
        OSMStop_names = self.OSMStops_data['name'].astype(str)
        OSMStop_names2 = self.OSMStops_data['name'].str.split(', ').str[1]
        OSMstop_nan = np.array(OSMStop_names2.isna())

        for i, each in enumerate(OSMstop_nan):
            if each == False:
                OSMStop_names.at[i] = OSMStop_names2.at[i]

        OSMStop_names = np.unique(OSMStop_names).astype(str)
        GVBStop_names = np.unique(np.array(self.PassCount_data['Halte_(vertrek)'])).astype(str)
        gtfsgood = []
        gtfsbad = []
        for each in GVBStop_names:
            if each in set(GTFSStop_names):
                gtfsgood.append(each)
            else:
                gtfsbad.append(each)
        OSMgood = []
        OSMbad = []
        for each in GVBStop_names:
            if each in set(OSMStop_names):
                OSMgood.append(each)
            else:
                OSMbad.append(each)

    def assignStopLocations(self):
        # Read GTFS Stop data (source: https://transitfeeds.com/)
        self.GTFSStops_data = load(path=self.path_GTFS_Stops, sep=',')

        self.GTFSStops_data['stop_name'] = self.GTFSStops_data['stop_name'].str.split(', ').str[1]

        GTFSStop_names = set(self.GTFSStops_data['stop_name'])

        GVBStops_origin = np.unique(np.array(self.PassCount_data['Halte_(vertrek)'])).astype(str)
        GVBStops_dest = np.unique(np.array(self.PassCount_data['Halte_(aankomst)'])).astype(str)
        GVBStops = np.unique(list(GVBStops_dest) + list(GVBStops_origin))

        gtfs_found = []
        gtfs_notfound = []
        for each in GVBStops:
            if each in set(GTFSStop_names):
                gtfs_found.append(each)
            else:
                gtfs_notfound.append(each)

        """
        hits = {}

        for unfound in gtfs_notfound:
            try:
                parts = unfound.split(' ')
            except:
                print(unfound)
            leftover = []
            for possible_hit in list(GTFSStop_names):
                for part in parts:
                    if np.char.find(str(possible_hit), part, start=0, end=None) != -1:
                        leftover.append(possible_hit)

            print('For \"' + unfound + '\" the folling possible hits were found: \n')
            for i, item in enumerate(leftover):
                print(str(i) + '. ' + item + '\n')
            choice = input('Choose the best fit via index. Type \" None \" to discard the Stop or \" further \n.')
            if choice == 'None':
                continue
            else:
                hits[unfound] = leftover[int(choice)]

        with open('Generated_data/stop_assignment1.txt', 'w') as file:
            file.write(json.dumps(hits))
        """

        with open(os.path.join(self.ROOT_DIR, GeneratedData, 'stop_assignment1.txt')) as assigned_dict:
            assigned_stops = json.load(assigned_dict)
        gtfs_notfound = [str(item) for item in gtfs_notfound if item not in list(assigned_stops.keys())]

        """
        hits = {}
        for unfound in gtfs_notfound:
            try:
                parts = unfound.split(' ')
                print('current parts are: \n')
                for each in parts:
                    print(each)
                additional = input('add an educated guess')
                if len(additional) > 0:
                    parts.append(additional)
            except:
                print(unfound)
            leftover = []
            for possible_hit in list(GTFSStop_names):
                for part in parts:
                    if np.char.find(str(possible_hit), part, start=0, end=None) != -1:
                        leftover.append(str(possible_hit))

            print('For \"' + unfound + '\" the folling possible hits were found: \n')
            for i, item in enumerate(leftover):
                print(str(i) + '. ' + item + '\n')
            choice = input('Choose the best fit via index. Type \" None \" to discard the Stop or \" further \n.')
            if choice == 'None':
                continue
            else:
                hits[unfound] = leftover[int(choice)]

        with open('Generated_data/stop_assignment2.txt', 'w') as file:
            file.write(json.dumps(hits))
        """

        with open(os.path.join(self.ROOT_DIR, GeneratedData, 'stop_assignment2.txt')) as assigned_dict:
            assigned_stops = {**assigned_stops, **json.load(assigned_dict)}

        GTFSStop_dict = {}
        for each in np.array(self.GTFSStops_data):
            GTFSStop_dict[each[2]] = [each[3], each[4]]
        GVBStop_locations = pd.DataFrame(columns=['Stop_name', 'LAT', 'LNG'])
        GVBStop_locations['Stop_name'] = pd.Series(GVBStops)

        for i, Stop in enumerate(GVBStops):
            if Stop in assigned_stops:
                Stop = assigned_stops[Stop]
            if Stop in GTFSStop_names:
                GVBStop_locations.iloc[i][1] = GTFSStop_dict[Stop][0]
                GVBStop_locations.iloc[i][2] = GTFSStop_dict[Stop][1]
            else:
                GVBStop_locations.iloc[i][1] = input('give LAT for' + Stop)
                GVBStop_locations.iloc[i][2] = input('give LNG for' + Stop)
        # GVBStop_locations.to_csv(path_or_buf=self.path_GVBStop_Locations, index=True, index_label='Buurt_code', sep=';')


    def relevantStops(self):
        self.GVBStop_locations = load(path=self.path_GVBStop_Locations, sep=';')

        # Declare actual Train Stations allowing to switch to regional transport (source:
        # https://en.wikipedia.org/wiki/List_of_railway_stations_in_Amsterdam)
        NS_stops = {'Centraal Station',
                    'Station Sloterdijk',
                    'Station Lelylaan',
                    'Station Zuid',
                    'Station RAI',
                    'Muiderpoortstation',
                    'Amstelstation',
                    'Station Bijlmer ArenA',
                    'Station Science Park',
                    'Station Holendrecht',
                    'Station Duivendrecht'
                    }

        BuurtBBGA = transportPrep()
        BuurtBBGA.loadBuurtBBGA()

        # Drop Stops without assigned location
        self.GVBStop_locations = self.GVBStop_locations.mask(self.GVBStop_locations.eq('None')).dropna()
        self.GVBStop_locations = self.GVBStop_locations.reset_index(drop=True)

        # Form shapely Points for all Stops
        GVB_Stop_Points = [Point(float(lng), float(lat)) for lng, lat in zip(
            self.GVBStop_locations['LNG'],
            self.GVBStop_locations['LAT'])]
        GVB_Stop_Points = geopandas.GeoSeries(GVB_Stop_Points)

        # Form shapely Polygons of all Buurten
        Buurt_Polygons = [shapely.wkt.loads(Buurt_Polygon) for Buurt_Polygon
                          in BuurtBBGA.BBGA_Buurt_data['WKT_LNG_LAT']]
        Buurt_Polygons = geopandas.GeoSeries(Buurt_Polygons)

        # Distance Matrix of all Stops (Points) and Buurten (Polygons)
        Stops_Buurten_distance_matrix = GVB_Stop_Points.apply(lambda Stop: Buurt_Polygons.distance(Stop))
        Short_Distances, a = np.where(Stops_Buurten_distance_matrix <= 0.0005)
        relevant_Stops = np.unique(Short_Distances)
        self.GVBStop_locations = self.GVBStop_locations.iloc[relevant_Stops]


        # Exclude stops with connections to regional trains
        for NS_stop in NS_stops:
            self.GVBStop_locations = self.GVBStop_locations[self.GVBStop_locations.Stop_name != NS_stop]


        # shrink distance matrix to relevant Stops and Transpose to be Buurt-focussed
        Stops_Buurten_distance_matrix = Stops_Buurten_distance_matrix.iloc[np.array(self.GVBStop_locations.index)]
        Stops_Buurten_distance_matrix = np.array(Stops_Buurten_distance_matrix).T

        # run Buurt-Stop-Assignment scheme
        for i, Buurt_Stop_Distances in enumerate(Stops_Buurten_distance_matrix):
            # check for stops in proximity and within the Buurt
            Stops = np.where(Buurt_Stop_Distances <= 0.0005)[0].tolist()
            # else check for closest Stop and except all Stops up to 10% further than this
            if len(Stops) == 0:
                mindist = np.min(Buurt_Stop_Distances[np.nonzero(Buurt_Stop_Distances)])
                extra_mile = mindist * 1.10
                Stops = np.where(Buurt_Stop_Distances <= extra_mile)[0].tolist()
            # marking assigned stops
            Buurt_Stop_Distances[Stops] = True
            Stops_Buurten_distance_matrix[i] = Buurt_Stop_Distances

        # bool all non assigned stops
        Stops_Buurten_distance_matrix[np.where(Stops_Buurten_distance_matrix != True)] = False
        Stops_Buurten_distance_matrix = pd.DataFrame(Stops_Buurten_distance_matrix,
                                                     index=BuurtBBGA.BBGA_Buurt_data.index,
                                                     columns=self.GVBStop_locations.Stop_name)

        Stops_Buurten_distance_matrix.to_csv(path_or_buf=self.path_BuurtStopsAss, sep=';')
        self.GVBStop_locations.to_csv(path_or_buf=self.path_relStop_locations, sep=';')


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
                    Buurt_flow_matrix[combination[0], combination[1]] += (flow/len(trip_combinations))

        Buurt_flow_matrix = pd.DataFrame(Buurt_flow_matrix,
                                         index=self.Buurten_Stops_Association.index,
                                         columns=self.Buurten_Stops_Association.index)
        Buurt_flow_matrix.to_csv(path_or_buf=self.path_Buurtflows, sep=';')

    def checkOneWays(self):
        self.loadPassCount()

        GVBStops = np.unique(np.array(self.PassCount_data['Halte_(vertrek)']))
        toolow = np.where(np.array(self.PassCount_data['Totaal_reizen']) == '<50')[0]
        self.PassCount_data.loc[toolow, 'Totaal_reizen'] = 'nan'
        self.PassCount_data = self.PassCount_data[self.PassCount_data.Totaal_reizen != 'nan']
        """
        plt.xlabel('Passengers on trips', size=20)
        plt.ylabel('Count', size=20)
        plt.title('Histogram of Passenger Count', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.hist(x=np.array(self.PassCount_data['Totaal_reizen']),bins=80)
        plt.xscale('log')
        plt.tight_layout()
        plt.show()
"""
        self.PassCount_data['Totaal_reizen'] = self.PassCount_data['Totaal_reizen'].astype(int)
        sum_pass = sum(list(self.PassCount_data['Totaal_reizen']))
        av =  sum_pass / len(list(self.PassCount_data['Totaal_reizen']))
        self.PassCount_data = self.PassCount_data.set_index('Halte_(vertrek)')
        self.PassCount_data = self.PassCount_data.sort_index()
        self.PassCount_data['no'] = np.arange(len(self.PassCount_data))
        self.PassCount_data['one_way'] = np.nan

        for origin, row in self.PassCount_data.iterrows():
            destination = row['Halte_(aankomst)']
            there = row['Totaal_reizen']
            try:
                new_df = self.PassCount_data.loc[destination]
                back = int(new_df.loc[new_df['Halte_(aankomst)'] == origin]['Totaal_reizen'])
                self.PassCount_data.iat[row['no'], 3] = there - back
            except:
                self.PassCount_data.iat[row['no'], 3] = there


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
        self.path_PC6Info = os.path.join(self.ROOT_DIR, RawData, self.PC6Info)
        self.path_BuurtBBGA = os.path.join(self.ROOT_DIR, GeneratedData, BBGA_Buurten)
        self.path_BuurtPC6 = os.path.join(self.ROOT_DIR, GeneratedData, self.Buurten_PC6)
        self.path_OTPtimes = os.path.join(self.ROOT_DIR, RawData, self.OTP_times)
        self.path_OTPtimes_old = os.path.join(self.ROOT_DIR, RawData, self.OTP_times_old)
        self.path_BuurtPTtimes = os.path.join(self.ROOT_DIR, GeneratedData, self.Buurten_PT_times)
        self.path_Buurtbiketimes = os.path.join(self.ROOT_DIR, GeneratedData, self.Buurten_bike_times)
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
        Buurten_times = pickle.load(open(os.path.join(self.ROOT_DIR, GeneratedData, 'Buurten_noParks.p'), "rb"))
        BuurtBBGA = set(self.BBGA_Buurt_data.index)
        BuurtDiff = [i for i, item in enumerate(Buurten_times) if item not in BuurtBBGA]

        minadvice_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, GeneratedData,'minadvice_PT.p'), "rb"))
        bike_times = pickle.load(open(os.path.join(self.ROOT_DIR, GeneratedData, 'bike_time_in_seconds.p'), "rb"))

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
        matrix_bike.to_csv(path_or_buf=self.path_Buurtbiketimes, sep=';')
        matrix_9292.to_csv(path_or_buf=self.path_BuurtPTtimes, sep=';')



    def compare9292(self):
        self.loadOTPtimes()

        Buurts_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, GeneratedData, 'Buurten_noParks.p'), "rb"))
        BuurtBBGA = set(self.BBGA_Buurt_data.index)  # convert to set for fast lookups
        BuurtDiff = [i for i, item in enumerate(Buurts_9292) if item not in BuurtBBGA]

        minadvice_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, GeneratedData,'minadvice_PT.p'), "rb"))

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

        self.BBGA_Buurt_data.to_csv(path_or_buf=os.path.join(self.ROOT_DIR, GeneratedData, 'clust_comp.csv'),
                                    index=True, index_label='Buurt_code', sep=';')

        clust_comp = pd.read_csv(filepath_or_buffer=os.path.join(self.ROOT_DIR, GeneratedData, 'clust_comp.csv'),
                                 sep=';')
        temp = np.array(clust_comp['clust_9292']).argsort()
        ranks_9292 = np.empty_like(temp)
        ranks_9292[temp] = np.arange(len(np.array(clust_comp['clust_9292'])))
        temp = np.array(clust_comp['clust_OTP']).argsort()
        ranks_OTP = np.empty_like(temp)
        ranks_OTP[temp] = np.arange(len(np.array(clust_comp['clust_OTP'])))
        clust_comp_kend = kendalltau(ranks_9292, ranks_OTP)

