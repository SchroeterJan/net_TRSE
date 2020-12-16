from Classes import *
import os
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import math
import pickle
from scipy.stats import kendalltau
from matplotlib import pyplot as plt



global RawData, GeneratedData, BBGA_Buurten
RawData = 'Raw_data'
GeneratedData = 'Generated_data'
BBGA_Buurten = 'BBGA_Buurt.csv'

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


class transportPrep:
    # Find root direction
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PC6Info = 'PC6_VLAKKEN_BAG.csv'
    Buurten_PC6 = 'Buurten_PC6.csv'
    OTP_times = 'OTP_times.csv'

    def __init__(self, fair=False):
        print("Initializing " + self.__class__.__name__)
        self.path_PC6Info = os.path.join(self.ROOT_DIR, RawData, self.PC6Info)
        self.path_BuurtBBGA = os.path.join(self.ROOT_DIR, GeneratedData, BBGA_Buurten)
        self.path_BuurtPC6 = os.path.join(self.ROOT_DIR, GeneratedData, self.Buurten_PC6)
        self.path_OTPtimes = os.path.join(self.ROOT_DIR, RawData, self.OTP_times)
        self.fair = fair





    def loadPC6(self):
        self.PC6_data = pd.read_csv(filepath_or_buffer=self.path_PC6Info, sep=';')

    def loadBuurtBBGA(self):
        self.BBGA_Buurt_data = pd.read_csv(filepath_or_buffer=self.path_BuurtBBGA, sep=';', index_col='Buurt_code')

    def loadOTPtimes(self):
        self.OTP_times_data = pd.read_csv(filepath_or_buffer=self.path_OTPtimes, sep=',')

    def selectPC6(self):
        self.loadPC6()
        self.loadBuurtBBGA()
        if os.path.isfile(self.path_BuurtPC6):
            print('removing existing BBGA Buurten')
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



    def compare9292(self):
        self.loadBuurtBBGA()
        self.loadOTPtimes()
        Buurts_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, GeneratedData, 'Buurten_noParks.p'), "rb"))
        BuurtBBGA = set(self.BBGA_Buurt_data.index)  # convert to set for fast lookups
        BuurtDiff = [i for i, item in enumerate(Buurts_9292) if item not in BuurtBBGA]

        minadvice_9292 = pickle.load(open(os.path.join(self.ROOT_DIR, GeneratedData,'minadvice_PT.p'), "rb"))

        OTP_times = np.array(self.OTP_times_data['DURATION'])
        for i, each in enumerate(OTP_times):
            try:
                a = float(each)%60
                if a <= 30.0:
                    OTP_times[i] = float(each)-a
                else:
                    OTP_times[i] = float(each)+(60.0-a)
            except:
                continue


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
        diff_mat = np.array(matrix_OTP-np.array(matrix_9292))
        diff_mean = np.nanmean(diff_mat)
        diff_av = np.nansum(a=np.triu(diff_mat, k=1), axis=1)  ####calc wrong, get triangular matrice instead

        #self.BBGA_Buurt_data['clust_9292']= Analysis().Clustering(matrix=matrix_9292,
        #                                                          Buurten=np.array(self.BBGA_Buurt_data.index))
        #self.BBGA_Buurt_data['clust_OTP'] = Analysis().Clustering(matrix=matrix_OTP,
        #                                                           Buurten=np.array(self.BBGA_Buurt_data.index))

        #self.BBGA_Buurt_data.to_csv(path_or_buf=os.path.join(self.ROOT_DIR, GeneratedData, 'clust_comp.csv'), index=True, index_label='Buurt_code', sep=';')

        clust_comp = pd.read_csv(filepath_or_buffer=os.path.join(self.ROOT_DIR, GeneratedData, 'clust_comp.csv'), sep=';')
        temp = np.array(clust_comp['clust_9292']).argsort()
        ranks_9292 = np.empty_like(temp)
        ranks_9292[temp] = np.arange(len(np.array(clust_comp['clust_9292'])))
        temp = np.array(clust_comp['clust_OTP']).argsort()
        ranks_OTP = np.empty_like(temp)
        ranks_OTP[temp] = np.arange(len(np.array(clust_comp['clust_OTP'])))
        clust_comp_kend = kendalltau(ranks_9292, ranks_OTP)


        a = 10




class PassengerCountPrep:
    # Find root direction
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PassCountfile = 'HBReizen_Schroter.csv'
    AmsStopsfile = 'stops_amsterdam.txt'
    OSM_stops = 'OSM_PT_stations_ams.csv'



    def __init__(self):
        print("Initializing " + self.__class__.__name__)
        self.path_PassCount = os.path.join(self.ROOT_DIR, RawData, self.PassCountfile)
        self.path_AmsStops = os.path.join(self.ROOT_DIR, RawData, self.AmsStopsfile)
        self.path_OSMStops = os.path.join(self.ROOT_DIR, RawData, self.OSM_stops)



    def loadPassCount(self):
        self.PassCount_data = pd.read_csv(filepath_or_buffer=self.path_PassCount, sep=';')

    def loadStopData(self):
        self.AmsStops_data = pd.read_csv(filepath_or_buffer=self.path_AmsStops, sep=',')
        self.OSMStops_data = pd.read_csv(filepath_or_buffer=self.path_OSMStops, sep=';')


    def checkStopLocations(self):
        self.loadPassCount()
        self.loadStopData()
        self.AmsStops_data['stop_name'] = self.AmsStops_data['stop_name'].str.split(', ').str[1]
        AmsStop_names = set(self.AmsStops_data['stop_name'])
        OSMStop_names = self.OSMStops_data['name'].astype(str)
        OSMStop_names2 = self.OSMStops_data['name'].str.split(', ').str[1]
        OSMstop_nan = np.array(OSMStop_names2.isna())

        a = OSMStop_names.at[5]
        for i, each in enumerate(OSMstop_nan):
            if each == False:
                OSMStop_names.at[i] = OSMStop_names2.at[i]

        OSMStop_names = np.unique(OSMStop_names)
        GVBStop_names = np.unique(np.array(self.PassCount_data['Halte_(vertrek)']))
        gtfsgood = []
        gtfsbad = []
        for each in GVBStop_names:
            if each in AmsStop_names:
                gtfsgood.append(each)
            else:
                gtfsbad.append(each)
        OSMgood = []
        OSMbad = []
        for each in GVBStop_names:
            if each in OSMStop_names:
                OSMgood.append(each)
            else:
                OSMbad.append(each)


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

