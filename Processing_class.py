import os
import pandas as pd
import numpy as np
import pickle

global RawData, GeneratedData
RawData = 'Raw_data'
GeneratedData = 'Generated_data'

class BBGAPrep:
    # Find root direction
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Declare paths to data
    BBGA = 'bbga_latest_and_greatest.csv'
    BuurtenInfo = 'GEBIED_BUURTEN.csv'
    BBGA_Buurten = 'BBGA_Buurt.csv'

    # Initialize class
    def __init__(self, vars, year):
        print("Initializing " + self.__class__.__name__)
        self.path_BBGA = os.path.join(self.ROOT_DIR, RawData, self.BBGA)
        self.path_BuurtBBGA = os.path.join(self.ROOT_DIR, GeneratedData, self.BBGA_Buurten)
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
            self.df_vars.to_csv(path_or_buf=self.path_BuurtBBGA, sep=';')
        else:
            print('ERROR while merging frames')


class transportPrep:
    # Find root direction
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    PC6Info = 'PC6_VLAKKEN_BAG.csv'

    def __init__(self, vars, year):
        print("Initializing " + self.__class__.__name__)
        self.path_PC6Info = os.path.join(self.ROOT_DIR, RawData, self.PC6Info)

    def loadPC6(self):
        self.PC6_data = pd.read_csv(filepath_or_buffer=self.path_PC6Info, sep=';')

