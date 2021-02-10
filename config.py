import os
import datetime
import itertools

import pandas as pd
import numpy as np

from haversine import haversine, Unit
import geopandas
from shapely.geometry import Point
import shapely.wkt

import requests as rq
import json
from bs4 import BeautifulSoup


import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx
from scipy.stats import kendalltau

import pickle



# declare paths to raw data
dir_data = r'C:\Users\jan.schroeter\Documents\Uni\Thesis\Data\Raw_Data'
# Find repo direction
path_repo = os.path.dirname(os.path.abspath(__file__))

# raw data Files
file_passcount = 'HBReizen_Schroter.csv'
file_stops = 'stop_locations.csv'
file_se = 'bbga_latest_and_greatest.csv'
file_geo = 'GEBIED_BUURTEN.csv'
file_biketimes = 'GH_bike_times.csv'
file_PTtimes = 'GH_pt_times.csv'

# path for generated data
path_generated = 'Generated_data'
file_neighborhood_se = 'neighborhood_se.csv'
file_flows = 'neighborhood_flows.csv'
path_neighborhood_se = os.path.join(path_repo, path_generated, file_neighborhood_se)
path_flows = os.path.join(path_repo, path_generated, file_flows)
path_bike = os.path.join(dir_data, file_biketimes)
path_PT = os.path.join(dir_data, file_PTtimes)



### CONSTANTS
# relevant column names
column_names = {'geo_id_col': 'Buurt_code',     # column holding area identifier
                'area_polygon': 'WKT_LNG_LAT',  # column holding area polygon wkt
                'pop_col': 'BEVTOTAAL',         # population
                'size_col': 'Opp_m2',           # area size - please provide in m2 else reconfigure 'filter_area'
                'geo_id_se': 'gebiedcode15',    # column in socio-economic data holding area identifier
                'year_col': 'jaar',             # column holding the corresponding year of a se variable
                'se_var_col': 'variabele',      # column holding the se variable identifiers
                'se_col': 'waarde',             # column holding the value of an se variable
                'stop_name': 'Stop_name',
                'pass_or': 'Halte_(vertrek)',
                'pass_dest': 'Halte_(aankomst)',
                'pass_vol': 'Totaal_reizen',
                'stop_lat': 'stop_lat',
                'stop_lng': 'stop_lng'
                }

# Declare actual Train Stations allowing to switch to regional transport (source:
# https://en.wikipedia.org/wiki/List_of_railway_stations_in_Amsterdam)
exclude_stops = {'Centraal Station',
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

# Census data variables
census_variables = ['BEVOPLLAAG_P', 'BEVOPLMID_P', 'BEVOPLHOOG_P',
                    'BEVTOTAAL', 'SKSES_GEM', 'IHHINK_GEM', 'PREGWERKL_P']

# limit for Population density
min_popdens = 100.0
# year of interest for Census data
se_year = 2017
# proximity definition in coordinate distance
proximity = 0.0005
range_factor = 0.1