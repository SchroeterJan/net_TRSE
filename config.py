from user_config import *

import os
import datetime
import itertools

import pandas as pd
import numpy as np

import geopandas

from shapely.geometry import Point
from shapely.geometry import Polygon
import shapely.wkt
from pyproj import Transformer


import requests as rq
import json
from bs4 import BeautifulSoup

import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx
from scipy.stats import kendalltau

import pickle


# Find repo direction
path_repo = os.path.dirname(os.path.abspath(__file__))

# path for generated data
path_generated = 'Generated_data'
file_neighborhood_se = 'neighborhood_se.csv'
file_flows = 'neighborhood_flows.csv'
file_biketimes_raw = 'GH_bike.csv'
file_pt_times = 'PT_times.csv'

path_neighborhood_se = os.path.join(path_repo, path_generated, file_neighborhood_se)
path_flows = os.path.join(path_repo, path_generated, file_flows)
path_bike_scrape = os.path.join(dir_data, file_biketimes_raw)
path_stops = os.path.join(dir_data, file_stops)
