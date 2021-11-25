from user_config import *

import os
import datetime
import itertools

import pandas as pd
import numpy as np
import math

import geopandas
import geoplot
import pygeoda

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
from sklearn.cluster import KMeans

import pickle


# Find repo direction
path_repo = os.getcwd()

# path for generated data
path_generated = 'Generated_data'
file_neighborhood_se = 'neighborhood_se.csv'
file_flows = 'neighborhood_flows.csv'
file_biketimes_raw = 'GH_bike.csv'
file_otp_times = 'OTP_times.csv'
file_pt_times = 'PT_times.csv'
file_bike_matrix = 'bike_matrix.csv'
file_otp_matrix = 'otp_matrix.csv'
file_euclid = 'euclid_matrix.csv'


path_neighborhood_se = os.path.join(path_repo, path_generated, file_neighborhood_se)
path_flows = os.path.join(path_repo, path_generated, file_flows)
path_bike_scrape = os.path.join(dir_data, file_biketimes_raw)
path_otp_scrape = os.path.join(dir_data, file_otp_times)
path_stops = os.path.join(dir_data, file_stops)
path_bike_matrix = os.path.join(path_repo, path_generated, file_bike_matrix)
path_otp_matrix = os.path.join(path_repo, path_generated, file_otp_matrix)
path_pt_matrix = os.path.join(path_repo, path_generated, file_pt_times)
path_euclid_matrix = os.path.join(path_repo, path_generated, file_euclid)

path_experiments = os.path.join(path_repo, 'experiment_data')
path_clustercoeff = os.path.join(path_experiments, 'neighborhood_clustercoeff.csv')

path_plotting = os.path.join(path_repo, 'plotting')
if not os.path.isdir(path_plotting):
    os.mkdir(path_plotting)
path_plot = os.path.join(path_plotting, 'plots')
if not os.path.isdir(path_plot):
    os.mkdir(path_plot)
path_hists = os.path.join(path_plot, 'hists')
if not os.path.isdir(path_hists):
    os.mkdir(path_hists)
path_explore = os.path.join(path_plot, 'explore')
if not os.path.isdir(path_explore):
    os.mkdir(path_explore)
path_maps = os.path.join(path_plot, 'maps')
if not os.path.isdir(path_maps):
    os.mkdir(path_maps)
