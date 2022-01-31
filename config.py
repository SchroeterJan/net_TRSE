import os

import yaml

with open('config_file.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    vars = yaml.load(file, Loader=yaml.FullLoader)


# declare paths to raw data
dir_raw = vars['dir_raw']

# raw data Files
file_geo = vars['file_geo']
# declare coordinate reference system of the polygon data, if already in epsg:4326 leave None
crs_proj = vars['crs_proj']

# socio-economic data set
file_se = vars['file_se']

# locations for travel times (defaults to area centers if None given)
file_locations = vars['file_locations']

file_passcount = vars['file_passcount']
file_stops = vars['file_stops']


### CONSTANTS
# relevant column names
column_names = vars['column_names']

# Declare actual Train Stations allowing to switch to regional transport (source:
# https://en.wikipedia.org/wiki/List_of_railway_stations_in_Amsterdam)
exclude_stops = vars['exclude_stops']

# identifiers for relevant census variables
census_variables = vars['census_variables']

scaling_variables = vars['scaling_variables']

model_variables = vars['model_variables']

# limit for Population density in people/km^2
min_popdens = vars['min_popdens']
# year of interest for Census data
se_year = vars['se_year']
# proximity definition in coordinate distance
proximity = vars['proximity']
range_factor = vars['range_factor']
# maximum definition of a short trip in meters
short_trip = vars['short_trip']


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
path_bike_scrape = os.path.join(dir_raw, file_biketimes_raw)
path_otp_scrape = os.path.join(dir_raw, file_otp_times)
path_stops = os.path.join(dir_raw, file_stops)
path_bike_matrix = os.path.join(path_repo, path_generated, file_bike_matrix)
path_otp_matrix = os.path.join(path_repo, path_generated, file_otp_matrix)
path_pt_matrix = os.path.join(path_repo, path_generated, file_pt_times)
path_euclid_matrix = os.path.join(path_repo, path_generated, file_euclid)


path_experiments = os.path.join(path_repo, 'experiment_data')
path_clustercoeff = os.path.join(path_experiments, 'neighborhood_clustercoeff.csv')

if not os.path.isdir(path_generated):
    os.mkdir(path_generated)
path_plotting = os.path.join(path_repo, 'plotting')
if not os.path.isdir(path_plotting):
    os.mkdir(path_plotting)
path_plot = os.path.join(path_plotting, 'plots')
if not os.path.isdir(path_plot):
    os.mkdir(path_plot)
path_hists = os.path.join(path_plot, 'hists')
if not os.path.isdir(path_hists):
    os.mkdir(path_hists)
path_hist_se = os.path.join(path_hists, 'se')
if not os.path.isdir(path_hist_se):
    os.mkdir(path_hist_se)
path_explore = os.path.join(path_plot, 'explore')
if not os.path.isdir(path_explore):
    os.mkdir(path_explore)
path_maps = os.path.join(path_plot, 'maps')
if not os.path.isdir(path_maps):
    os.mkdir(path_maps)
path_q = os.path.join(path_hists, 'q_acc')
if not os.path.isdir(path_q):
    os.mkdir(path_q)


# declare new path for experiment data
# if not os.path.isdir(path_experiments):
#     os.mkdir(path_experiments)
