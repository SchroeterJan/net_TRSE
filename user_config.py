import yaml


with open('config_file.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    vars = yaml.load(file, Loader=yaml.FullLoader)


# declare paths to raw data
dir_data = vars['dir_data']

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