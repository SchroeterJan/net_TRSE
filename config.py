import os
import pandas as pd
import numpy as np



# declare paths to raw data
dir_data = ''
# Find repo direction
path_repo = os.path.dirname(os.path.abspath(__file__))

# raw data Files
file_se = ''
file_geo = ''

# path for generated data
path_generated = 'Generated_data'
file_neighborhood_se = 'neighborhood_se.csv'
path_neighborhood_se = os.path.join(path_repo, path_generated, file_neighborhood_se)



### CONSTANTS
# relevant column names
column_names = {'geo_id_col': None,      # column holding area identifier
                'pop_col': None,         # population
                'size_col': None,        # area size - please provide in m2 else reconfigure 'filter_area'
                'geo_id_se': None,       # column in socio-economic data holding area identifier
                'year_col': None,        # column holding the corresponding year of a se variable
                'se_var_col': None,      # column holding the se variable identifiers
                'se_col': None           # column holding the value of an se variable
                }

# Census data variables
census_variables = []

# limit for Population density
popdens = 100.0
# year of interest for Census data
se_year = 2017
# proximity definition in coordinate distance
proximity = 0.0005
range_factor = 0.1