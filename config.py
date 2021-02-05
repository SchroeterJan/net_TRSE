import os





# declare paths to raw data
dir_data = r'C:\Users\jan.schroeter\Documents\Uni\Thesis\Data\Raw_Data'
# Find repo direction
path_repo = os.path.dirname(os.path.abspath(__file__))

# raw data Files
file_passcount = 'HBReizen_Schroter.csv'
file_stops = 'Stop_Locations.csv'
file_se = 'bbga_latest_and_greatest.csv'
file_geo = 'GEBIED_BUURTEN.csv'

# path for generated data
path_generated = 'Generated_data'
file_neighborhood_se = 'neighborhood_se.csv'
path_neighborhood_se = os.path.join(path_repo, path_generated, file_neighborhood_se)



### CONSTANTS
# relevant column names
column_names = {'geo_id_col': 'Buurt_code',     # column holding area identifier
                'pop_col': 'BEVTOTAAL',         # population
                'size_col': 'Opp_m2',           # area size - please provide in m2 else reconfigure 'filter_area'
                'geo_id_se': 'gebiedcode15',    # column in socio-economic data holding area identifier
                'year_col': 'jaar',             # column holding the corresponding year of a se variable
                'se_var_col': 'variabele',      # column holding the se variable identifiers
                'se_col': 'waarde',             # column holding the value of an se variable
                'pass_or': 'Halte_(vertrek)',
                'pass_dest': 'Halte_(aankomst)'
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