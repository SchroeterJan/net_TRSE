# declare paths to raw data
dir_data = r'C:\Users\jan.schroeter\Documents\Uni\Thesis\Data\Raw_Data'

# raw data Files
file_geo = 'GBD_buurt_Actueel.csv'
# declare coordinate reference system of the polygon data, if already in epsg:4326 leave blank ("")
crs_proj = 'epsg:28992'

# socio-economic data set
file_se = 'bbga_latest_and_greatest.csv'

# locations for travel times (defaults to area centers if "" given)
file_locations = r'deprecated/Buurten_PC6.csv'

file_passcount = 'HBReizen_Schroter.csv'
file_stops = 'stop_locations.csv'


### CONSTANTS
# relevant column names
column_names = {'geo_id_col': 'code',     # column holding area identifier
                'area_polygon': 'geometrie',  # column holding area polygon wkt
                'pop_col': 'BEVTOTAAL',         # population
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

# identifiers for relevant census variables
census_variables = ['BEVOPLLAAG_P', 'BEVOPLMID_P', 'BEVOPLHOOG_P',
                    'BEVTOTAAL', 'SKSES_GEM', 'IHHINK_GEM', 'PREGWERKL_P']

# limit for Population density in people/km^2
min_popdens = 100.0
# year of interest for Census data
se_year = 2017
# proximity definition in coordinate distance
proximity = 0.0005
range_factor = 0.1