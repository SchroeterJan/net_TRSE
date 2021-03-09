# NetTRSE
Network Science approach to Transport-related social exclusion.
Analyzing Cities as networks of travel-time and -flow.

Mandatory input:

- geographic data about the areas of interest
- socio-economic data
- travel-times between areas (different modes possible)*

optional input:

- passenger flows for public transport

*scripts for scraping travel-times from open source routing engines is provided, please read Routing section


## Getting Started

The setup is explained along an example case using Amsterdam as its subject.
Along this explanation, the user_config.py must be populated accordingly.

Steps:

1. create directory for raw data and populate dir_data variable

### Geographic data

The geographic data must contain two features:

- area identifier
- area shape

The example uses public data provided by the administration of Amsterdam.
(source: https://data.amsterdam.nl/datasets/5L2CSm77FLZGYA/registratie-gebieden/ ; last access 25.02.21)
Neighborhood units referred to as "Buurten" serve as granularity. (relevant file is therefore "Gebieden Buurt" as csv)

In the example areas are given as polygons, a format immediate to *shapely*, which is used for interpretation.

Steps:

1. populate file_geo variable with the data sets name
2. name geo_id_col in column_names
3. name area_polygon in column_names 
4. declare coordinate reference system of your data 
(for help see: https://gis.stackexchange.com/questions/7839/identifying-coordinate-system-of-shapefile-when-unknown)


### Socio-economic data

Obligatory features for the socio-economic data are:

1. area identifier
2. area population

The example works with publicly available census data provided by the administration of amsterdam.
(source: https://data.amsterdam.nl/datasets/G5JpqNbhweXZSw/basisbestand-gebieden-amsterdam-bbga/ ; last access)

This set stores information on several years and numerous area types. 
Filters are therefore applied that may not be necessary for data sets of different format.

Steps:

1. populate file_se variable with the data sets name
2. name matching area identifier column as geo_id_se in column_names
3. name pop_col in column_names
 



    a. Bike (e.g. Graphhopper=https://github.com/graphhopper/graphhopper)
    
    b. Public Transport (e.g. OpenTripPlanner=https://github.com/opentripplanner/OpenTripPlanner)
