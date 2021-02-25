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

For the example, public data provided by the administration of amsterdam is used.
(source: https://data.amsterdam.nl/datasets/5L2CSm77FLZGYA/registratie-gebieden/ ; last access 25.02.21)
Neighborhood units referred to as "Buurten" serve as granularity. (relevant file is therefore "Gebieden Buurt" as CSV)

The data set provides a column "code" as area identifier and a column "geometrie" storing the areas shape as a polygon.
Such polygon shape is preferred since shapely is used for interpretation and works immediately on this format.

Steps:

1. 
### Socio-economic data

The socio-economic data must contain one feature:

1. area identifier

The example works with publicly available census data provided by the administration of amsterdam.
(source: https://data.amsterdam.nl/datasets/G5JpqNbhweXZSw/basisbestand-gebieden-amsterdam-bbga/ ; last access)

This set stores information on several years and numerous area types. 
Filters are therefore applied that may not be necessary for data sets of different format.




    a. Bike (e.g. Graphhopper=https://github.com/graphhopper/graphhopper)
    
    b. Public Transport (e.g. OpenTripPlanner=https://github.com/opentripplanner/OpenTripPlanner)
