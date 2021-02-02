from Processing_class import *
from config import *
from Routing import *








# Initiate data preparation class
se_prep = SE_Neighborhoods(dir_data=dir_data, columns=column_names, min_popdens=popdens)


def prepare_se():

    # check if prepared data already exists
    if os.path.isfile(path_neighborhood_se):
        print('removing existing Neighborhood SE Data')
        os.remove(path_neighborhood_se)

    # crop socio-economic data to relevant year and areas
    se_prep.crop_se(year=se_year)
    se_prep.geo_data = se_prep.geo_data.set_index(se_prep.geo_id_col, drop=False)

    # keep only relevant socio-economic variables
    for variable in census_variables:
        se_prep.extract_var(var=variable)

    # filter low populated areas
    se_prep.filter_areas(min_popdens=popdens)

    # write resulting data set to disk
    se_prep.geo_data.to_csv(path_or_buf=path_neighborhood_se, index=True, index_label='Buurt_code', sep=';')




prepare_se()



### PREPARE TRANSPORT DATA
#TransPrep = transportPrep()
#TransPrep.selectPC6()
#TransPrep.prepareTransport_times()


### CHECK 9292 DATA
#TransPrep.compare9292()


### COLLECT JOURNEY TIMES TRHOUGH ROUTING
#otp_grabber = OTP_grabber()
#otp_grabber.planner()

#gh_grabber = GH_grabber()
#gh_grabber.planner()


### PREPARE PASSENGER COUNT DATA
#PrepPassC = PassengerCountPrep()
#PrepPassC.assignStopLocations()
#PrepPassC.relevantStops(proximity_measure=proximity, range_factor=range_factor)
#PrepPassC.filterPassCount()
#PrepPassC.assignPassCounts()


