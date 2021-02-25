from processors import *
from config import *


def areas():
    se_prep = SE_Neighborhoods()
    se_prep.area_size()
    # check if prepared data already exists
    # if os.path.isfile(path_neighborhood_se):
    #     print('removing existing Neighborhood SE Data')
    #     os.remove(path_neighborhood_se)

    # crop socio-economic data to relevant year and areas
    se_prep.crop_se(year=se_year)
    se_prep.geo_data = se_prep.geo_data.set_index(se_prep.geo_id_col, drop=False)

    # keep only relevant socio-economic variables
    for variable in census_variables:
        se_prep.extract_var(var=variable)

    # filter low populated areas
    se_prep.filter_areas()

    # write resulting data set to disk
    print('writing preprocessed data to disk')
    se_prep.geo_data.to_csv(path_or_buf=path_neighborhood_se, index=True, index_label='Buurt_code', sep=';')


def flows():
    flow_prep = Passenger_Counts()
    flow_prep.area_stop_matching()
    flow_prep.filter_passcount()
    area_flow_matrix = flow_prep.assign_passcounts()

    print('Writing flow matrix to disk')
    area_flow_matrix.to_csv(path_or_buf=path_flows, sep=';')


def bike_times():
    trans_prep = TransportPrep()
    if not os.path.isfile(path_bike):
        print('Gather bike times\n Make sure GH server is running!')
        trans_prep.get_gh_times()

    bike_times = trans_prep.order_times()
    bike_times.to_csv(path_or_buf=os.path.join(path_repo, path_generated, 'Bike_times_GH.csv'), sep=';')

areas()
# flows()

# bike_times()



