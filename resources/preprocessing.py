from resources.prep_resources import *


def areas():
    se_prep = SENeighborhoods()

    # crop socio-economic data to relevant year and areas
    se_prep.crop_se(year=se_year)
    se_prep.geo_data = se_prep.geo_data.set_index(keys=column_names['geo_id_col'], drop=False)
    se_prep.filter_areas()

    # keep only relevant socio-economic variables
    for variable in census_variables:
        se_prep.extract_var(var=variable)


    # check if prepared data already exists
    if os.path.isfile(path_neighborhood_se):
        print('removing existing Neighborhood SE Data')
        os.remove(path_neighborhood_se)
    # write resulting data set to disk
    print('writing preprocessed data to disk')
    se_prep.geo_data.to_csv(path_or_buf=path_neighborhood_se, index=True, index_label='Buurt_code', sep=';')


def od_matrices():
    matrix_files = [path_bike_matrix, path_otp_matrix, path_euclid_matrix]
    od_prepare = ODPrep()
    matrices = [build_matrix(length=len(od_prepare.neighborhood_se[column_names['geo_id_col']]),
                             data_list=list(od_prepare.load_data(path_bike_scrape)['DURATION'] / 1000.0)),
                build_matrix(length=len(od_prepare.neighborhood_se[column_names['geo_id_col']]),
                             data_list=list(od_prepare.load_data(path_otp_scrape)['DURATION'])),
                od_prepare.calc_euclid()]

    for i, matrix in enumerate(matrices):
        matrix = od_prepare.matrix_to_frame(matrix)
        matrix.to_csv(path_or_buf=matrix_files[i], sep=';')


def flows():
    flow_prep = PassengerCounts()
    flow_prep.area_stop_matching()
    flow_prep.filter_connections()
    area_flow_matrix = flow_prep.assign_passcounts()

    if os.path.isfile(path_flows):
        print('removing existing flow Data')
        os.remove(path_flows)
    print('Writing flow matrix to disk')
    area_flow_matrix.to_csv(path_or_buf=path_flows, sep=';')


areas()
# flows()
# od_matrices()
