from processors import *
from config import *


def prepare_se():
    se_prep = SE_Neighborhoods()
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
    se_prep.filter_areas()

    # write resulting data set to disk
    print('writing preprocessed data to disk')
    se_prep.geo_data.to_csv(path_or_buf=path_neighborhood_se, index=True, index_label='Buurt_code', sep=';')


def prepare_flow():
    flow_prep = Passenger_Counts()
    flow_prep.area_stop_matching()
    flow_prep.filter_passcount()
    area_flow_matrix = flow_prep.assign_passcounts()
    print('Writing flow matrix to disk')
    area_flow_matrix.to_csv(path_or_buf=path_flows, sep=';')


def get_gh_times(path, profile):
    grabber = GH_grabber()
    if os.path.isfile(path):
        print('removing existing Graphhopper times')
        os.remove(path)
    BuurtPC6 = pd.read_csv(filepath_or_buffer=path_PC6, sep=';')

    if profile == 'PT':
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Firefox(options=options,
                                   executable_path=r'C:\Users\jan.schroeter\BrowserDrivers\geckodriver.exe')
        print("Headless Firefox Initialized")


    with open(file=path, mode='w') as GHtimes:
        GHtimes.write('ORIGING_LAT,ORIGIN_LNG,DESTINATION_LAT,DESTINATION_LNG,DURATION,DISTANCE,WEIGHT\n')
        for i, or_row in enumerate(BuurtPC6):
            for j, dest_row in enumerate(BuurtPC6[i + 1:]):
                if profile == 'bike':
                    res = grabber.bike_planner(or_=or_row, dest_=dest_row)
                elif profile == 'PT':

                    res = grabber.pt_planner(driver=driver)
                else:
                    print('profile does not exist')
                GHtimes.write(res)

            print('Finished row ' + str(i))

    if profile == 'PT':
        driver.close()

# prepare_se()
# prepare_flow()



