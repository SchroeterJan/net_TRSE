from config import *




class OTP_grabber:
    # Constants
    Buurten_PC6 = 'Buurten_PC6.csv'
    OTP_times = 'OTP_times.csv'

    OTP_SERVER_URL = "http://localhost:8080/"
    META = "otp/routers/"
    PLAN = "default/plan"


    def __init__(self):
        print("Connecting to " + self.__class__.__name__)
        self.path_OTPtimes = os.path.join(dir_data, self.OTP_times)
        router_ID = json.loads(rq.get(self.OTP_SERVER_URL + self.META).text)['routerInfo'][0]['routerId']
        print('Router ID: ' + router_ID)
        self.loadBuurtPC6()




    def planner(self):
        if os.path.isfile(self.path_OTPtimes):
            print('removing existing OTP times')
            os.remove(self.path_OTPtimes)
        with open(file=self.path_OTPtimes, mode='w') as OTPtimes:
            OTPtimes.write('ORIGING_LAT,ORIGIN_LNG,DESTINATION_LAT,DESTINATION_LNG,DURATION,WALK_TIME,WALK_DIST,TRANSIT_TIME,TRANSFERS\n')
            for i, or_row in enumerate(BuurtPC6[:, 2:]):
                for j, dest_row in enumerate(BuurtPC6[i+1:, 2:]):
                    par = {'fromPlace': str(or_row[1]) + ',' + str(or_row[2]),
                           'toPlace': str(dest_row[1]) + ',' + str(dest_row[2]),
                           'time': '8:00am',
                           'date': '08-30-2021',
                           'mode': 'WALK,TRAM,SUBWAY,RAIL,BUS,FERRY,TRANSIT',
                           'maxWalkDistance': 1000,
                           'arriveBy': 'false',
                           'bikeBoardCost': 0,
                           'bikeSwitchCost': 0,
                           'bikeSwitchTime': 0,
                           'clampInitialWait': -1,
                           'minTransferTime': 180,
                           'numItineraries': 10,
                           'optimize': 'QUICK',
                           'pathComparator': 'duration',
                           'showIntermediateStops': 'true',
                           'wheelchair': 'false',
                           'debugItineraryFilter': 'true',
                           }
                    advices = json.loads(rq.get(
                        self.OTP_SERVER_URL + self.META + self.PLAN, params=par).text)['plan']['itineraries']
                    try:
                        advice_dur_list = []
                        for each in advices:
                            advice_dur_list.append(each['duration'])
                        best_advice = advices[np.argmin(advice_dur_list)]
                        OTPtimes.write(par['fromPlace']
                                       + ',' + par['toPlace']
                                       + ',' + str(best_advice['duration'])
                                       + ',' + str(best_advice['walkTime'])
                                       + ',' + str(best_advice['walkDistance'])
                                       + ',' + str(best_advice['transitTime'])
                                       + ',' + str(best_advice['transfers'])
                                       + '\n')
                    except:
                        OTPtimes.write('None,None,None,None,None,None,None,None,None\n')
                print('Finished row ' + str(i))



class GH_grabber:
    # Constants
    GH_SERVER_URL = "http://localhost:8989/"
    GH_PT = 'maps/pt/'
    Endpoint_info = 'info/'
    Endpoint_route = "route/"

    def __init__(self):
        print("Connecting to " + self.__class__.__name__)
        print("Graphhopper routing engine version: " + json.loads(rq.get(self.GH_SERVER_URL + self.Endpoint_info).text)['version'])


    # def pt_planner(self, driver):
    #     # print(datetime.datetime.now().isoformat())
    #     par = {'pt.earliest_departure_time': '2020-12-08T08:00:00Z',
    #            'pt.arrive_by': 'false',
    #            'locale': 'en-US',
    #            'profile': 'pt',
    #            'pt.profile': 'false',
    #            'pt.profile_duration': 'PT120M',
    #            'pt.limit_street_time': 'PT30M',
    #            'pt.ignore_transfers': 'false',
    #            'point': ['52.34960205354996,4.893035888671876', '52.3688917060255,4.8903751373291025']
    #            }
    #
    #     url = rq.Request('GET', url=self.GH_SERVER_URL + self.GH_PT, params=par).prepare().url
    #     driver.get(url)
    #     soup = BeautifulSoup(driver.page_source)
    #
    #     for tag in soup.find_all('div'):
    #         try:
    #             tag_class = tag.attrs['class']
    #         except:
    #             tag_class = ''
    #         if tag_class == 'tripDisplay':
    #             trip_tag = tag


    def bike_planner(self, or_, dest_):
        par = {'point': [str(or_[1])+ ',' + str(or_[2]), str(dest_[1]) + ',' +str(dest_[2])],
               'type': 'json',
               'locale': 'de',
               'elevation': 'true',
               'profile': 'bike',
               }
        advice = json.loads(rq.get(self.GH_SERVER_URL + self.Endpoint_route, params=par).text)['paths'][0]
        try:
            # advice_dur_list = []
            # for each in advices:
            #    advice_dur_list.append(each['duration'])
            # best_advice = advices[np.argmin(advice_dur_list)]
            res = par['point'][0] + ',' + \
                  par['point'][1] + ',' + \
                  str(advice['time']) + ',' + \
                  str(advice['distance']) + ',' + \
                  str(advice['weight']) + '\n'
        except:
            res = 'None,None,None,None,None,None,None\n'
        return res




BuurtPC6 = pd.read_csv(filepath_or_buffer=os.path.join(dir_data, file_locations), sep=';').to_numpy()

# otp_grab = OTP_grabber()
# otp_grab.planner()


grabber = GH_grabber()
if os.path.isfile(path_bike_scrape):
    print('removing existing Graphhopper times')
    os.remove(path_bike_scrape)

with open(file=path_bike_scrape, mode='w') as GHtimes:
    GHtimes.write('ORIGING_LAT,ORIGIN_LNG,DESTINATION_LAT,DESTINATION_LNG,DURATION,DISTANCE,WEIGHT\n')
    for i, or_row in enumerate(BuurtPC6[:, 2:]):
        for j, dest_row in enumerate(BuurtPC6[i + 1:, 2:]):
            res = grabber.bike_planner(or_=or_row, dest_=dest_row)
            GHtimes.write(res)

        print('Finished row ' + str(i))