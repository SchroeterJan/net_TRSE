from Processing_class import *
import requests as rq
import json
import subprocess

class OTP_grabber:
    # Constants
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    Buurten_PC6 = 'Buurten_PC6.csv'
    OTP_times = 'OTP_times_old.csv'

    OTP_SERVER_URL = "http://localhost:8080/"
    META = "otp/routers/"
    PLAN = "default/plan"


    def __init__(self):
        print("Connecting to " + self.__class__.__name__)
        self.path_BuurtPC6 = os.path.join(self.ROOT_DIR, GeneratedData, self.Buurten_PC6)
        self.path_OTPtimes = os.path.join(self.ROOT_DIR, RawData, self.OTP_times)
        router_ID = json.loads(rq.get(self.OTP_SERVER_URL + self.META).text)['routerInfo'][0]['routerId']
        print('Router ID: ' + router_ID)
        self.loadBuurtPC6()


    def loadBuurtPC6(self):
        self.BuurtPC6 = np.array(pd.read_csv(filepath_or_buffer=self.path_BuurtPC6, sep=';'))



    def planner(self):
        if os.path.isfile(self.path_OTPtimes):
            print('removing existing BBGA Buurten')
            os.remove(self.path_OTPtimes)
        with open(file=self.path_OTPtimes, mode='w') as OTPtimes:
            OTPtimes.write('ORIGING_LAT,ORIGIN_LNG,DESTINATION_LAT,DESTINATION_LNG,DURATION,WALK_TIME,WALK_DIST,TRANSIT_TIME,TRANSFERS\n')
            for i, or_row in enumerate(self.BuurtPC6):
                for j, dest_row in enumerate(self.BuurtPC6[i+1:]):
                    par = {'arriveBy': False,
                           'bikeBoardCost': 0,
                           'bikeSwitchCost': 0,
                           'bikeSwitchTime': 0,
                           'clampInitialWait': -1,
                           'date': '2020-12-08',
                           'fromPlace': str(or_row[1]) + ',' + str(or_row[2]),
                           'maxWalkDistance': 200,
                           'minTransferTime': 180,
                           'mode': 'WALK,TRAM,SUBWAY,RAIL,BUS,FERRY, TRANSIT',
                           'numItineraries': 10,
                           'optimize': 'QUICK',
                           'pathComparator': 'duration',
                           'showIntermediateStops': True,
                           'time': '09:00:00',
                           'toPlace': str(dest_row[1]) + ',' + str(dest_row[2]),
                           'wheelchair': False,
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
                        OTPtimes.write('None,None,None,None,None\n')
                print('Finished row ' + str(i))

