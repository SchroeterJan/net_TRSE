import requests as rq
import json

class OTP_grabber:
    # Constants

    OTP_SERVER_URL = "http://localhost:8080/"
    META = "otp/routers/"
    PLAN = "default/plan"

    def __init__(self):
        print("Connecting to " + self.__class__.__name__)
        router_ID = json.loads(rq.get(self.OTP_SERVER_URL + self.META).text)['routerInfo'][0]['routerId']
        print('Router ID: ' + router_ID)

    def planner(self):
        par = {'arriveBy': False,
               'bikeBoardCost': 0,
               'bikeSwitchCost': 0,
               'bikeSwitchTime': 0,
               'clampInitialWait': 0,
               'date': '2020-12-08',
               'fromPlace': '52.37908, 4.89956',
               'maxWalkDistance': 200,
               'minTransferTime': 180,
               'mode': 'WALK,TRAM,SUBWAY,RAIL,BUS,FERRY',
               'numItineraries': 10,
               'optimize': 'QUICK',
               'pathComparator': 'duration',
               'showIntermediateStops': True,
               'time': '09:00:00',
               'toPlace': '52.34680, 4.91833',
               'wheelchair': False,
               }
        plan = json.loads(rq.get(self.OTP_SERVER_URL + self.META + self.PLAN, params=par).text)['plan']

        a = 10




grabber = OTP_grabber().planner()

a = 10