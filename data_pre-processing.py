from Processing_class import *
from Routing import *


### CONSTANTS
year = 2017
variables_of_interest = ['BEVOPLLAAG_P', 'BEVOPLMID_P', 'BEVOPLHOOG_P',
                         'BEVTOTAAL', 'SKSES_GEM', 'IHHINK_GEM', 'PREGWERKL_P']
popdens = 100.0

proximity = 0.0005
range_factor = 0.1


### EXECUTE BBGA PREPARATION
#BBGAPrep(vars=variables_of_interest, year=year, min_popdens=popdens)


### PREPARE TRANSPORT DATA
TransPrep = transportPrep()
#TransPrep.selectPC6()
TransPrep.prepareTransport_times()


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


