from Processing_class import *
from otp_approach import *


###RESEARCH CONSTANTS
year = 2017
variables_of_interest = ['BEVOPLLAAG_P', 'BEVOPLMID_P', 'BEVOPLHOOG_P',
                         'BEVTOTAAL', 'SKSES_GEM', 'IHHINK_GEM', 'PREGWERKL_P']


###EXECUTE BBGA PREPARATION
#BBGAPrep(vars=variables_of_interest, year=year)


#Prep = transportPrep()
#Prep.selectPC6()


###GENERATE OTP PT DATA
otp_grabber = OTP_grabber()
otp_grabber.planner()


###CHECK 9292 DATA

#Prep.compare9292()

#Prep = PassengerCountPrep()
#Prep.checkOneWays()



