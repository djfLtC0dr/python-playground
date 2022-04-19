# This is a Workout Of the Day (WOD) app, based on user input, it either pulls data from 
# a website that requires a user to click on an href tag to load content into the website 
# or loads data from a local PDF File.
from wod import Cycle, Gpp, Sslp, Tsac
from web_data import WebData
from pdf_data import PdfData
import typing
import ssl

import pandas as pd
from datetime import date
#from date_util import date_util as dtutil

# List obj to store weekday-only dates as strings
workout_dates = []
# Start of cycle
start_dt = date(2021,11,29)
# End of cycle
end_dt = date(2021,12,31)
#end_dt = date(2022,2,25)
# store input keys
gpp = Gpp(0)
cycle = Cycle(start_dt, end_dt)
# variable to store the loaded SSLP cycle
dfSSLP = pd.DataFrame()


valid_gpp_type = False
# User input options
valid_inputs = [0,1,2,3,4]
# Narrow scope to scrape based on athlete type
while not valid_gpp_type:
  try:
    gpp = Gpp(int(input("Enter the number corresponding to the type of athlete that best describes you:\n1) Functional Fitness;\n2) Garage Gym Warrior;\n3) Tactical;\n4) Novice \n")))
    if gpp.type in valid_inputs:
      valid_gpp_type = True
  except ValueError as e:
    print(str(e))

# Obtain type of workout for follow-on processing
if (gpp.TYPES[gpp.type] == 'ATHLETICS'):
  wd = WebData(WebData.DEUCE, 'ATHLETICS')
  wd.webscrape()
elif (gpp.TYPES[gpp.type] == 'GARAGE'):
  wd = WebData(WebData.DEUCE, 'GARAGE')
  wd.webscrape()
elif gpp.TYPES[gpp.type]  == 'TSAC':
  pdf_data = PdfData(PdfData.PDF_SC_LV_PAGES, cycle.workout_dates)
  pdf_data.load_pdf()
elif gpp.TYPES[gpp.type]  == 'SSLP':
  sslp = Sslp(cycle.start_dt, cycle.end_dt)
  print(sslp.load_data_sslp_ph1())
else:
  print(gpp.TYPES[gpp.type])
  pass # NOPERATOR do nothing

