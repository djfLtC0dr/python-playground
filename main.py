# This is a Workout Of the Day (WOD) app, based on user input, it either pulls data from 
# a website that requires a user to click on an href tag to load content into the website 
# or loads data from a local PDF File.
from wod import Cycle, Deuce, Gpp, Sslp, Tsac
from web_data import WebData
from pdf_data import PdfData
import traceback
import pandas as pd
from datetime import date

# Start of cycle
start_dt = date(2021,11,29)
# End of cycle
end_dt = date(2021,12,31)
#end_dt = date(2022,2,25)
# store input keys
gpp = Gpp(0)
cycle = Cycle(start_dt, end_dt)

try:
  gpp = Gpp(int(input("Enter the number corresponding to the type of athlete that best describes you:\n1) Functional Fitness;\n2) Garage Gym Warrior;\n3) Tactical;\n4) Novice \n")))
except ValueError as e:
  print(str(e))

try:
  # Obtain type of workout for follow-on processing
  if (gpp.TYPES[gpp.type] == 'GARAGE') or (gpp.TYPES[gpp.type] == 'ATHLETICS'):
    deuce = Deuce(cycle.start_dt, cycle.end_dt)
    wd = WebData(WebData.DEUCE, gpp.TYPES[gpp.type])
    wd.webscrape(deuce.workout_dates)
  elif gpp.TYPES[gpp.type]  == 'TSAC':
    tsac = Tsac(cycle.start_dt, cycle.end_dt)
    pdf_data = PdfData(PdfData.PDF_SC_LV_PAGES, tsac.workout_dates)
    print(pdf_data.load_pdf_to_json())
  elif gpp.TYPES[gpp.type]  == 'SSLP':
    sslp = Sslp(cycle.start_dt, cycle.end_dt)
    print(sslp.load_data_sslp_ph1())
  else:
    print(gpp.TYPES[gpp.type])
    pass # NOPERATOR do nothing
except: 
  traceback.print_exc()
