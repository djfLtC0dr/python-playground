# This is a Workout Of the Day (WOD) app, based on user input, it either pulls data from 
# a website that requires a user to click on an href tag to load content into the website 
# or loads data from a local PDF File or loads web/CSV data
from wod import Cycle, Deuce, Gpp, Sslp, Tsac
import traceback
from datetime import date

try:
  gpp = Gpp(int(input("Enter the number corresponding to the type of athlete that best describes you:\n1) Functional Fitness;\n2) Garage Gym Warrior;\n3) Tactical;\n4) Novice \n")))
except ValueError as e:
  print(str(e))

try:
  # Cycle dates #end_dt = date(2022,2,25) 
  start_dt, end_dt = date(2021,11,29), date(2021,12,31)
  cycle = Cycle(start_dt, end_dt)
  # Obtain type of workout for follow-on processing
  if (gpp.TYPES[gpp.type] == 'GARAGE') or (gpp.TYPES[gpp.type] == 'ATHLETICS'):
    deuce = Deuce(gpp.TYPES[gpp.type], cycle.start_dt, cycle.end_dt)
    print(deuce.cycle_wods_json)
  elif gpp.TYPES[gpp.type]  == 'TSAC':
    tsac = Tsac(cycle.start_dt, cycle.end_dt)
    print(tsac.cylce_wods_json)
  elif gpp.TYPES[gpp.type]  == 'SSLP':
    sslp = Sslp(start_dt=cycle.start_dt, end_dt =cycle.end_dt)
    print(sslp.cycle_wods_json)
  else:
    print("That's a 'No-Rep!' Choose Option 4--Novice--to get a Novice Linear Progression Phase 1 Breakdown based on StartingStrength.com")
    pass # NOPERATOR do nothing
except: 
  traceback.print_exc()
