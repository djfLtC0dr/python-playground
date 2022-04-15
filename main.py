# This is a Workout Of the Day (WOD) app, based on user input, it either pulls data from 
# a website that requires a user to click on an href tag to load content into the website 
# or loads data from a local PDF File.
import typing
from datetime import timedelta, date
import ssl
import pdfplumber
import pandas as pd
#from date_util import date_util as dtutil

# selenium 
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options() 
chrome_options.add_argument('--no-sandbox') 
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("--headless")

# List obj to store weekday-only dates as strings
workout_dates = []
# Start of cycle
start_dt = date(2021,11,29)
# End of cycle
end_dt = date(2021,12,31)
#end_dt = date(2022,2,25)
# store input key
gpp = 0
# Constant URL + File Strings
DEUCE = "http://www.deucegym.com/community/" # Webscrape
SSLP = "https://startingstrength.com/get-started/programs" # Webscrape
MASH = "mash-evolution.pdf" # PDF
# Pages of the mash-evolution TSAC macrocycle 
pdf_sc_lv_pages = [*range(379, 420, 1)]
# variable to store the loaded SSLP cycle
dfSSLP = pd.DataFrame()
# csv to store the SSLP Macro Cycle
sslp_csv = "sslp.csv"
# date format
dt_str_format = "%Y-%m-%d"

GPP_type = {
  0 : 'NOPERATOR',
  1 : 'ATHLETICS',
  2 : 'GARAGE',
  3 : 'TSAC',
  4 : 'SSLP'
}

valid_gpp_type = False
# User input options
valid_inputs = [0,1,2,3,4]
# Narrow scope to scrape based on athlete type
while not valid_gpp_type:
  try:
    gpp = int(input("Enter the number corresponding to the type of athlete that best describes you:\n1) Functional Fitness;\n2) Garage Gym Warrior;\n3) Tactical;\n4) Novice \n"))
    if gpp in valid_inputs:
      valid_gpp_type = True
    else:
      raise ValueError 
  except ValueError:
    strValueError = "Invalid number entry, please enter a number 1, 2, 3, or 4"
    print(strValueError)

# Iterator/Generator to return += date + days inclusive from start to end date of cycle
def daterange(date1: date, date2: date):
    for n in range(int ((date2 - date1).days) + 1): # + 1 because range is exclusive
        yield date1 + timedelta(n)

# Generate array of workout dates simulating a 30-day trial
for dt in daterange(start_dt, end_dt):
    # deucegym.com only posts workouts on weekdays so we need to exclude weekends
    # integer values corresponding to ISO weekday Sat & Sun.
    weekend = [6,7]
    if dt.isoweekday() not in weekend: 
        # workout_dates is formatted to correspond to the deucegym.com URL pattern
        workout_dates.append(dt.strftime(dt_str_format))
#print(workout_dates)

# Replace \n and * in code with empty string
def replace_chars(s: str) -> str:
  s.replace('\n', '').replace('*', '') 
  return s

# Pulls data from websites using selenium--SSLP is static tables, DEUCE is dynamic
def webscrape(url: str, wod_type = 'NLP') -> None:
  driver = webdriver.Chrome(options=chrome_options) 
  if wod_type == 'NLP': # NOVICE Starting Strength
    driver.get(url)
    try:
      nlp_elements = driver.find_elements(By.CLASS_NAME, "proggy")
      # df.append deprecated so using tmp list of dataframes to append then concat 
      tmp = []      
      for nlp_phase in nlp_elements:
        nlp_phase_innerHTML = replace_chars(nlp_phase.get_attribute('innerHTML'))
        tmp.append(pd.read_html("<table>" + nlp_phase_innerHTML + "</table>")[0])
      nlp_df = pd.concat(tmp, ignore_index=True)
      nlp_df.to_csv('sslp.csv', encoding='utf-8', index=False)
      #print(calc_sslp_ph1())
    finally:
      driver.quit()      
  else: # DEUCE GPP Athlete or Garage Gym Warrior
    for wod_date in workout_dates:
      driver.get(url+wod_date)
      try:
        wod_link = driver.find_element_by_partial_link_text(url)
        ActionChains(driver).move_to_element(wod_link).click(wod_link).perform()
        wod_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "wod_block"))
        )
        wod_innerHTML = wod_element.get_attribute('innerHTML')
        df_wod = pd.read_html("<table>" + wod_innerHTML + "</table>")
        print(df_wod)
      finally:
        driver.quit()    

# TODO This still needs work
def load_pdf(pp: list) -> pd.DataFrame:
  # Load the PDF
  pdf = pdfplumber.open(MASH)
  # Save data to a pandas dataframe.
  p0 = pdf.pages[pp[0]]
  # returns a list of lists, with each inner list representing a row in the table. 
  list_wods = p0.extract_tables()
  
  # Recursion to clean-up data
  def recursively_apply(l, f):
    for n, i in enumerate(l):
        if isinstance(i, list): # check if i is type list
            l[n] = recursively_apply(l[n], f)
        elif isinstance(i, str): # check if i is type str
            l[n] = f(i)
        elif i is None:
            l[n] = '' # nothing to replace there can be only one instance of None
    return l
  
  # Iterate the lists and clean-up/parse strings
  list_wods = recursively_apply(list_wods, replace_chars)
  # Get the data into a dataframe 
  lst_dfs = []
  for l_wod in list_wods:
    for wod in l_wod:
      # dictionary storing the data
      wod_data = {
          'ROE': wod[0],
          'RPE': wod[1]
      }
      #create DataFrame by passing dictionary wrapped in a list
      odf = pd.DataFrame.from_dict([wod_data])
      lst_dfs.append(odf)
  df_wods = pd.concat(lst_dfs, ignore_index=True)
  #print(df_wods)
  return df_wods

def load_csv(file_path: str):
  boolSSLP = False
  try:
    dfSSLP = pd.read_csv(file_path)
    return dfSSLP
  except FileNotFoundError:
    return boolSSLP

# % reference => https://www.t-nation.com/training/know-your-ratios-destroy-weaknesses/
# Bench Press: 75% of back squat
# Powerlifting Deadlift: 120% of back squat 
# Military Press (strict): 45% of back squat
# Power Clean: 68% of back squat
def calc_sslp_ph1():
  one_rm_bs = int(input("What is your 1RM Back Squat?\n"))
  one_rm_bp = (one_rm_bs * .75)
  one_rm_sp = (one_rm_bs * .45)
  one_rm_dl = (one_rm_bs * 1.20)
  bs_pct_inc = 2.5
  bp_pct_inc = 2.0
  sp_pct_inc = 1.5
  dl_pct_inc = 3.5
  pct_1rm = .80
  ph1_rx_bs_loading = []
  ph1_rx_sp_bp_loading = []
  ph1_rx_dl_loading = []
  for i in range(1, len(workout_dates), 5):
    # starting at 80% to allow reasonable linear progression
    bs = str((one_rm_bs * pct_1rm) + (i * bs_pct_inc))
    bp = str((one_rm_bp * pct_1rm) + (i * bp_pct_inc))
    sp = str((one_rm_sp * pct_1rm) + (i * sp_pct_inc))
    dl = str((one_rm_dl * pct_1rm) + (i * dl_pct_inc))
    ph1_rx_bs_loading.append(bs)
    ph1_rx_sp_bp_loading.append(sp + '/' + bp)
    ph1_rx_dl_loading.append(dl)
  return [ph1_rx_bs_loading, ph1_rx_sp_bp_loading, ph1_rx_dl_loading, 'Phase 2-TBD', 'Phase 2-TBD', 'Phase 2-TBD', 'Phase 3-TBD', 'Phase 3-TBD', 'Phase 3-TBD']

def load_data_sslp_ph1() -> pd.DataFrame:
  #python 3.8 "Walrus Operator" to ensure this only runs max 2x
  dfSSLP = load_csv(sslp_csv)
  while (n := len(dfSSLP.index)) != 0:
    dfSSLP = dfSSLP.assign(Phase_1_RX_Loads=calc_sslp_ph1())
    return dfSSLP
  else:
      webscrape(SSLP)
      load_data_sslp_ph1()

# Obtain type of workout for follow-on processing
if GPP_type[gpp] == 'ATHLETICS':
  webscrape(DEUCE, 'ATHLETICS')
elif GPP_type[gpp] == 'GARAGE':
  webscrape(DEUCE, 'GARAGE')
elif GPP_type[gpp] == 'TSAC':
  load_pdf(pdf_sc_lv_pages)
elif GPP_type[gpp] == 'SSLP':
  print(load_data_sslp_ph1())
    # TODO instantiate new SSLP class with calc methods
else:
  print(GPP_type[gpp])
  pass # NOPERATOR do nothing

