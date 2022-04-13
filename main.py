# This is a Workout Of the Day (WOD) app, based on user input, it either pulls data from 
# a website that requires a user to click on an href tag to load content into the website 
# or loads data from a local PDF File.
from datetime import timedelta, date
import pdfplumber
import pandas as pd

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

# List obj to store weekday-only dates
workout_dates = []
# Start of cycle
start_dt = date(2021,11,29)
# End of cycle
end_dt = date(2022,12,31)
#end_dt = date(2022,2,25)
# store input key
gpp = 0
# Constant URL + File Strings
DEUCE = "http://www.deucegym.com/community/" # Webscrape
SSLP = "https://startingstrength.com/get-started/programs" # Webscrape
MASH = "mash-evolution.pdf" # PDF

pdf_strength_con_low_vol_pages = [*range(379, 420, 1)]

GPP_type = {
  1 : 'ATHLETICS',
  2 : 'GARAGE',
  3 : 'TSAC',
  4 : 'SSLP'
}

valid_gpp_type = False
# User input options
valid_inputs = [1,2,3,4]
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
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days) + 1): # + 1 because range is exclusive
        yield date1 + timedelta(n)

# Generate array of workout dates simulating a 30-day trial
for dt in daterange(start_dt, end_dt):
    # deucegym.com only posts workouts on weekdays so we need to exclude weekends
    # integer values corresponding to ISO weekday Sat & Sun.
    weekend = [6,7]
    if dt.isoweekday() not in weekend: 
        # workout_dates is formatted to correspond to the deucegym.com URL pattern
        workout_dates.append(dt.strftime("%Y-%m-%d"))
#print(workout_dates)

# TODO load data into database code 
def webscrape(url, wod_type = 'NLP'):
  driver = webdriver.Chrome(options=chrome_options) 
  if wod_type == 'NLP':
    driver.get(url)
    try:
      nlp_elements = driver.find_elements(By.CLASS_NAME, "proggy")
      # df.append deprecated so using tmp list of dataframes to append then concat 
      tmp = []      
      for nlp_phase in nlp_elements:
        nlp_phase_innerHTML = nlp_phase.get_attribute('innerHTML')
        tmp.append(pd.read_html("<table>" + nlp_phase_innerHTML + "</table>")[0])
      nlp_df = pd.concat(tmp, ignore_index=True)
      # TODO: load data into monthy cycle using 2.5-5.0lb. increments
      print(nlp_df)
    finally:
      driver.quit()      
  else:
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
def load_pdf(pp):
  # Load the PDF
  pdf = pdfplumber.open(MASH)
  # Save data to a pandas dataframe.
  p0 = pdf.pages[pp[0]]
  # returns a list of lists, with each inner list representing a row in the table. 
  list_wods = p0.extract_tables()

  def replace_chars(s):
    return s.replace('\n', '') #empty string

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
      #cols = ['ROE', 'RPE']
      # dictionary storing the data
      wod_data = {
          'ROE': wod[0],
          'RPE': wod[1]
      }
      #i = dict(day = (start_dt.strftime("%Y-%m-%d")))
      #dfWod = pd.DataFrame.from_dict([w.get('roe'), w.get('rpe')])
      #dfWod.columns = cols
      # df.append deprecated so using list of dict items to append then concat    
      #l = list(w.items())
      #odf = pd.DataFrame(l)
      odf = pd.DataFrame.from_dict([wod_data])
      lst_dfs.append(odf)
      #print(wod)
  df_wods = pd.concat(lst_dfs, ignore_index=True)
# We should always be expecting 3 rows--M-W-F--Lower Volume
  #print(df_wods.loc[0, :])
  print(df_wods)
       
# Obtain type of workout to pull into database
if GPP_type[gpp] == 'ATHLETICS':
  webscrape(DEUCE, 'ATHLETICS')
elif GPP_type[gpp] == 'GARAGE':
  webscrape(DEUCE, 'GARAGE')
elif GPP_type[gpp] == 'TSAC':
  load_pdf(pdf_strength_con_low_vol_pages)
else:
  webscrape(SSLP)    
#print(GPP_type[gpp])
