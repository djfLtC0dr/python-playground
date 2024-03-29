# selenium 
from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import json

class WebData:
  def __init__(self, url: str, gpp_type: str, workout_dates: list = [], params: str = ''):
    self.url = url
    self.gpp_type = gpp_type
    self.workout_dates = workout_dates
    self.params = params
    self.wods_json = self.webscrape_data_to_json

  # Replace \n and * in  and \t code with empty string
  @staticmethod
  def replace_chars(s: str) -> str:
      s = s.replace('\n', '').replace('*', '').replace('\t', '')
      return s   

  # Pulls data from websites using selenium--SSLP is static tables, DEUCE is dynamic
  # Printing the static SSLP to csv for parsing
  # DEUCE ultimately needs to be jsonified and saved to db
  def webscrape_data_to_csv(self) -> bool:
    bool_csv_created = False
    driver = webdriver.Chrome(options=self.get_chrome_options())
    driver.get(self.url)
    elem_to_find = "proggy"
    try:
      nlp_elements = driver.find_elements(By.CLASS_NAME, elem_to_find)
      # df.append deprecated so using tmp list of dataframes to append then concat 
      tmp = []      
      for nlp_phase in nlp_elements:
          nlp_phase_innerHTML = self.replace_chars(nlp_phase.get_attribute('innerHTML'))
          tmp.append(pd.read_html("<table>" + nlp_phase_innerHTML + "</table>")[0])
      nlp_df = pd.concat(tmp, ignore_index=True)
      nlp_df.to_csv('sslp.csv', encoding='utf-8', index=False)
      bool_csv_created = True
      #print(calc_sslp_ph1())
    finally:
        driver.quit()   
        return bool_csv_created
    
  def webscrape_data_to_json(self) -> str:
    if self.gpp_type == 'DEUCE':
      return self.webscrape_deuce_data_to_json()
    else: # PushJerk
      return self.webscrape_pj_data_to_json()

  def webscrape_deuce_data_to_json(self) -> str:
    json_formatted_str = ''
    driver = webdriver.Chrome(options=self.get_chrome_options())
    # TODO: get this working for singleton the loop it => for wod_date in workout_dates:
    wod_date = self.workout_dates[0]
    driver.get(self.url+wod_date)
    popup_xpath = '/html/body/div[3]/div/img'
    try:
        popup = driver.find_element_by_xpath(popup_xpath)
        if popup.is_displayed:
          popup.click() # Closes the popup
    except Exception: #NoSuchElementException:
      # no popup
      pass
    else: 
      inner_html = '\n\t\t\t<h3>11/29/21 WOD</h3>\n\t\t\t<h2 style="text-align: center;"><b>DEUCE ATHLETICS GPP</b></h2>\n<p><span style="font-weight: 400;">Complete 4 rounds for quality of:</span></p>\n<p><span style="font-weight: 400;">8 Barbell Strict Press </span><span style="font-weight: 400;">(3x1x)<br>\n</span><span style="font-weight: 400;">8 Single Kettlebell Lateral Lunge</span></p>\n<p><span style="font-weight: 400;">Then, AMRAP 12</span></p>\n<p><span style="font-weight: 400;">1,2,3,…,∞<br>\n</span><span style="font-weight: 400;">Front Squat (135/95)<br>\n</span><span style="font-weight: 400;">DB Renegade Row (40/20)</span></p>\n<p><span style="font-weight: 400;">**Every 2 min, 1 7th Street Corner Run</span></p>\n<h2 style="text-align: center;"><b>DEUCE GARAGE GPP</b></h2>\n<p><span style="font-weight: 400;">5-5-5-5-5<br>\n</span><span style="font-weight: 400;">Pendlay Row</span></p>\n<p><span style="font-weight: 400;">Then, complete 3 rounds for quality of:</span></p>\n<p><span style="font-weight: 400;">10 Single Arm Bent Over row (ea)<br>\n</span><span style="font-weight: 400;">10-12 Parralette Push Ups<br>\n</span><span style="font-weight: 400;">10 Hollow Body Lat Pulls&nbsp;</span></p>\n<p><span style="font-weight: 400;">Then, AMRAP8</span></p>\n<p><span style="font-weight: 400;">6 Chest to Bar Pull Ups<br>\n</span><span style="font-weight: 400;">8 HSPU<br>\n</span><span style="font-weight: 400;">48 Double unders</span></p>\n\t\t'
      inner_html = self.replace_chars(inner_html)
      df_wod = pd.read_html('<table>' + inner_html + '</table>')
      print(df_wod)
      wod_link_xpath = '/html/body/div[1]/main/center/article/div/p/a'
      wod_link = driver.find_element_by_xpath(wod_link_xpath)
      ActionChains(driver).move_to_element(wod_link).click(wod_link).perform()
      wod_element = WebDriverWait(driver, 10).until(
          EC.presence_of_element_located((By.CLASS_NAME, "wod_block"))
      )
      wod_innerHTML = wod_element.get_attribute('innerHTML')
      df_wod = pd.read_html("<table>" + wod_innerHTML + "</table>")
      wod_json_str = df_wod.to_json(orient='records')
      obj_data = json.loads(wod_json_str)
      json_formatted_str += json.dumps(obj_data, indent=4) 
    finally:
      driver.quit()  
      return json_formatted_str    

  def webscrape_pj_data_to_json(self) -> str:
    wod_list = []
    try:
        driver = webdriver.Chrome(chrome_options=self.get_chrome_options())
        driver.get(self.url + self.gpp_type + self.params)
        source_code = driver.page_source
        soup = BeautifulSoup(source_code,'lxml')
        entry_block = soup.find_all('div', class_='entry-title')
        for entries in entry_block:
          pj_entry_url = entries.find('a')
          wod_list.append(pj_entry_url)
        # for w in wods:
        #     pubDate = w.find('pubDate').text
        #     encoded_content = w.find('encoded')
        #     for child in encoded_content.children:
        #         content = str(child)
        #     wod = {
        #         'pubDate': pubDate,
        #         'content': content,
        #     }
        #     wod_list.append(wod)
    except Exception as e:
        print('The scraping job failed. See exception: ')
        print(e)
    return json.dumps(wod_list)    

  def get_chrome_options():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
    return options    