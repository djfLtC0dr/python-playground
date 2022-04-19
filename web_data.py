# selenium 
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import ssl

chrome_options = Options() 
chrome_options.add_argument('--no-sandbox') 
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("--headless")

class WebData:
    # Constant URL + File Strings
    DEUCE = "http://www.deucegym.com/community/" # Webscrape
    SSLP = "https://startingstrength.com/get-started/programs" # Webscrape    
    driver = webdriver.Chrome(options=chrome_options)

    def __init__(self, url: str, gpp_type: str):
        self.url = url
        self.gpp_type = gpp_type

    # Replace \n and * in code with empty string
    def replace_chars(self, s: str) -> str:
        s.replace('\n', '').replace('*', '') 
        return s   

    # Pulls data from websites using selenium--SSLP is static tables, DEUCE is dynamic
    def webscrape(self) -> None:
        if self.gpp_type == 'NLP': # NOVICE Starting Strength
            self.driver.get(self.url)
            try:
                nlp_elements = self.driver.find_elements(By.CLASS_NAME, "proggy")
                # df.append deprecated so using tmp list of dataframes to append then concat 
                tmp = []      
                for nlp_phase in nlp_elements:
                    nlp_phase_innerHTML = self.replace_chars(nlp_phase.get_attribute('innerHTML'))
                    tmp.append(pd.read_html("<table>" + nlp_phase_innerHTML + "</table>")[0])
                nlp_df = pd.concat(tmp, ignore_index=True)
                nlp_df.to_csv('sslp.csv', encoding='utf-8', index=False)
                #print(calc_sslp_ph1())
            finally:
                self.driver.quit()      
        else: # DEUCE GPP Athlete or Garage Gym Warrior
            for wod_date in self.workout_dates:
                self.driver.get(self.url+wod_date)
            try:
                wod_link = self.driver.find_element_by_partial_link_text(self.url)
                ActionChains(self.driver).move_to_element(wod_link).click(wod_link).perform()
                wod_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "wod_block"))
                )
                wod_innerHTML = wod_element.get_attribute('innerHTML')
                df_wod = pd.read_html("<table>" + wod_innerHTML + "</table>")
                print(df_wod)
            finally:
                self.driver.quit()  