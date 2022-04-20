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
    

    def __init__(self, url: str, gpp_type: str):
        self.url = url
        self.gpp_type = gpp_type

    # Replace \n and * in  and \t code with empty string
    @staticmethod
    def replace_chars(s: str) -> str:
        s = s.replace('\n', '').replace('*', '').replace('\t', '')
        return s   

    # Pulls data from websites using selenium--SSLP is static tables, DEUCE is dynamic
    # Printing the static SSLP to csv for parsing
    # DEUCE ultimately needs to be jsonified and saved to db
    def webscrape(self, workout_dates=None) -> None:
        driver = webdriver.Chrome(options=chrome_options)
        if self.gpp_type == 'SSLP': # SSLP: NOVICE Starting Strength
            driver.get(self.url)
            try:
                nlp_elements = driver.find_elements(By.CLASS_NAME, "proggy")
                # df.append deprecated so using tmp list of dataframes to append then concat 
                tmp = []      
                for nlp_phase in nlp_elements:
                    nlp_phase_innerHTML = self.replace_chars(nlp_phase.get_attribute('innerHTML'))
                    tmp.append(pd.read_html("<table>" + nlp_phase_innerHTML + "</table>")[0])
                nlp_df = pd.concat(tmp, ignore_index=True)
                nlp_df.to_csv('sslp.csv', encoding='utf-8', index=False)
                #print(calc_sslp_ph1())
            finally:
                driver.quit()      
        else: # DEUCE GPP Athlete or GARAGE Gym Warrior
            # TODO: get this working for singleton the loop it => for wod_date in workout_dates:
            wod_date = workout_dates[0]
            driver.get(self.url+wod_date)
            popup_xpath = '/html/body/div[3]/div/img'
            try:
                popup = driver.find_element_by_xpath(popup_xpath)
                if popup.is_displayed:
                    popup.click() # Closes the popup
                    inner_html = '\n\t\t\t<h3>11/29/21 WOD</h3>\n\t\t\t<h2 style="text-align: center;"><b>DEUCE ATHLETICS GPP</b></h2>\n<p><span style="font-weight: 400;">Complete 4 rounds for quality of:</span></p>\n<p><span style="font-weight: 400;">8 Barbell Strict Press </span><span style="font-weight: 400;">(3x1x)<br>\n</span><span style="font-weight: 400;">8 Single Kettlebell Lateral Lunge</span></p>\n<p><span style="font-weight: 400;">Then, AMRAP 12</span></p>\n<p><span style="font-weight: 400;">1,2,3,…,∞<br>\n</span><span style="font-weight: 400;">Front Squat (135/95)<br>\n</span><span style="font-weight: 400;">DB Renegade Row (40/20)</span></p>\n<p><span style="font-weight: 400;">**Every 2 min, 1 7th Street Corner Run</span></p>\n<h2 style="text-align: center;"><b>DEUCE GARAGE GPP</b></h2>\n<p><span style="font-weight: 400;">5-5-5-5-5<br>\n</span><span style="font-weight: 400;">Pendlay Row</span></p>\n<p><span style="font-weight: 400;">Then, complete 3 rounds for quality of:</span></p>\n<p><span style="font-weight: 400;">10 Single Arm Bent Over row (ea)<br>\n</span><span style="font-weight: 400;">10-12 Parralette Push Ups<br>\n</span><span style="font-weight: 400;">10 Hollow Body Lat Pulls&nbsp;</span></p>\n<p><span style="font-weight: 400;">Then, AMRAP8</span></p>\n<p><span style="font-weight: 400;">6 Chest to Bar Pull Ups<br>\n</span><span style="font-weight: 400;">8 HSPU<br>\n</span><span style="font-weight: 400;">48 Double unders</span></p>\n\t\t'
                    inner_html = self.replace_chars(inner_html)
                    df_wod = pd.read_html('<table>' + inner_html + '</table>')
                    print(df_wod)
                else:
                    wod_link_xpath = '/html/body/div[1]/main/center/article/div/p/a'
                    wod_link = driver.find_element_by_xpath(wod_link_xpath)
                    ActionChains(driver).move_to_element(wod_link).click(wod_link).perform()
                    wod_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "wod_block"))
                    )
                    wod_innerHTML = wod_element.get_attribute('innerHTML')
                    df_wod = pd.read_html("<table>" + wod_innerHTML + "</table>")
                    print(df_wod)
            except Exception as e:
                print(str(e))
            finally:
                driver.quit()  