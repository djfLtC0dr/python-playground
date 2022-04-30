# import urllib.request
# from urllib.request import urlopen
from html.parser import HTMLParser
#import re
import requests
from requests_html import HTMLSession
from requests_html import HTML
import pandas as pd

# Problem 1 (2 points)
# Assign the 'name' variable an object that is your name of type str.
name: str = "Dan Fawcett"

# Problem 2 (2 points)
# Create a class and implement it for your problem of interest
class Deuce():
  """ This class serves as a storage mechanism for list of urls"""  
  # Constant URL http://www.deucegym.com/community/2021-12-01/
  DEUCE_URL = "http://www.deucegym.com/community/" # Webscrape
  A_BLOG_XPATH = '/html/body/div[1]/main/center/article/div/a'
  DIV_WODBLOCK_XPATH = '/html/body/div[1]/main/center/article/div/div[3]'
  DEUCE_ATHLETICS_GPP = 'DEUCE ATHLETICS GPP'
  DEUCE_GARAGE_GPP = 'DEUCE GARAGE GPP'
  
  def __init__(self, gpp_type=DEUCE_GARAGE_GPP, workout_dates=['2021-12-01', '2021-12-02']):
    self.gpp_type = gpp_type
    self.workout_dates = workout_dates
    self.wod_urls = []
    self.get_wod_url()
    self.web_data = WebData()
    #self.cycle_wods_json = self._web_data.cycle_wods_json()

  def add_wod_url(self, a_href: str):
    self.wod_urls.append(a_href)

  def get_wod_url(self) -> None:
    # TODO: get this working for singleton then loop it => for wod_date in workout_dates:
    wod_date = self.workout_dates[0]
    wod_url_base = Deuce.DEUCE_URL + wod_date
    self.web_data = WebData(wod_url_base)
    obj_html = self.web_data.html
    sel_url = obj_html.xpath(Deuce.A_BLOG_XPATH)
    #list_wod_links = re.findall("href=[\"\'](.*?)[\"\']", xhtml)
    wod_url = sel_url[0].links.pop()
    # for url in list_wod_links:
    #     if wod_url_base in url:
    #       wod_url = url
    #       break
    #print("wod_url => ", wod_url)
    #a_href =  url_get_contents(wod_url[0]).decode('utf-8')
    self.add_wod_url(wod_url)

# Problem 3 (2 points)
# Create another class and implement it for your problem of interest
class WebData:
    """ This class serves as a storage mechanism for an HTTPResponse object data decoded to utf-8 string"""    
    def __init__(self, url: str = ''):
        self.url = url
        if url != '':
          self.html: HTML = self.webscrape_html_data(self.url)

    def webscrape_html_data(self, url) -> HTML:
        """ Opens a website and read its binary contents (HTTP Response Body)"""   
        try:
            session = HTMLSession()
            raw_html_data = session.get(url)     
            clean_html_data = HTML(html=replace_chars(raw_html_data.text))
        except requests.exceptions.RequestException as e:
            print(e)
        return clean_html_data

# Problem 4 (2 points)
# Create another class and implement it for your problem of interest
class HTMLDeuceParser(HTMLParser):
    """ This class serves as a html Deuce GPP parser. It extends HTMLParser and is able to parse div
    tags containing class="wod_block" which you feed in. You can access the result per .wod_table field.
    """
    def __init__(self, html: HTML):
        self.recording = False
        self.html = html
        self.wodblock = self.html.xpath(Deuce.DIV_WODBLOCK_XPATH) 
        self.tag = ''
        self.wod_table = []
        #self.convert_charrefs = False
        # initialize the base class
        HTMLParser.__init__(self)         

    def handle_starttag(self, tag, attrs):      
        if tag == 'div':
            self.tag = 'th'
            # for value in attrs:
                # print(value)
                # print("Encountered the beginning of a %s tag" % tag)
            self.recording = True 
        elif tag == 'h2':
            self.tag = 'th'
            # for value in attrs:
                # print(value)
                # print("Encountered the beginning of a %s tag" % tag)
            self.recording = True  
        elif tag == 'p':
            self.tag = 'tr'
            # for value in attrs:
                # print(value)
                # print("Encountered the beginning of a %s tag" % tag)
            self.recording = True                 
        elif tag == 'span':
            self.tag = 'td'
            # for value in attrs:
                # print(value)
                # print("Encountered the beginning of a %s tag" % tag)
            self.recording = True                            
        else:
            self.recording = True
            return

    def handle_endtag(self, tag):
        if tag == 'div' and self.recording == True:
            self.tag = 'th'
            self.recording = False 
            # print("Encountered the end of a %s tag" % tag)
        elif tag == 'h2' and self.recording == True:
            self.tag = 'th'
            self.recording = False 
            # print("Encountered the end of a %s tag" % tag)     
        elif (tag == 'b' or tag == 'br') and self.recording == True:
            pass # FIXME: Does this break the table?
            # self.recording = False
        elif tag == 'p' and self.recording == True:
            self.tag = 'tr'
            self.recording = False 
            # print("Encountered the end of a %s tag" % tag)       
        elif tag == 'span' and self.recording == True:
            self.tag = 'td'
            self.recording = False 
            # print("Encountered the end of a %s tag" % tag)              
        else:
            return # We don't want <h3>              

    def handle_data(self, data):
        if self.recording == True:
            self.wod_table.append('<' + self.tag + '>' + data + '</' + self.tag + '>')

# If you need to, you can create any additional classes or functions here as well.
# Replace \n and \t and \r embedded \0 strings with empty string
def replace_chars(s: str) -> str:
    s = s.replace('\n', '').replace('\t', '').replace('\r', '').replace('\0', '')
    return s  

def create_table(table_data: list) -> str:
    table_data[0] = '<table><tr>' + str(table_data[0]) + '</tr><tr>'
    garage_idx = table_data.index('<th>' + Deuce.DEUCE_GARAGE_GPP + '</th>')
    table_data[garage_idx] = '</tr><tr><th>' + Deuce.DEUCE_GARAGE_GPP + '</th>'
    table_data[-1] = str(table_data[-1]) + '</tr></table>'
    created_table = ''.join(table_data)
    return created_table
# Problem 5 (2 points)
# Assign a variable named 'obj_1' an example instance of one of your classes
obj_1 = Deuce()    
    
# Problem 6 (2 points)
#  Assign a variable named 'obj_2' an example instance of another one of your
#  classes
obj_2 = WebData(obj_1.wod_urls[0])

# Problem 7 (2 points)
#  Assign a variable named 'obj_3' an example instance of one of your classes
#  that extends another class
obj_3 = HTMLDeuceParser(obj_2.html)
obj_3.feed(obj_3.wodblock[0].html)
html_table = create_table(obj_3.wod_table)
df_wod = pd.read_html(html_table)
print(df_wod)
# obj_3 = HTMLDeuceParser()
# obj_3.feed(obj_2.html_data
# #print(obj_3.wod_data)
# obj_3.close()