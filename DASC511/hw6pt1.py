import urllib.request
from urllib.request import urlopen
import re

# Problem 1 (2 points)
# Assign the 'name' variable an object that is your name of type str.
name: str = "Dan Fawcett"

# Problem 2 (2 points)
# Create a class and implement it for your problem of interest
class Deuce():
  """ This class serves as a storage mechanism for list of urls"""  
  # Constant URL http://www.deucegym.com/community/2021-12-01/
  DEUCE_URL = "http://www.deucegym.com/community/" # Webscrape
  
  def __init__(self, gpp_type='DEUCE ATHLETICS GPP', workout_dates=['2021-12-01', '2021-12-02']):
    self.gpp_type = gpp_type
    self.workout_dates = workout_dates
    self.wod_urls = []
    self.get_wod_url()
    self.web_data:WebData = WebData()
    #self.cycle_wods_json = self._web_data.cycle_wods_json()

  def add_wod_url(self, a_href: str):
    self.wod_urls.append(a_href)

  def get_wod_url(self) -> None:
    # TODO: get this working for singleton then loop it => for wod_date in workout_dates:
    wod_date = self.workout_dates[0]
    wod_url_base = Deuce.DEUCE_URL + wod_date
    self.web_data.__init__(wod_url_base)
    xhtml = self.web_data.html_data
    list_wod_links = re.findall("href=[\"\'](.*?)[\"\']", xhtml)
    wod_url = ''
    for url in list_wod_links:
        if wod_url_base in url:
          wod_url = url
          break
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
          self.html_data: str = self.webscrape_html_data(self.url)

    def webscrape_html_data(self, url) -> str:
        """ Opens a website and read its binary contents (HTTP Response Body)"""   
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36", 
            "Content-type": "application/json; charset=utf-8",
            "Accept": "application/json",
            }
        req = urllib.request.Request(url=url, headers=headers)
        with urlopen(req) as response:
            response_content = replace_chars(response.read().decode('utf-8'))
        return response_content

# Problem 4 (2 points)
# Create another class and implement it for your problem of interest
from html.parser import HTMLParser

class HTMLDeuceParser(HTMLParser):
    """ This class serves as a html Deuce GPP parser. It extends HTMLParser and is able to parse div
    tags containing class="wod_block" which you feed in. You can access the result per .wod_data field.
    """
    def __init__(self):
        self.recording = False,
        self.wod_data = []
        #self.convert_charrefs = False
        # initialize the base class
        HTMLParser.__init__(self)

    def handle_starttag(self, tag, attrs):      
        if tag == 'div':
            for name, value in attrs:
                if name == 'class' and value == 'wod_block':
                    #print(value)
                    #print("Encountered the beginning of a %s tag" % tag)
                    self.recording = True 
        else:
            return

    def handle_endtag(self, tag):
        if tag == 'div' and self.recording == True:
            self.recording = False 
            #print("Encountered the end of a %s tag" % tag)

    def handle_data(self, data):
        if self.recording == True:
            self.wod_data.append(data)

# If you need to, you can create any additional classes or functions here as well.
# Replace \n and \t and \r embedded \0 strings with empty string
def replace_chars(s: str) -> str:
    s = s.replace('\n', '').replace('\t', '').replace('\r', '').replace('\0', '')
    return s  

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
obj_3 = HTMLDeuceParser()
obj_3.feed(obj_2.html_data)
#print(obj_3.wod_data)
obj_3.close()