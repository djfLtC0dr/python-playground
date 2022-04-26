import urllib.request
from urllib.request import urlopen
import re

import urllib.response  

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
    #self.cycle_wods_json = self._web_data.cycle_wods_json()

  def add_wod_url(self, a_href: str):
    self.wod_urls.append(a_href)

  def get_wod_url(self) -> None:
    # TODO: get this working for singleton then loop it => for wod_date in workout_dates:
    wod_date = self.workout_dates[0]
    wod_url_base = Deuce.DEUCE_URL + wod_date
    wd = WebData(wod_url_base)
    xhtml = wd.html_data
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
        self.html_data = self.webscrape_html_data(self.url)

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
    
# Problems 8 through 14 are worth 4 points each.
#    For each problem you must implement a test method in the following
#     TestCase class. Each method name should be unique and start with 'test_'.
#     Each method should test something different about the classes you created
#     above. Use unittest.TestCase's assert methods to check your implementation.
#     You will get 2 points for each test method created. You will get 2
#     additional points for each of these tests that complete successfully.
#     See https://docs.python.org/3/library/unittest.html for examples.

import unittest
import sys
import inspect
###################################################
#DO NOT MODIFY this class's name or what it extends
###################################################
class MyTestCases(unittest.TestCase):

    # Problem 8
    # add test case method
    def test_deuce_url_exists(self) -> bool:
        deuce1 = Deuce('DEUCE GARAGE GPP', ['2021-12-03'])
        self.assertTrue(len(deuce1.wod_urls) > 0)

    # Problem 9
    # add test case method
    def test_deuce_url_is_type_str(self) -> bool:
        deuce2 = Deuce('DEUCE GARAGE GPP', ['2021-12-03'])
        self.assertTrue(isinstance(deuce2.wod_urls[0], str))

    # Problem 10
    # add test case method
    def test_web_data_exists(self) -> bool:
        deuce3 = Deuce('DEUCE GARAGE GPP', ['2021-12-03'])
        wd1 = WebData(deuce3.wod_urls[0])
        self.assertTrue(len(wd1.html_data) > 0)

    # Problem 11
    # add test case method
    def test_div_wodblock_exists(self) -> bool:
        deuce4 = Deuce('DEUCE GARAGE GPP', ['2021-12-03'])
        wd2 = WebData(deuce4.wod_urls[0])
        regexp = re.compile(r'div class="wod_block"')
        self.assertTrue(bool(re.search(regexp, wd2.html_data)))

    # Problem 12
    # add test case method
    def test_div_wodblock__athletics__gpp_parsed(self) -> bool:
        deuce5 = Deuce('DEUCE ATHLETICS GPP', ['2021-12-03'])
        wd3 = WebData(deuce5.wod_urls[0])
        hdp1 = HTMLDeuceParser()
        hdp1.feed(wd3.html_data)
        test_data = hdp1.wod_data
        self.assertTrue(deuce5.gpp_type in test_data)  

    # Problem 13
    # add test case method
    def test_div_wodblock_garage_gpp_parsed(self) -> bool:
        deuce6 = Deuce('DEUCE GARAGE GPP', ['2021-12-03'])
        wd4 = WebData(deuce6.wod_urls[0])
        hdp2 = HTMLDeuceParser()
        hdp2.feed(wd4.html_data)
        test_data = hdp2.wod_data
        self.assertTrue(deuce6.gpp_type in test_data)   

    # Problem 14
    # add test case method
    def test_div_all_wod_elements_parsed(self) -> bool:
        deuce7 = Deuce('DEUCE GARAGE GPP', ['2021-12-03'])
        wd5 = WebData(deuce7.wod_urls[0])
        hdp3 = HTMLDeuceParser()
        hdp3.feed(wd5.html_data)
        test_data = hdp3.wod_data
        self.assertEqual(20, len(test_data))  
    ##################################################
    #DO NOT MODIFY any of these test case methods
    ##################################################

    def get_classes(self):
        clses = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__ and member is not MyTestCases)
        return clses

    def test_name_assigned(self):
        m = sys.modules[__name__]
        name = getattr(m,"name",None)
        self.assertTrue(name is not None)
        self.assertTrue(isinstance(name,str))
        self.assertTrue(len(name) > 0)

    def test_one_class_created(self):
        self.assertTrue(len(self.get_classes()) >= 1)

    def test_two_classes_created(self):
        self.assertTrue(len(self.get_classes()) >= 2)

    def test_three_classes_created(self):
        self.assertTrue(len(self.get_classes()) >= 3)

    def test_obj_1_instance_created(self):
        clses = self.get_classes()
        m = sys.modules[__name__]
        obj_1 = getattr(m,"obj_1",None)
        self.assertTrue(obj_1 is not None)
        self.assertTrue(isinstance(obj_1,tuple([cls[1] for cls in clses])))

    def test_obj_2_instance_created(self):
        clses = self.get_classes()
        m = sys.modules[__name__]
        obj_2 = getattr(m,"obj_2",None)
        obj_1 = getattr(m,"obj_1",None)
        self.assertTrue(obj_2 is not None)
        self.assertTrue(isinstance(obj_2,tuple([cls[1] for cls in clses])))
        self.assertNotEqual(type(obj_2),type(obj_1))

    def test_obj_3_instance_created(self):
        clses = self.get_classes()
        m = sys.modules[__name__]
        obj_3 = getattr(m,"obj_3",None)
        self.assertTrue(obj_3 is not None)
        self.assertTrue(isinstance(obj_3,tuple([cls[1] for cls in clses])))
        bases = tuple(b for b in type(obj_3).__bases__ if b is not object)
        print(dir(obj_3))
        self.assertTrue(len(bases) > 0)
        
##################################################
#DO NOT MODIFY any of the code below!
##################################################
if __name__ == "__main__":
    unittest.main()

