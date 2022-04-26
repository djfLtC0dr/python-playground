import urllib.request
import re

# Problem 1 (2 points)
# Assign the 'name' variable an object that is your name of type str.
name: str = "Dan Fawcett"

# Problem 2 (2 points)
# Create a class and implement it for your problem of interest
class Deuce():
  # Constant URL http://www.deucegym.com/community/2021-12-01/
  DEUCE_URL = "http://www.deucegym.com/community/" # Webscrape
  
  def __init__(self, gpp_type='GARAGE', workout_dates=['2021-12-01', '2021-12-02']):
    self.gpp_type = gpp_type
    self.workout_dates = workout_dates
    self.wod_urls = []
    self.get_wod_url()
    #self.cycle_wods_json = self._web_data.cycle_wods_json()

  def add_wod_url(self, a_href: str):
    self.wod_urls.append(a_href)

  def get_wod_url(self) -> None:
    # TODO: get this working for singleton the loop it => for wod_date in workout_dates:
    wod_date = self.workout_dates[0]
    wod_url_base = Deuce.DEUCE_URL + wod_date
    xhtml = url_get_contents(Deuce.DEUCE_URL + wod_date).decode('utf-8')
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
from html.parser import HTMLParser

class HTMLElementParser(HTMLParser):
    """ This class serves as a html table parser. It is able to parse multiple
    tables which you feed in. You can access the result per .tables field.
    """
    def __init__(self):
        self.data_separator=' ',
        self.recording = False,
        self.data = []
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
            self.data.append(data)

"""         self._data_separator = data_separator

        self._in_td = False
        self._in_th = False
        self._current_table = []
        self._current_row = []
        self._current_cell = []
        self.tables = []
        self.named_tables = {}
        self.name = ""

    def handle_starttag(self, tag, attrs):
        #We need to remember the opening point for the content of interest.
        #The other tags (<div>, <p>) are only handled at the closing point.
        
        if tag == "div":
            name = [a[1] for a in attrs if a[0] == "id"]
            if len(name) > 0:
                self.name = name[0]
        if tag == 'td':
            self._in_td = True
        if tag == 'th':
            self._in_th = True

    def handle_data(self, data):
        #This is where we save content to a cell
        if self._in_td or self._in_th:
            self._current_cell.append(data.strip())
    
    def handle_endtag(self, tag):
        #Here we exit the tags. If the closing tag is </tr>, we know that we
        can save our currently parsed cells to the current table as a row and
        prepare for a new row. If the closing tag is </table>, we save the
        current table and prepare for a new one.
        
        if tag == 'td':
            self._in_td = False
        elif tag == 'th':
            self._in_th = False

        if tag in ['td', 'th']:
            final_cell = self._data_separator.join(self._current_cell).strip()
            self._current_row.append(final_cell)
            self._current_cell = []
        elif tag == 'tr':
            self._current_table.append(self._current_row)
            self._current_row = []
        elif tag == 'table':
            self.tables.append(self._current_table)
            if len(self.name) > 0:
                self.named_tables[self.name] = self._current_table
            self._current_table = []
            self.name = "" """



# Problem 4 (2 points)
# Create another class and implement it for your problem of interest
class WebData:
  def __init__(self, url: str = ''):
    self.url = url
    self.html_data = self.webscrape_html_data(self.url)


  def webscrape_html_data(self, url) -> str:
    """ Scrape the landing page for the day, then  pull out all the href tags
    re-scrape for the actual wod embedded in the linkable article w/in 
    <div class="entry-summary>
    """
    xhtml = url_get_contents(url).decode('utf-8')
    return replace_chars(xhtml)

# If you need to, you can create any additional classes or functions here as well.
def url_get_contents(url):
    """ Opens a website and read its binary contents (HTTP Response Body) """   
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36", 
        }
    req = urllib.request.Request(url=url, headers=headers)
    f = urllib.request.urlopen(req)
    return f.read()

# Replace \n and \t code with empty string
def replace_chars(s: str) -> str:
    s = s.replace('\n', '').replace('\t', '')
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
obj_3 = HTMLElementParser()
obj_3.feed(obj_2.html_data)
print(obj_3.data)
obj_3.close()
    
# Get all tables
#pprint(obj_3.tables)

# Get tables with id attribute
#pprint(obj_3.named_tables)

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

    # Problem 9
    # add test case method

    # Problem 10
    # add test case method
    
    # Problem 11
    # add test case method

    # Problem 12
    # add test case method

    # Problem 13
    # add test case method

    # Problem 14
    # add test case method

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

