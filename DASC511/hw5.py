
# Problem 1 (2 points)
# Assign the 'name' variable an object that is your name of type str.
name: str = "Dan Fawcett"

# Problem 2 (2 points)
# Create a class and implement it for your problem of interest
class Deuce():
  # Constant URL http://www.deucegym.com/community/2021-12-01/
  DEUCE_URL = "http://www.deucegym.com/community/" # Webscrape
  
  def __init__(self, gpp_type, *args, **kwargs):
    super(Deuce, self).__init__(*args, **kwargs)
    self._web_data = WebData(Deuce.DEUCE_URL, gpp_type, self.workout_dates)
    self.cycle_wods_json = self._web_data.cycle_wods_json()

# Problem 3 (2 points)
# Create another class and implement it for your problem of interest
from html.parser import HTMLParser

class HTMLTableParser(HTMLParser):
    """ This class serves as a html table parser. It is able to parse multiple
    tables which you feed in. You can access the result per .tables field.
    """
    def __init__(
        self,
        decode_html_entities=False,
        data_separator=' ',
    ):

        HTMLParser.__init__(self, convert_charrefs=decode_html_entities)

        self._data_separator = data_separator

        self._in_td = False
        self._in_th = False
        self._current_table = []
        self._current_row = []
        self._current_cell = []
        self.tables = []
        self.named_tables = {}
        self.name = ""

    def handle_starttag(self, tag, attrs):
        """ We need to remember the opening point for the content of interest.
        The other tags (<table>, <tr>) are only handled at the closing point.
        """
        if tag == "table":
            name = [a[1] for a in attrs if a[0] == "id"]
            if len(name) > 0:
                self.name = name[0]
        if tag == 'td':
            self._in_td = True
        if tag == 'th':
            self._in_th = True

    def handle_data(self, data):
        """ This is where we save content to a cell """
        if self._in_td or self._in_th:
            self._current_cell.append(data.strip())
    
    def handle_endtag(self, tag):
        """ Here we exit the tags. If the closing tag is </tr>, we know that we
        can save our currently parsed cells to the current table as a row and
        prepare for a new row. If the closing tag is </table>, we save the
        current table and prepare for a new one.
        """
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
            self.name = ""

# Problem 4 (2 points)
# Create another class and implement it for your problem of interest
class WebData:
  def __init__(self, url: str, gpp_type: str, workout_dates: list):
    self.url = url
    self.gpp_type = gpp_type
    self.workout_dates = workout_dates
    self.cycle_wods_json = self.webscrape_data_to_json 

  def url_get_contents(url) -> urllib.request._UrlopenRet:
    """ Opens a website and read its binary contents (HTTP Response Body) """
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    return f.read()
  
  def webscrape_data_to_json(self) -> str:
    json_formatted_str = ''
    driver = webdriver.Chrome(options=chrome_options)
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

# If you need to, you can create any additional classes or functions here as well.

# Replace \n and * in  and \t code with empty string
@staticmethod
def replace_chars(s: str) -> str:
    s = s.replace('\n', '').replace('*', '').replace('\t', '')
    return s  

# Problem 5 (2 points)
# Assign a variable named 'obj_1' an example instance of one of your classes
import urllib.request
import pprint

def main():
    url = 'https://w3schools.com/html/html_tables.asp'
    xhtml = url_get_contents(url).decode('utf-8')

    p = HTMLTableParser()
    p.feed(xhtml)

    # Get all tables
    pprint(p.tables)

    # Get tables with id attribute
    pprint(p.named_tables)


if __name__ == '__main__':
    main()

# Problem 6 (2 points)
#  Assign a variable named 'obj_2' an example instance of another one of your
#  classes


# Problem 7 (2 points)
#  Assign a variable named 'obj_3' an example instance of one of your classes
#  that extends another class


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

