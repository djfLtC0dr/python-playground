from wod import PushJerk
from db import MongoDB
import json

class Model:
  def __init__(self, pj_type):
    self.pj_type = pj_type
    self.db = MongoDB

  @property
  def pj_type(self):
    return self.__pj_type

  @pj_type.setter
  def pj_type(self, value):
    """
    Validate the email
    :param value:
    :return:
    """
    self.__pj_type = value

  def scrape(self):
    #print ("value is:" + self.pj_type)
    pj = PushJerk(self.pj_type)
    pj_wods_json = json.loads(pj.wods_json)
    return pj_wods_json

  def get_collections(self):
    return self.db.get_collections(self)

  def insert_doc(self, doc):
    self.db.insert_doc(self, doc=doc)