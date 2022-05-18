from dotenv import load_dotenv
from db import MongoDB
from wod import PushJerk
import json

class Model:
  def __init__(self, pj_type):
    load_dotenv() 
    self.mongo = MongoDB
    self.pj_type = pj_type

  @property
  def pj_type(self):
      return self.__pj_type

  @pj_type.setter
  def pj_type(self, value):
      self.__pj_type = value

  def scrape(self):
      pj = PushJerk(self.pj_type)
      pj_wods_json = json.loads(pj.wods_json)
      return pj_wods_json

  def get_collection(self):
      return self.mongo.get_collection(self)

  def insert_doc(self, doc):
      return self.mongo.insert_doc(self, doc)

  def get_collections(self):
      return self.mongo.get_collections(self)
 