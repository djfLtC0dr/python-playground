from wod import PushJerk
import os
import pymongo
from pymongo import MongoClient
import dns
import json

class Model:
  def __init__(self, pj_type):
    self.pj_type = pj_type
    # Set the connection string
    self._cnx_string = os.environ["CNX_STRING_1"] + os.environ["DB_USERNAME"] + ":" + os.environ["DB_PWD"] + os.environ["CNX_STRING_2"] + os.environ["DB_NAME"] + os.environ["CNX_STRING_3"]
    # Define a new client connection
    self._cnx = MongoClient(self._cnx_string)
    # Set the database
    self.db = self._cnx.db
    # Set the collection
    self.clx_wods = self.db.clx_wods

  @property
  def pj_type(self):
      return self.__pj_type

  @pj_type.setter
  def pj_type(self, value):
      self.__pj_type = value

  def scrape(self):
      #print ("value is:" + self.pj_type)
      pj = PushJerk(self.pj_type)
      pj_wods_json = json.loads(pj.wods_json)
      return pj_wods_json

  def get_collection(self):
      return self.clx_wods

  def insert_doc(self, doc) -> pymongo.results.InsertOneResult:
      return self.clx_wods.insert_one(doc)

  def get_collections(self):
      return self.db.list_collections()
 