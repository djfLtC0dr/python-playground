import os
import pymongo
from pymongo import MongoClient
import dns

class MongoDB: 
    def __init__(self):
      # Set the connection string
      self._cnx_string = os.environ["CNX_STRING_1"] + os.environ["DB_USERNAME"] + ":" + os.environ["DB_PWD"] + os.environ["CNX_STRING_2"] + os.environ["DB_NAME"] + os.environ["CNX_STRING_3"]
      # Define a new client connection
      self._cnx = MongoClient(self._cnx_string)
      # Set the database
      self.db = self._cnx.db
      # Set the collection
      self.clx_wods = self.db.clx_wods
      self.collections = self.get_collections()
      print(self.collections)

    def get_collection(self):
      return self.clx_wods

    def insert_doc(self, doc) -> pymongo.results.InsertOneResult:
      return self.clx_wods.insert_one(doc)

    def get_collections(self):
      return self.db.list_collections()

# # Connecting the to collection
# clx_wod = db['clxWod']

# query = {"date": '31 Mar 2022'}
# d = clx_wod.delete_many(query)
# print(d.deleted_count, " documents deleted !!")

