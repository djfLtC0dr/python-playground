import os
import pymongo
import dns

class MongoDB: 
    def __init__(self):
      # Set the connection string
      self._cnx_string = os.environ["CNX_STRING_1"] + os.environ["DB_USERNAME"] + ":" + os.environ["DB_PWD"] + os.environ["CNX_STRING_2"] + os.environ["DB_NAME"] + os.environ["CNX_STRING_3"]
      # Define a new client to connect
      self._client = pymongo.MongoClient(self._cnx_string)
      # Set the database
      self._db = self._client['woddb']
      
    def create_collection(self, name: str = "") -> pymongo.collection:
      return self._db[name]

    def list_collections(self) -> list:
      return self._db.list_collection_names()



# # Connecting the to collection
# clx_wod = db['clxWod']

# query = {"date": '31 Mar 2022'}
# d = clx_wod.delete_many(query)
# print(d.deleted_count, " documents deleted !!")

