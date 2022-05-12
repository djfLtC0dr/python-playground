import os
import pymongo
import dns

class MongoDB: 
    def __init__(self):
      # Set the connection string
      self._cnx_string = os.environ["CNX_STRING_1"] + os.environ["DB_USERNAME"] + ":" + os.environ["DB_PWD"] + os.environ["CNX_STRING_2"] + os.environ["DB_NAME"] + os.environ["CNX_STRING_3"]
      # Define a new client to connect
      self.client = pymongo.MongoClient(self._cnx_string)
      # Set the database
      self.db = self.get_database()
      self.clx = self.get_collection()
      self.list_collections = self.get_collections()

    def get_database(self):
      return self.client['woddb']
    
    def get_collection(self):
      return self.db['clx_wods']

    def get_collections(self):
      return self.db.list_collections()

    def insert_doc(self, doc):
      self.clx.insert_one(doc)



# # Connecting the to collection
# clx_wod = db['clxWod']

# query = {"date": '31 Mar 2022'}
# d = clx_wod.delete_many(query)
# print(d.deleted_count, " documents deleted !!")

