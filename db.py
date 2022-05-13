from dotenv import load_dotenv
import os
import pymongo
from pymongo import MongoClient
import dns

class MongoDB: 
    load_dotenv() 
    MONGO_URI = str(os.environ["CNX_STRING_1"] + os.environ["DB_USERNAME"] + ":" + os.environ["DB_PWD"] + os.environ["CNX_STRING_2"] + os.environ["DB_NAME"] + os.environ["CNX_STRING_3"])
    cnx = MongoClient(MONGO_URI)
    
    def __init__(self):        
        # Set the database
        self.db = self.get_database()
        # Set the collection
        self.collection = self.get_collection()

    def get_database(self):
        db = MongoDB.cnx.get_database()
        return db
    
    def get_collection(self):
        db = MongoDB.cnx.get_database()
        clx = db.get_collection('clx_wods')
        return clx

    def insert_doc(self, doc) -> pymongo.results.InsertOneResult:
        db = MongoDB.cnx.get_database()
        clx = db.get_collection('clx_wods')        
        return clx.insert_one(doc)

    def get_collections(self):
        db = MongoDB.cnx.get_database()
        return db.list_collections()

    # Delete docs from collection
    def delete_doc(self, doc_id:int = 0):
        db = MongoDB.cnx.get_database()
        clx = db.get_collection('clx_wods')      
        query = {'_id': doc_id} 
        d = clx.delete_many(clx.find({}, query))
        return d.deleted_count
