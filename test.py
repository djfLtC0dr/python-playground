
import unittest
import datetime
from db import MongoDB
from pymongo import MongoClient


class MyTestCases(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    mongo = MongoDB  # instantiate the MongoDB Class

    # test case function to check the MongoDB.set_connection function
    def test_0_set_connection_string(self):
        print("Start set_connection test\n")
        cnx_string = self.mongo.MONGO_URI
        db_name = 'woddb'
        # check if the db_name is in the cnx_string means load_dotenv worked
        self.assertIn(db_name, cnx_string)  # null db id will fail the test
        print(cnx_string)
        print("\nFinish set_connection test\n")    
    
    # test case function to check the MongoDB.set_connection function
    def test_1_set_mongo_client_cnx(self):
        print("Start mongo_client_cnx test\n")
        cnx = self.mongo.cnx
        # check if the obtained user id is null or not
        self.assertIsInstance(cnx, MongoClient) 
        print(cnx.server_info())
        print("\nFinish set_connection test\n")   

    # test case function to check the MongoDB.get_db function
    def test_2_get_db(self):
        print("Start get_db test\n")
        db_name = str(self.mongo.get_database(self))
        # check if the obtained db is null or not
        self.assertIn('woddb', db_name)  # null db id will fail the test
        print(db_name)
        print("\nFinish get_db test\n")

    # test insert_doc
    def test_3_insert_doc(self):
        print("Start insert_doc test\n")
        date_time = datetime.datetime(2022, 5, 13, 0, 0)
        sqt = '195'
        doc = {'date': date_time, 'five_rm_sqt': sqt}
        result = self.mongo.insert_doc(self, doc)
        self.assertTrue(result.acknowledged)
        print(result.acknowledged)
        print("\nFinish insert_doc test\n")

    # test case function to check the Person.get_name function
    def test_4_get_collection(self):
        print("\nStart get_collection test\n")
        clx = self.mongo.get_collection(self)
        self.assertIsNotNone(clx)
        print(clx)
        print("\nFinish get_collection test\n")

    def test_5_delete_docs(self):
        print("\nStart delete_docs test\n")        
        # delete all docs
        deleted_count = self.mongo.delete_doc(self)
        self.assertGreater(deleted_count, 0)
        print(f'Deleted {deleted_count} docs')
        self.mongo.cnx.close()
        print("\nFinish delete_docs test\n")       
        self.mongo.cnx.close() 

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()