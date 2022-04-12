import os
import pymongo
import dns

# Define a new client.
cnxString = os.environ["CNX_STRING_1"] + os.environ["DB_USERNAME"] + ":" + os.environ["DB_PWD"] + os.environ["CNX_STRING_2"] + os.environ["DB_NAME"] + os.environ["CNX_STRING_3"]
#mongodb+srv://djfDBUser:<password>@djfcluster.lg6aj.mongodb.net/myFirstDatabase?retryWrites=true&w=majority
#print(cnxString)
client = pymongo.MongoClient(cnxString)

# Get the database (database name by default is "test")
db = client.woddb # OR db = client.test

# Getting the collection
clx = db.list_collection_names()

print(clx)
