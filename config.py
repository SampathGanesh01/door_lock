import os

# MongoDB Connection String
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://sampathasbl_db_user:r6pfNjkPIsXx1L8t@asbl-door.p34tuyr.mongodb.net/?appName=asbl-door")

# Database name
DATABASE_NAME = "facedoor_db"

# Collection name for faces
COLLECTION_NAME = "users"
