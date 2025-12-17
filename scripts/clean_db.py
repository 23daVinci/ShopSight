from pymilvus import connections, utility

COLLECTION_NAME = "ShopSight_Inventory"

# 1. Connect
print("Connecting to Milvus...")
connections.connect("default", host="localhost", port="19530")

# 2. Check and Drop
if utility.has_collection(COLLECTION_NAME):
    print(f"Dropping collection: {COLLECTION_NAME}")
    utility.drop_collection(COLLECTION_NAME)
    print("Collection dropped successfully.")
else:
    print(f"Collection {COLLECTION_NAME} does not exist.")