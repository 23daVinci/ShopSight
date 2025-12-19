import os
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from ingest import ShopSightService  # <--- Import your new class

# Config
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")

# Initialize Service (Singleton)
shop_service = ShopSightService(host=MILVUS_HOST)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to DB and Load Models
    shop_service.connect()
    yield
    # Shutdown logic (optional)

app = FastAPI(title="ShopSight API", lifespan=lifespan)

@app.post("/search")
async def search_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Delegate logic to the service class
    results = shop_service.search(contents)
    
    return {"matches": results}

@app.post("/admin/ingest")
async def ingest_endpoint():
    """
    Optional: Trigger ingestion via API (useful for testing)
    """
    # You could add your bulk ingestion logic to the class and call it here
    return {"message": "Ingestion functionality can be added to the class!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)