import os
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ingest import ShopSightService

# Config
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
DATA_DIR = "/data/images" 

shop_service = ShopSightService(host=MILVUS_HOST)

# Startup and Shutdown Events
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting ShopSight API...")
    shop_service.connect()
    yield
    # Add any necessary cleanup here (eg. closing DB connections)
    print("🛑 Shutting down...")

# FastAPI App Initialization
app = FastAPI(title="ShopSight API", lifespan=lifespan)

# CORS Middleware Setup for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    results = await asyncio.to_thread(shop_service.search, contents)
    return {"matches": results}

# --- UPDATED: Ingestion Endpoint ---
@app.post("/admin/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks, 
    batch_size: int = 50, 
    max_batches: int = None
):
    """
    Triggers background ingestion with batch control.
    Params:
      - batch_size: Number of images per DB insert (default: 50)
      - max_batches: Stop after inserting this many batches (optional)
    """
    if not os.path.exists(DATA_DIR):
        return {"error": f"Directory {DATA_DIR} not found. Check docker volumes."}

    # Pass parameters to the background task
    background_tasks.add_task(run_ingestion, batch_size, max_batches)
    
    msg = f"Ingestion started. Batch Size: {batch_size}, Max Batches: {max_batches if max_batches else 'Unlimited'}."
    return {"message": msg}

def run_ingestion(batch_size: int = 50, max_batches: int = None):
    print(f"📦 Scanning {DATA_DIR}...")
    
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    # Sort files for deterministic order
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(valid_exts)])
    
    total_files = len(files)
    if total_files == 0:
        print("⚠️ No images found to ingest.")
        return

    print(f"🚀 Starting ingestion. Found {total_files} images.")
    print(f"⚙️ Config: Batch Size={batch_size}, Max Batches={max_batches if max_batches else 'Unlimited'}")

    current_batch_paths = []
    current_batch_vectors = []
    batches_processed = 0

    for i, filename in enumerate(files, 1):
        file_path = os.path.join(DATA_DIR, filename)
        
        # Print progress relative to total files
        print(f"[{i}/{total_files}] Processing {filename}...")

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            
            vector = shop_service.get_embedding(content)
            
            if vector is not None:
                current_batch_paths.append(filename)
                current_batch_vectors.append(vector.flatten())
                
                # Check if batch is full
                if len(current_batch_paths) >= batch_size:
                    _batch_insert(current_batch_paths, current_batch_vectors, batches_processed + 1)
                    
                    # Reset batch and increment counter
                    current_batch_paths, current_batch_vectors = [], []
                    batches_processed += 1
                    
                    # Check exit condition
                    if max_batches and batches_processed >= max_batches:
                        print(f"🛑 Max batches limit ({max_batches}) reached. Stopping ingestion.")
                        return

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Process any remaining images as the final batch
    # (Only if we haven't hit the max_batches limit yet)
    if current_batch_paths:
        _batch_insert(current_batch_paths, current_batch_vectors, batches_processed + 1)
    
    print("✅ Ingestion Process Complete.")

def _batch_insert(paths, vectors, batch_num):
    print(f"💾 [Batch {batch_num}] Inserting {len(paths)} vectors into Milvus...")
    shop_service.collection.insert([paths, vectors])
    shop_service.collection.flush()
    
    # Re-index to ensure immediate searchability
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    shop_service.collection.create_index("embedding", index_params)