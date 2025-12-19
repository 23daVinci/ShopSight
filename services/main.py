import io
import os
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from pymilvus import connections, Collection

# --- CONFIGURATION ---
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
COLLECTION_NAME = "ShopSight_Inventory"
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = "19530"

# --- 1. GLOBAL MODEL LOADING ---
# Load model into memory ONCE when app starts.
print("Loading Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# --- 2. MILVUS CONNECTION ---
print("Connecting to Milvus...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)
collection.load() # Ensure collection is in memory

app = FastAPI(title="ShopSight Visual Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In real production, specify your domain (e.g., "http://myapp.com")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_embedding(image_bytes):
    """
    Helper: Converts raw image bytes -> Vector
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Return numpy array of the [CLS] token
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    """
    Endpoint: Upload an image -> Get similar product paths
    """
    # 1. Read Image
    contents = await file.read()
    
    # 2. Generate Vector
    query_vector = get_embedding(contents)
    
    if query_vector is None:
        return {"error": "Could not process image"}

    # 3. Search Milvus
    search_params = {
        "metric_type": "L2", 
        "params": {"nprobe": 10} # Search 10 clusters (faster than searching all)
    }
    
    results = collection.search(
        data=query_vector, 
        anns_field="embedding", 
        param=search_params, 
        limit=5, # Return top 5 results
        output_fields=["image_path"] # <--- CRITICAL: Return the filename
    )

    # 4. Format Results
    matches = []
    for hit in results[0]:
        matches.append({
            "score": hit.distance, # Lower is better for L2 distance
            "image_path": hit.entity.get("image_path")
        })
        
    return {"matches": matches}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)