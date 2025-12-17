import glob
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# --- CONFIGURATION ---
BATCH_SIZE = 64  # PyTorch is often more memory efficient, so we can bump this up
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
COLLECTION_NAME = "ShopSight_Inventory"
IMAGE_FOLDER = "../data/images/*.jpg"
LIMIT_DATASET = 4400 

# --- 1. SETUP DEVICE & MODEL ---
# Automatically detect the fastest available hardware
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Model...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME).to(device)
model.eval() # Set to evaluation mode (disables dropout, etc.)

def process_batch(image_paths):
    """
    Loads and processes a batch of images using PyTorch.
    """
    images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            
    if not images:
        return None, []

    # Preprocess: return_tensors="pt" gives us PyTorch tensors
    inputs = processor(images=images, return_tensors="pt")
    
    # Move inputs to the GPU/Device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference (No Gradients needed = Faster + Less RAM)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract [CLS] token (index 0)
    # .cpu() moves data back to RAM, .numpy() converts to array
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embeddings, valid_paths

# --- 2. MILVUS SETUP ---
print("Connecting to Milvus...")
connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, "Product Image Embeddings")

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

collection = Collection(COLLECTION_NAME, schema)

# --- 3. MAIN LOOP ---
all_images = glob.glob(IMAGE_FOLDER)
if LIMIT_DATASET:
    all_images = all_images[:LIMIT_DATASET]

print(f"Starting ingestion for {len(all_images)} images on {device}...")

for i in range(0, len(all_images), BATCH_SIZE):
    batch_paths = all_images[i : i + BATCH_SIZE]
    
    embeddings, valid_paths = process_batch(batch_paths)
    
    if embeddings is not None:
        data_to_insert = [
            valid_paths,
            embeddings
        ]
        collection.insert(data_to_insert)
        
    print(f"Processed {i + len(batch_paths)}/{len(all_images)}")

# --- 4. BUILD INDEX ---
print("Building Index (IVF_FLAT)...")
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

print("Ingestion Complete!")