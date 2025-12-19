# service.py
import io
import torch
import time
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

class ShopSightService:
    def __init__(self, host="localhost", port="19530", collection_name="ShopSight_Inventory"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        
        # 1. Load AI Model (Heavy Operation - do this once!)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Service using device: {self.device}")
        
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(self.device)
        self.model.eval()

    def connect(self):
        """Connects to DB and loads Collection."""
        print(f"🔌 Connecting to Milvus at {self.host}...")
        retry_count = 0
        while retry_count < 10:
            try:
                connections.connect("default", host=self.host, port=self.port)
                break
            except Exception:
                print("⏳ Milvus not ready, retrying in 2s...")
                time.sleep(2)
                retry_count += 1
        
        # Check if collection exists; if not, create empty one
        if not utility.has_collection(self.collection_name):
            self._create_schema()
        
        self.collection = Collection(self.collection_name)
        
        # Ensure index exists
        if not self.collection.has_index():
            self._build_index()
            
        self.collection.load()
        print("✅ Service Ready: Collection Loaded.")

    def _create_schema(self):
        """Creates the table structure if missing."""
        print("🆕 Creating Schema...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        schema = CollectionSchema(fields, "ShopSight Inventory")
        Collection(self.collection_name, schema)

    def _build_index(self):
        print("🔨 Building Index...")
        idx_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        self.collection.create_index("embedding", idx_params)

    def get_embedding(self, image_bytes):
        """Helper: Converts raw bytes -> Vector (For API Search)"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Return numpy array
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None

    def search(self, image_bytes, top_k=5):
        """End-to-end search: Image Bytes -> List of Results"""
        query_vector = self.get_embedding(image_bytes)
        if query_vector is None: 
            return []

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=query_vector, 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k, 
            output_fields=["image_path"]
        )
        
        # Format results nicely
        matches = []
        for hit in results[0]:
            matches.append({
                "score": hit.distance,
                "image_path": hit.entity.get("image_path")
            })
        return matches