import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import numpy as np
import os

# 1. SETUP: Load the Pretrained Vision Transformer
# We use the 'base' model with 16x16 patches, pretrained on ImageNet-21k
MODEL_NAME = 'google/vit-base-patch16-224-in21k'

print(f"Loading {MODEL_NAME} with PyTorch...")

# Check for GPU (CUDA) or MPS (Mac), otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Processor handles resizing (224x224) and normalization
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Load the model and move it to the correct device (GPU/CPU)
model = ViTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()  # Set model to evaluation mode (disables dropout/training specific layers)

def extract_features(image_path):
    """
    Takes an image path, preprocesses it, and runs it through the ViT.
    Returns: A numpy array of shape (768,)
    """
    try:
        # 2. PREPROCESS: Load and format the image
        img = Image.open(image_path).convert("RGB")
        
        # return_tensors="pt" gives us PyTorch tensors
        inputs = processor(images=img, return_tensors="pt")
        
        # Move inputs to the same device as the model (GPU or CPU)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 3. INFERENCE: Run the model
        # torch.no_grad() tells PyTorch not to calculate gradients (saves RAM and speed)
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. EXTRACTION: Get the "CLS" token
        # output shape: (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.last_hidden_state
        
        # [0, 0, :] -> First batch, First token (CLS), All 768 features
        feature_vector = last_hidden_states[0, 0, :]
        
        # Move back to CPU and convert to numpy for storage
        return feature_vector.cpu().numpy()

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- TEST RUN ---
if __name__ == "__main__":
    # Test with a dummy image if you don't have local data yet
    test_image_path = "../data/images/1163.jpg"
    
    if not os.path.exists(test_image_path):
        print("Downloading dummy image for testing...")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        Image.open(requests.get(url, stream=True).raw).save(test_image_path)

    print(f"Extracting features from {test_image_path}...")
    vector = extract_features(test_image_path)
    
    if vector is not None:
        print("\n--- SUCCESS ---")
        print(f"Vector Shape: {vector.shape}")
        print(f"First 10 values: {vector[:10]}")
        
        # Sanity Check
        assert vector.shape == (768,), "Error: Output vector has wrong dimensions!"