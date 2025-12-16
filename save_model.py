import tensorflow as tf
import os
import shutil

# Define the export path with a version number '1'
# Structure will be: ./models/resnet/1/
MODEL_DIR = "models/resnet"
VERSION = "1"
EXPORT_PATH = os.path.join(MODEL_DIR, VERSION)

# Clean up previous runs if they exist
if os.path.exists(EXPORT_PATH):
    shutil.rmtree(EXPORT_PATH)

print("Loading pre-trained ResNet50 model...")
model = tf.keras.applications.ResNet50(weights='imagenet')

print(f"Exporting model to {EXPORT_PATH}...")
# specific 'tf' format is crucial for TensorFlow Serving
model.save(EXPORT_PATH, save_format='tf') 

print("✅ Model exported successfully.")