# 🛍️ ShopSight: AI-Powered Visual Search Engine

**ShopSight** is a visual search engine designed to revolutionize e-commerce product discovery. Instead of relying on text keywords, users can upload an image (e.g., a specific shoe or shirt) to find visually similar items from a product inventory.

Powered by Vision Transformers (ViT) and Milvus Vector Database, ShopSight converts images into high-dimensional vector embeddings to perform lightning-fast similarity searches.

## 🚀 Features

* 🔍 Visual Similarity Search: Find products based on visual features (color, pattern, shape) rather than metadata.
* 🧠 Advanced AI Models: Utilizes Google's Vision Transformer (ViT-base-patch16-224) for state-of-the-art feature extraction.
* ⚡ High-Performance Indexing: Uses Milvus for scalable, millisecond-latency vector retrieval.
* 🐳 Microservices Architecture: Fully containerized with Docker for easy deployment and scaling.
* 🔌 RESTful API: Robust FastAPI backend with auto-generated documentation.
* 💻 Interactive UI: Clean, responsive frontend for testing and demonstration.

## 🛠️ Tech Stack

Core Services
* Backend: Python, FastAPI, Uvicorn
* ML/AI: PyTorch, HuggingFace Transformers (ViT)
* Database: Milvus (Vector DB), etcd, MinIO
* Frontend: HTML5, Bootstrap 5, Nginx

Infrastructure
* Containerization: Docker & Docker Compose
* Data Processing: NumPy, Pillow

## 🏗️ Architecture

The system consists of four main containerized services:
1. Frontend (Nginx): Serves the web UI and product images.
2. Backend (FastAPI): Handles API requests, runs the AI inference model, and manages data ingestion.
3. Milvus (Vector DB): Stores and indexes image embeddings for similarity search.
4. MinIO & etcd: Storage and metadata management dependencies for Milvus.
