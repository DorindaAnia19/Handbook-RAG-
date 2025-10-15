import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
COLLECTION_NAME = "handbook"
PDF_PATH = "handbook.pdf"