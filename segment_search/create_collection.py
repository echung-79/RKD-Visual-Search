import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance

load_dotenv()

# Run to create collection on Qdrant Cluster

client = QdrantClient(
    url = os.environ["QDRANT_URL"],
    api_key = os.environ["QDRANT_API_KEY"],
    cloud_inference=True
)

client.create_collection(
    collection_name = os.environ["COLLECTION_NAME"],
    vectors_config={
        "DinoV2" : models.VectorParams(
            size = 1024,
            distance = Distance.COSINE)
            }
)
