import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance

load_dotenv()

client = QdrantClient(
    url = os.environ["QDRANT_URL"],
    api_key = os.environ["QDRANT_API_KEY"],
    cloud_inference=True
)

client.create_collection(
    collection_name = "Segments",
    vectors_config={
        "SIGLIP" : models.VectorParams(
            size = 768,
            distance = Distance.COSINE)
            }
)
