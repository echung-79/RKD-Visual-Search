import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance

load_dotenv()

client = QdrantClient(
    url = os.environ["QDRANT_URL"],
    api_key = os.environ["QDRANT_API_KEY"],
    cloud_inference = True
)

client.create_collection(
    collection_name = os.environ["COLLECTION_NAME"],
    vectors_config={
        "description" : models.VectorParams(
            size = 384,
            distance = Distance.COSINE),
        "image" : models.VectorParams(
            size = 1024,
            distance = models.Distance.COSINE,
            multivector_config = models.MultiVectorConfig(
                comparator = models.MultiVectorComparator.MAX_SIM)),
        "siglip_description" : models.VectorParams(
            size = 768,
            distance = Distance.COSINE)
            },
    sparse_vectors_config={"title-sparse": models.SparseVectorParams()}
)
