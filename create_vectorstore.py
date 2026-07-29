from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance

client = QdrantClient(
    url="https://389b72a9-0e48-4b10-a506-48eb90a8384e.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NmE5NWIzNDMtNTA1OC00YzU5LWI3ZmUtN2Q0NWFlMjRlNGM5In0.IcZaxps02ap0d-QxGkswAydyE3Y1aij8Rpfwm6mbMVQ",
    cloud_inference=True
)

client.create_collection(
    collection_name = "RKDTestSet3",
    vectors_config={
        "description" : models.VectorParams(
            size = 384,
            distance = Distance.COSINE),
        "image" : models.VectorParams(
            size = 768,
            distance = models.Distance.COSINE,
            multivector_config = models.MultiVectorConfig(
                comparator = models.MultiVectorComparator.MAX_SIM))
            },
    sparse_vectors_config={"title-sparse": models.SparseVectorParams()}
)
