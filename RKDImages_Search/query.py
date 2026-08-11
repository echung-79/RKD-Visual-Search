import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Document

from embed import embed_text

load_dotenv()

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    cloud_inference=True
)

COLLECTION = "RKDTestSet4"

# Every scoring method available on the collection, keyed by an id used
# throughout the app, mapped to a human-readable label.

METHODS = {
    "keyword": "Keyword (BM25 on title)",
    "conceptual": "Conceptual (description text)",
    "siglip_text": "SigLIP description",
    "visual": "Visual (image content)",
}


def _method_query(method, text, siglip_vec):
    """Returns the (query, using) pair Qdrant needs for a given scoring method."""
    if method == "keyword":
        return Document(text=text, model="Qdrant/bm25"), "title-sparse"
    if method == "conceptual":
        return Document(text=text, model="sentence-transformers/all-MiniLM-L6-v2"), "description"
    if method == "siglip_text":
        return siglip_vec, "siglip_description"
    if method == "visual":
        # Text and images share the same SigLIP2 embedding space, so the text
        # vector can be compared directly against each record's image vectors.
        return siglip_vec, "image"
    raise ValueError(f"Unknown method: {method}")


def _needs_siglip_vec(methods):
    return any(m in ("siglip_text", "visual") for m in methods)


def single_method_search(method, text, limit=10, siglip_vec=None):
    """Runs one scoring method in isolation and returns its ranked points."""
    if method in ("siglip_text", "visual") and siglip_vec is None:
        siglip_vec = embed_text(text)[0].tolist()
    query, using = _method_query(method, text, siglip_vec)
    result = client.query_points(
        collection_name=COLLECTION,
        query=query,
        using=using,
        limit=limit,
        with_payload=True,
    )
    return result.points


def composite_search(text, methods=None, limit=10, siglip_vec=None):
    """Fuses the given methods with Qdrant's built-in Reciprocal Rank Fusion."""
    methods = methods or list(METHODS)
    if _needs_siglip_vec(methods) and siglip_vec is None:
        siglip_vec = embed_text(text)[0].tolist()

    prefetch = []
    for method in methods:
        query, using = _method_query(method, text, siglip_vec)
        prefetch.append(models.Prefetch(query=query, using=using, limit=limit))

    result = client.query_points(
        collection_name=COLLECTION,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit,
    )
    return result.points


def text_query(x, limit=10):
    """CLI helper: runs every method individually plus the RRF composite."""
    siglip_vec = embed_text(x)[0].tolist()
    results = {
        method: single_method_search(method, x, limit=limit, siglip_vec=siglip_vec)
        for method in METHODS
    }
    results["composite"] = composite_search(x, list(METHODS), limit=limit, siglip_vec=siglip_vec)
    return results


if __name__ == "__main__":
    query = input("Input your query: ")
    for method, points in text_query(query).items():
        label = METHODS.get(method, "Composite (RRF)")
        print(f"\n=== {label} ===")
        for p in points:
            print(p.id, round(p.score, 4), p.payload.get("title_en"))
