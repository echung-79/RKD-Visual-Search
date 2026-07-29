from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
import pandas as pd
from preprocess import preprocess

# Use Cloud-Hosted Qdrant Cluster for Vector Store
client = QdrantClient(
    url="https://389b72a9-0e48-4b10-a506-48eb90a8384e.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NmE5NWIzNDMtNTA1OC00YzU5LWI3ZmUtN2Q0NWFlMjRlNGM5In0.IcZaxps02ap0d-QxGkswAydyE3Y1aij8Rpfwm6mbMVQ",
    cloud_inference=True
)


if __name__ == "__main__":
    rkd_df = pd.read_csv('data/adlib.csv', encoding='utf-8-sig')
    input_features = ['title_en', 'attributions', 'genre_en', 'keywords_en', 'date_begin', 'date_end', 'objectcategorie_en']
    processed_df = preprocess(rkd_df[['priref'] + input_features])

    print('DF Processed')

    # points generator
    points = []
    for row in processed_df.itertuples():
        vector = {
            'description': Document(
                text = str(row.embedding_text),
                model = "sentence-transformers/all-MiniLM-L6-v2"),
            'title-sparse': Document(
                text = str(row.title_en),
                model = "Qdrant/bm25"),
        }
        if row.image_vecs:
            vector['image'] = row.image_vecs
        points.append(models.PointStruct(
            id = row.priref,
            vector = vector,
            payload = {
                col: getattr(row, col)
                for col in processed_df.columns
                if col in input_features},
        ))

    print('Points Generated')

    # upload points to collection in batch
    client.upload_points(
      collection_name="RKDTestSet3",
      points=points,
      batch_size=100,
      parallel=4
    )
    print('Uploaded')
