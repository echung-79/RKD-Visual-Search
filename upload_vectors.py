import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
import pandas as pd
from encode_image import DATA_DIR
from preprocess import preprocess

load_dotenv()

# Use Cloud-Hosted Qdrant Cluster for Vector Store
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    cloud_inference=True
)


if __name__ == "__main__":
    rkd_df = pd.read_csv(os.path.join(DATA_DIR, 'adlib.csv'), encoding='utf-8-sig')
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
            'siglip_description': row.description_vec,
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
      collection_name="RKDTestSet4",
      points=points,
      batch_size=100,
      parallel=4
    )
    print('Uploaded')
