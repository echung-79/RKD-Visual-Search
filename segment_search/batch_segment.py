import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

import torch
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from itertools import batched

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "adlib.csv"
IMAGE_DIR = DATA_DIR / "Test Set"

COLLECTION_NAME = "RKDSegments"

MAX_DIM = 640
yolo = YOLO("yolo26x-seg.pt")

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    cloud_inference=True
)

# load the model and processor
MODEL_ID = "google/siglip2-base-patch16-224"
device = 'cuda' if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(MODEL_ID).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def preprocess(path):
    """ Loads image from path in appropriate size"""
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = MAX_DIM / max(image.shape[:2])
    if scale < 1:
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype('float32')

if __name__ == "__main__":
    adlib_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    portraits_df = adlib_df[adlib_df["genre_en"] == "portrait"]

    records = [
        {
            "id": row.priref,
            "paths": [
                IMAGE_DIR / filename.strip()
                for filename in str(row.media_reference_numbers).split(";")
                if filename.strip()],
            "attributions" : row.attributions
        }
        for row in portraits_df.itertuples()
    ]

    progress = tqdm(records, desc="Segmenting & embedding", unit="record")
    
    for record in progress:
        progress.set_postfix(priref=record["id"], refresh=False)

        images = [preprocess(path) for path in record["paths"]]
        results = yolo.predict(images, conf=0.3, classes=[0])

        segments = []
        for image, result in zip(images, results):
            if result.masks is None:
                continue
            for mask in result.masks.data:
                # crop the image to the mask
                mask_np = mask.cpu().numpy().astype("uint8")
                mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

                ys, xs = np.where(mask_np)
                if len(xs) == 0:
                    continue
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()

                crop = image[y0:y1 + 1, x0:x1 + 1].copy()
                crop_mask = mask_np[y0:y1 + 1, x0:x1 + 1]
                crop[~crop_mask] = 255
                segments.append(crop)

        if not segments:
            continue

        segment_paths = []
        for i, segment in enumerate(segments):
            out_path = f"segments/{record['id']}_segment{i}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
            segment_paths.append(out_path)

        pil_images = [Image.fromarray(segment) for segment in segments]
        inputs = processor(pil_images)
        feats = model.get_image_features(**inputs).pooler_output
        vectors = feats.cpu().numpy()
        normalized_vectors = [l2_normalize(vector) for vector in vectors]

        # points generator
        points = [
            models.PointStruct(
                id = f"{record["id"]}.{i}",
                vector={"SIGLIP": vector.tolist()},
                payload={
                    "priref": record["id"],
                    "segment_index": i,
                    "segment_path": segment_path,
                },
            )
            for i, (vector, segment_path) in enumerate(zip(normalized_vectors, segment_paths))
        ]

        # upload points to collection in batch
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
            batch_size=100,
            parallel=4
        )
        progress.set_postfix(priref=record["id"], segments=len(segments), refresh=False)
