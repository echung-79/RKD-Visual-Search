"""
Detects segments in RKD artwork images with YOLO, embeds each segment with
DINOv2, and uploads the resulting vectors to the "Segments" collection in
Qdrant.
"""

# Imports
import os
import re
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
import cv2
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

# Constants
load_dotenv()
IIIF_MAX_SIDE = 512
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEGMENT_DIR = os.path.join(DATA_DIR, "segments")
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
CONF_LIMIT = 0.2
OBJECT_CLASSES = [0]
ALLOWED_GENRES = ['portrait']
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')

os.makedirs(SEGMENT_DIR, exist_ok=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

class ImageFetchError(Exception):
    """Raised when an IIIF image URL can't be downloaded or decoded."""

# Load models
yolo = YOLO("yolo26x-seg.pt")  # Object Detection Model

# load the model and processor
MODEL_ID = "facebook/dinov2-with-registers-large" # "google/siglip2-base-patch16-224"
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL_ID).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Image fetching logic
def adjust_iiif_size(url: str, max_side: int = IIIF_MAX_SIDE) -> str:
    """
    Adjusts iiif url to bound it's size when requested
    """
    parts = url.split('/')
    if '/iiif/' not in url or len(parts) < 4:
        return url
    parts[-3] = f'!{max_side},{max_side}'
    return '/'.join(parts)

def fetch_image(url: str) -> Image.Image:
    """
    Downloads and preprocesses an image from its IIIF URL, requesting it
    pre-scaled to at most IIIF_MAX_SIDE pixels per side.
    """
    try:
        response = requests.get(adjust_iiif_size(url), headers = REQUEST_HEADERS, timeout = 30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except (requests.RequestException, OSError) as e:
        raise ImageFetchError(f"Invalid image url: {url}") from e

    if max(image.size) > IIIF_MAX_SIDE:
        image.thumbnail((IIIF_MAX_SIDE, IIIF_MAX_SIDE), Image.LANCZOS)

    return image

# Image embedding logic
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray"""
    norms = np.linalg.norm(x, axis = -1, keepdims = True)
    norms[norms == 0] = 1.0
    return (x / norms).astype('float32')

@torch.no_grad()
def embed_images(images: list) -> np.ndarray:
    """Runs a list of Images through our processor and model"""
    inputs = processor(images = images, return_tensors = "pt").to(device)

    # DinoV2 inference
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0].cpu().numpy()

    # Siglip inference:
    # feats = model.get_image_features(**inputs).pooler_output
    # vector = feats.cpu().numpy()

    return l2_normalize(vector)

def sanitize_filename(name: str) -> str:
    """Strips characters that aren't safe to use in a file name"""
    return INVALID_FILENAME_CHARS.sub('_', name)

def parse_year(value) -> int:
    """Extracts the leading year from a date value like 1624, '1624', or '1636-07-03'"""
    return int(str(value).split('-')[0])

# Save masked images
def save_mask(file_name, mask, dest, source):
    """
    Saves masked images and returns a preprocessed image for embedding
    """
    h, w = source.shape[:2]

    # masks can come back at model resolution; align to the source image
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop = source[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    alpha = (mask_crop * 255).astype(np.uint8)
    rgba = np.dstack([crop, alpha])

    out_path = os.path.join(dest, file_name)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(out_path, bgra)

    embed_source = crop.copy()
    embed_source[~mask_crop] = 255  # neutral background outside the mask
    embedding_image = Image.fromarray(embed_source)

    return embedding_image

if __name__ == "__main__":
    # Connect to Qdrant Cluster
    client = QdrantClient(
        url = os.environ["QDRANT_URL"],
        api_key = os.environ["QDRANT_API_KEY"],
        cloud_inference = True
    )
    # Needed so queries can group results by record (see query_collection.py)
    client.create_payload_index(collection_name = os.environ['COLLECTION_NAME'],
                                field_name = 'Priref', field_schema = 'integer')
    # Read and parse through our CSV
    records_df = pd.read_csv(os.path.join(DATA_DIR, os.environ['CSV_NAME']), encoding='utf-8-sig')
    masks_saved = 0
    progress = tqdm(records_df.itertuples(), total = len(records_df), desc="Indexing records")
    for row in progress:
        # Filtering records by genre
        genres = [g.strip() for g in row.genre_en.split(';')] if pd.notna(row.genre_en) else []
        if ALLOWED_GENRES and not any(g in ALLOWED_GENRES for g in genres):
            continue
        # Skip if no IIIF URLS
        if pd.isna(row.media_reference_numbers):
            continue
        # Record metadata
        priref = int(row.priref)
        title = row.title_en if pd.notna(row.title_en) else f"Untitled_{priref}"
        safe_title = sanitize_filename(title)
        urls = [u.strip() for u in row.media_reference_numbers.split(';') if u.strip()]

        seg_index = 0
        for url in urls:
            try:
                image = fetch_image(url)
            except ImageFetchError:
                continue
            results = yolo.predict(image, conf = CONF_LIMIT, classes = OBJECT_CLASSES)[0]

            if results.masks is None:
                continue

            masks = results.masks.data.cpu().numpy().astype(bool)
            confs = results.boxes.conf.cpu().numpy()
            source = np.array(image)

            images = []
            seg_ids = []
            seg_confs = []
            for mask, conf in zip(masks, confs):
                seg_name = f"{safe_title}.{seg_index}.png"
                result = save_mask(seg_name, mask, SEGMENT_DIR, source)
                if result is None:
                    continue
                images.append(result)
                seg_ids.append(seg_index)
                seg_confs.append(conf)
                seg_index += 1
                masks_saved += 1
                progress.set_postfix(masks_saved = masks_saved)

            if not images:
                continue

            vectors = embed_images(images)  # batched inference

            points = [
                PointStruct(
                    id = priref * 1000 + seg_id,
                    vector = {"DinoV2": vector},
                    payload = {"Origin": title,
                            "Segment": seg_id,
                            "Attributions": row.attributions,
                            "Priref": row.priref,
                            "Date": float((parse_year(row.date_begin) + parse_year(row.date_end)) / 2),
                            "Confidence": float(conf)})
                for seg_id, vector, conf in zip(seg_ids, vectors, seg_confs)
            ]

            client.upload_points(collection_name = os.environ['COLLECTION_NAME'],
                            points = points)
