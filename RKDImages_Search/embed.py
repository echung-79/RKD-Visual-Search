# Imports
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm

import os
from io import BytesIO

from transformers import AutoModel, AutoProcessor

# Constants
MAX_SIDE = 224
IIIF_MAX_SIDE = 224
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'dino_image_embeddings.csv')
FAILED_URLS_PATH = os.path.join(DATA_DIR, 'failed_urls.txt')
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
MAX_WORKERS = 16
BATCH_SIZE = 32

# load the model and processor
MODEL_ID = "facebook/dinov2-with-registers-large" #"google/siglip2-base-patch16-224"
device = 'cuda' if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(MODEL_ID).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# load the model and processor
TEXT_MODEL_ID = "google/siglip2-base-patch16-224"
text_model = AutoModel.from_pretrained(TEXT_MODEL_ID).to(device)
text_processor = AutoProcessor.from_pretrained(TEXT_MODEL_ID)

# Helper classes and functions
class ImageFetchError(Exception):
    """Raised when an IIIF image URL can't be downloaded or decoded."""

def load_failed_urls(path: str = FAILED_URLS_PATH) -> set:
    """Loads the set of previously failed image URLs, if any have been recorded."""
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def save_failed_urls(urls: set, path: str = FAILED_URLS_PATH) -> None:
    """Persists the set of failed image URLs, one per line, sorted for stable diffs."""
    with open(path, 'w', encoding='utf-8') as f:
        for url in sorted(urls):
            f.write(url + '\n')

def _bound_iiif_size(url: str, max_side: int = IIIF_MAX_SIDE) -> str:
    """
    Adjusts iiif url to bind it's size when requesting images
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
        response = requests.get(_bound_iiif_size(url), headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except (requests.RequestException, OSError) as e:
        raise ImageFetchError(f"Invalid image url: {url}") from e

    if max(image.size) > IIIF_MAX_SIDE:
        image.thumbnail((IIIF_MAX_SIDE, IIIF_MAX_SIDE), Image.LANCZOS)
    return image

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray"""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype('float32')

@torch.no_grad()
def embed_image(path: str):
    """
    Given an image path an image is loaded, preprocessed,
    and embedded using a locally run dinov2 model
    """
    # Load Image from Path
    try:
        image = Image.open(path)
    except OSError as e:
        raise FileNotFoundError(f"Invalid image path: {path}") from e

    # Convert to RGB and downscale if necessary
    image = image.convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    inputs = processor(images = image, return_tensors = "pt").to(device)
    # DinoV2 inference:
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0].cpu().numpy()
    # Siglip inference:
    # feats = model.get_image_features(**inputs).pooler_output
    # vector = feats.cpu().numpy()
    return l2_normalize(vector)

@torch.no_grad()
def embed_text(x: str):
    """
    Given a string, embeds it using
    a locally run siglip2 model
    """
    inputs = text_processor(text = x, return_tensors = "pt").to(device)
    feats = text_model.get_text_features(**inputs).pooler_output
    vector = feats.cpu().numpy()
    return l2_normalize(vector)

@torch.no_grad()
def embed_images_batch(images: list) -> np.ndarray:
    """Runs a batch of already-downloaded images through siglip2."""
    inputs = processor(images = images, return_tensors = "pt").to(device)
    # DinoV2 inference
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0].cpu().numpy()
    # Siglip inference:
    # feats = model.get_image_features(**inputs).pooler_output
    # vector = feats.cpu().numpy()
    return l2_normalize(vector)

def embed_images_parallel(rows, max_workers = MAX_WORKERS, batch_size = BATCH_SIZE):
    """
    Given (priref, url) rows, downloads images concurrently across threads
    while the main thread batches completed downloads and runs them through the model.
    """
    embeddings, good_prirefs = [], []
    failed_urls = []
    batch_images, batch_prirefs = [], []

    def flush_batch():
        nonlocal batch_images, batch_prirefs
        if not batch_images:
            return
        embeddings.append(embed_images_batch(batch_images))
        good_prirefs.extend(batch_prirefs)
        batch_images, batch_prirefs = [], []

    progress = tqdm(total = len(rows), desc = "Embedding images", unit = "img")

    with ThreadPoolExecutor(max_workers = max_workers) as pool:
        future_to_row = {pool.submit(fetch_image, url): (priref, url) for priref, url in rows}
        for future in as_completed(future_to_row):
            priref, url = future_to_row[future]
            try:
                image = future.result()
            except ImageFetchError as e:
                failed_urls.append(url)
                tqdm.write(f"Skipping unreachable image: {e}")
            else:
                batch_images.append(image)
                batch_prirefs.append(priref)
                if len(batch_images) >= batch_size:
                    flush_batch()

            progress.set_postfix(ok=len(good_prirefs) + len(batch_prirefs), failed=len(failed_urls))
            progress.update(1)

    flush_batch()
    progress.close()

    return embeddings, good_prirefs, failed_urls


if __name__ == "__main__":
    rkd_df = pd.read_csv(os.path.join(DATA_DIR, 'adlib_with_iiif.csv'), encoding='utf-8-sig')
    # Each priref can be associated with multiple IIIF urls
    all_rows = [
        (priref, url.strip())
        for priref, urls in zip(rkd_df['priref'], rkd_df['media_reference_numbers'])
        if isinstance(urls, str) and urls
        for url in urls.split(';')
        if url.strip()
    ]

    if not all_rows:
        raise SystemExit("No media_reference_numbers values found in adlib_with_iiif.csv — nothing to embed")

    known_failed_urls = load_failed_urls()
    rows = [(priref, url) for priref, url in all_rows if url not in known_failed_urls]
    skipped = len(all_rows) - len(rows)
    if skipped:
        print(f"Skipping {skipped} previously failed URLs (see {FAILED_URLS_PATH})")

    if not rows:
        raise SystemExit("All URLs have previously failed — nothing to embed")

    embeddings, priref_list, failed_urls = embed_images_parallel(rows)

    if failed_urls:
        save_failed_urls(known_failed_urls | set(failed_urls))
        print(f"Recorded {len(failed_urls)} newly failed URLs to {FAILED_URLS_PATH}")

    if not embeddings:
        raise SystemExit("All images failed to download — nothing to embed")

    embeddings = np.concatenate(embeddings, axis=0)

    # CSV with a row for each vector and the priref of the record it belongs to
    embeddings_df = pd.DataFrame({
        'priref': priref_list,
        'embedding': [embedding.tolist() for embedding in embeddings],
    })

    embeddings_df.to_csv(IMAGE_EMBEDDINGS_PATH, index=False, encoding='utf-8-sig')
    print(f"Saved {len(embeddings_df)} image embeddings to {IMAGE_EMBEDDINGS_PATH}")
