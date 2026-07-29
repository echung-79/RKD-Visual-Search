import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# load the model and processor
MODEL_ID = "google/siglip2-base-patch16-224"
device = 'cuda' if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(MODEL_ID).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)

IMAGES_DIR = 'data/Test Set'
MAX_IMAGE_SIDE = 1024  # cap source resolution before the processor's own resize to 224x224


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype('float32')


def preprocess_image(im: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Convert to RGB and downscale to a standard resolution cap.
    Source scans (.jp2) can be far larger than the model needs, so this keeps
    decoding/resizing cheap and memory bounded before the processor resizes to 224x224."""
    im = im.convert('RGB')
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im

@torch.no_grad()
def embed_images(images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
    """One L2-normalized vector per image"""
    chunks = []
    with tqdm(total=len(images), desc="Embedding images", unit="img") as pbar:
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs).pooler_output
            chunks.append(feats.cpu().numpy())
            pbar.update(len(batch))
    return l2_normalize(np.concatenate(chunks, axis=0))

def search(query_vec: np.ndarray, index: np.ndarray, k: int = 5):
    """Cosine similarity"""
    scores = index @ query_vec        # (N,) similarity to every record
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]

def main():
    rkd_df = pd.read_csv('data/adlib.csv', encoding='utf-8-sig')
    # one record can have multiple images, with filepaths given by media_reference_numbers
    # images are cached locally as data/images/{filename}; skip ones we don't have
    priref_list, filename_list, image_paths = [], [], []
    for priref, refs in zip(rkd_df['priref'], rkd_df['media_reference_numbers']):
        if not isinstance(refs, str) or not refs:
            continue
        for filename in refs.split('; '):
            path = os.path.join(IMAGES_DIR, filename)
            if os.path.exists(path):
                priref_list.append(priref)
                filename_list.append(filename)
                image_paths.append(path)

    if not image_paths:
        raise SystemExit(f"No local images found under {IMAGES_DIR}/ — nothing to embed")

    images, good_prirefs, good_filenames = [], [], []
    for priref, filename, path in tqdm(
        zip(priref_list, filename_list, image_paths),
        total=len(image_paths), desc="Preprocessing images", unit="img",
    ):
        try:
            images.append(preprocess_image(Image.open(path)))
        except OSError as e:
            print(f"Skipping unreadable image {path}: {e}")
            continue
        good_prirefs.append(priref)
        good_filenames.append(filename)
    priref_list, filename_list = good_prirefs, good_filenames

    if not images:
        raise SystemExit("All local images failed to load — nothing to embed")

    embeddings = embed_images(images)

    ############################
    ### Save Embeddings ###
    ############################

    # one row per image (not per priref) — a priref with N cached images
    # gets N rows here, all sharing the same priref value
    output_path = 'data/image_embeddings.npz'
    np.savez(
        output_path,
        priref=np.array(priref_list),
        filename=np.array(filename_list),
        embeddings=embeddings,
    )
    print(f"Saved {len(embeddings)} image embeddings ({len(set(priref_list))} prirefs) to {output_path}")


if __name__ == '__main__':
    main()
