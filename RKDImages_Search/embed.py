import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MAX_SIDE = 1024

# load the model and processor
MODEL_ID = "google/siglip2-base-patch16-224"
device = 'cuda' if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(MODEL_ID).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'Test Set')


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype('float32')


@torch.no_grad()
def embed_image(path: str):
    """
    Given an image path an image is loaded, preprocessed,
    and embedded using a locally run siglip2 model
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

    inputs = processor(image, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs).pooler_output
    vector = feats.cpu().numpy()
    return l2_normalize(vector)

@torch.no_grad()
def embed_text(x: str):
    """
    Given a string, embeds it using
    a locally run siglip2 model
    """
    inputs = processor(text = x, return_tensors = "pt").to(device)
    feats = model.get_text_features(**inputs).pooler_output
    vector = feats.cpu().numpy()
    return l2_normalize(vector)

if __name__ == "__main__":
    rkd_df = pd.read_csv(os.path.join(DATA_DIR, 'adlib.csv'), encoding='utf-8-sig')
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

    embeddings, good_prirefs, good_filenames = [], [], []
    for priref, filename, path in tqdm(
        zip(priref_list, filename_list, image_paths),
        total=len(image_paths), desc="Embedding images", unit="img",
    ):
        try:
            embeddings.append(embed_image(path))
        except FileNotFoundError as e:
            print(f"Skipping unreadable image {path}: {e}")
            continue
        good_prirefs.append(priref)
        good_filenames.append(filename)
    priref_list, filename_list = good_prirefs, good_filenames

    if not embeddings:
        raise SystemExit("All local images failed to load — nothing to embed")

    embeddings = np.concatenate(embeddings, axis=0)

    ############################
    ### Save Embeddings ###
    ############################

    # one row per image (not per priref) — a priref with N cached images
    # gets N rows here, all sharing the same priref value
    output_path = os.path.join(DATA_DIR, 'image_embeddings.npz')
    np.savez(
        output_path,
        priref=np.array(priref_list),
        filename=np.array(filename_list),
        embeddings=embeddings,
    )
    print(f"Saved {len(embeddings)} image embeddings ({len(set(priref_list))} prirefs) to {output_path}")
