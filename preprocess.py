import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from encode_image import DATA_DIR, embed_text

IMAGE_EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'image_embeddings.npz')

############################
### Pre Processing Logic ###
############################

def split(value):
    """Helper to split up values within data frame into a list"""
    if pd.isna(value):
        return []
    seen, seen_lower = [], set()
    for part in str(value).split(';'):
        part = part.strip()
        if part and part.lower() not in seen_lower:
            seen.append(part)
            seen_lower.add(part.lower())
    return seen

def date_phrase(begin, end):
    return f"in {begin}" if begin == end else f"circa {begin}-{end}"

def _build_text(row):
    objectcategorie = split(row['objectcategorie_en'])
    genre = split(row['genre_en'])
    attributions = split(row['attributions'])
    keywords = split(row['keywords_en'])

    # Drop keywords already mentioned in genres
    genre_lower = {g.lower() for g in genre}
    keywords = [kw for kw in keywords if kw.lower() not in genre_lower]

    # Join lists into individual strings
    objectcategorie_str = ', '.join(objectcategorie) if objectcategorie else 'object'
    genre_str = ', '.join(genre) if genre else 'undetermined'
    attributions_str = '; '.join(attributions) if attributions else 'unknown artist'
    date_str = date_phrase(row['date_begin'], row['date_end'])
    keywords_str = ', '.join(keywords)

    text = f"{row['title_en']}. A {objectcategorie_str} ({genre_str}) by {attributions_str}, {date_str}."
    if keywords_str:
        text += f" Depicting {keywords_str}."
    return text

def _load_image_vecs(path):
    """Group per-image SigLIP2 embeddings by priref (a priref can have multiple images)."""
    data = np.load(path)
    vecs_by_priref = {}
    for priref, vec in zip(data['priref'], data['embeddings']):
        vecs_by_priref.setdefault(priref, []).append(vec.tolist())
    return vecs_by_priref


def preprocess(df, image_embeddings_path=IMAGE_EMBEDDINGS_PATH):
    """Builds a natural language document and attaches SigLIP2 image + description vectors for each record"""
    df = df.copy()
    df['embedding_text'] = df.apply(_build_text, axis=1)
    vecs_by_priref = _load_image_vecs(image_embeddings_path)
    df['image_vecs'] = df['priref'].map(lambda p: vecs_by_priref.get(p, []))
    df['description_vec'] = [
        embed_text(text)[0].tolist()
        for text in tqdm(df['embedding_text'], desc="Embedding descriptions", unit="rec")
    ]
    return df
