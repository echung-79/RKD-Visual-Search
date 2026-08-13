import ast
import os

import pandas as pd
from tqdm import tqdm

from embed import DATA_DIR, IMAGE_EMBEDDINGS_PATH, embed_text

TESTSET_PATH = os.path.join(DATA_DIR, 'testset.csv')

############################
### Pre Processing Logic ###
############################

def split(value):
    """Helper to tokenize dataframe values into lista"""
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

def load_image_vectors(path=IMAGE_EMBEDDINGS_PATH):
    """Reads image_embeddings.csv and links each SIGLIP vector to a priref"""
    embeddings_df = pd.read_csv(path, encoding = 'utf-8-sig')
    embeddings_df['embedding'] = embeddings_df['embedding'].apply(ast.literal_eval)
    return embeddings_df.groupby('priref')['embedding'].agg(list)

def preprocess(df):
    """Builds and runs inference on a natural language description of each record.
    These embeddings are joined with those from image_embeddings.csv into a dataframe"""
    df = df.copy()
    df['embedding_text'] = df.apply(_build_text, axis=1)
    image_vecs = df['priref'].map(load_image_vectors())
    df['image_vecs'] = image_vecs.apply(lambda v: v if isinstance(v, list) else [])
    df['description_vec'] = [
        embed_text(text)[0].tolist()
        for text in tqdm(df['embedding_text'], desc = "Embedding descriptions", unit = "rec")
    ]
    return df
