import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from song_recommender.data import TrackIndexer, load_png_resized, load_raw_resized

def spec_baseline_embedding(spec_path_list, config) -> np.ndarray:
    if config['baseline_embedding']['image_flag'] == True:
        chans = [load_png_resized(path, image_size=config['baseline_embedding']['image_size']) 
                 for path in spec_path_list]
    else:
        chans = [load_raw_resized(path, config['baseline_embedding']['image_size']) 
                 for path in spec_path_list]

    x = np.stack(chans, axis=0)  # (C, H, W)
    
    return x.reshape(-1)

def build_embeddings(df: pd.DataFrame, config) -> np.ndarray:

    indexer = TrackIndexer(df)

    n = len(df) if config['baseline_embedding']['max_songs'] is None else min(len(df), config['baseline_embedding']['max_songs'])

    emb_list = []

    for i, spotify_id in enumerate(df['spotify_id'].values):

        if config['baseline_embedding']['image_flag'] == True:
            spec_path_list = indexer.get_spec_png_paths(spotify_id)
        else:
            spec_path_list = indexer.get_spec_raw_paths(spotify_id)

        if i < n:
            if config['baseline_embedding']['log_every'] and i % config['baseline_embedding']['log_every'] == 0:
                print(f"embedding {i+1}/{n}")

            emb_list.append(spec_baseline_embedding(spec_path_list, config))
        else:
            break

    emb = np.stack(emb_list, axis=0)
    
    return normalize(emb, norm='l2')