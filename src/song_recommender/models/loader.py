import json
import numpy as np
from gensim.models import Word2Vec

from song_recommender.paths import (
    TAG_CLUSTER_MAP,
    TAG_KEYS,
    TAG_VECTORS,
    WORD2VEC_TAG_MODEL,
    VALID_TAGS,
    CLUSTER_TAGS
)

# tags

def load_word2vec_tag_model():
    return Word2Vec.load(str(WORD2VEC_TAG_MODEL))

def load_tag_cluster_map():
    with open(TAG_CLUSTER_MAP) as f:
        return json.load(f)

def load_tag_keys():
    with open(TAG_KEYS) as f:
        return json.load(f)

def load_valid_tags():
    with open(VALID_TAGS) as f:
        return set(json.load(f))

def load_tag_vectors():
    return np.load(TAG_VECTORS)

# loads dictionary {cluster name int : list of tags in cluster}
def load_cluster_tags():
    with open(CLUSTER_TAGS) as f:
        return json.load(f)