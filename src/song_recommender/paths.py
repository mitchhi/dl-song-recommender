from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[2]

# artifacts directory (gitignored)
ARTIFACTS_DIR = ROOT / 'artifacts'

# audio directories
AUDIO_DIR = ARTIFACTS_DIR / 'audio'
STEMS_DIR = ARTIFACTS_DIR / 'stems'

# spectrogram directories
SPECTROGRAM_PNG_DIR = ARTIFACTS_DIR / 'spectrograms_png'
SPECTROGRAM_RAW_DIR = ARTIFACTS_DIR / 'spectrograms_raw'

# metadata directories
DATA_DIR = ROOT / 'data'
SPLITS_DIR = DATA_DIR / 'splits'

# configs
CONFIGS_DIR = ROOT / 'configs'

# models
MODELS_DIR = ROOT / 'src/song_recommender/models'

## tags
TAG_CLUSTER_DIR = MODELS_DIR / 'tag_clusters'
TAG_EMBEDDING_DIR = MODELS_DIR / 'tag_embeddings'

VALID_TAGS = TAG_EMBEDDING_DIR / 'valid_tags.json'
WORD2VEC_TAG_MODEL = TAG_EMBEDDING_DIR / 'word2vec_tags.model'
TAG_KEYS = TAG_EMBEDDING_DIR / "tag_keys.json"
TAG_VECTORS = TAG_EMBEDDING_DIR / "tag_vectors.npy"
TAG_CLUSTER_MAP = TAG_CLUSTER_DIR / 'tag_cluster_map.json'

## baseline

BASELINE_DIR = MODELS_DIR / 'baselines'

TAG_CENTROID_BASELINE = BASELINE_DIR / 'tag_centroid_baseline.npy'
SPEC_SIM_BASELINE = BASELINE_DIR / 'spec_sim_baseline.npz'
RANDOM_RETRIEVAL_BASELINE = BASELINE_DIR / 'random_retrieval_baseline.npy'