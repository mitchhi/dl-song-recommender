import numpy as np
import torch

def precision_at_k(relevance):

    if len(relevance) == 0:
        return 0.0
    return np.sum(relevance) / len(relevance)

def recall_at_k(relevance, total_relevant):

    if total_relevant == 0:
        return 0.0
    return np.sum(relevance) / total_relevant

def average_precision_at_k(relevance):

    hits = 0
    score = 0.0

    for i, r in enumerate(relevance):
        if r:
            hits += 1
            score += hits / (i + 1)

    if hits == 0:
        return 0.0

    return score / hits

def ndcg_at_k(relevance):

    relevance = np.asarray(relevance)

    discounts = 1 / np.log2(np.arange(2, len(relevance) + 2))

    dcg = np.sum(relevance * discounts)

    ideal = np.sort(relevance)[::-1]

    idcg = np.sum(ideal * discounts)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def jaccard_similarity(set_a, set_b):

    union = set_a.union(set_b)

    if not union:
        return 0.0

    return len(set_a.intersection(set_b)) / len(union)

def dominant_cluster_accuracy_at_k(query_dom, neighbor_doms):

    # query_dom: dominant_cluster of the query track
    # neighbor_doms: list of dominant_cluster for retrieved tracks

    if query_dom is None or len(neighbor_doms) == 0:
        return 0.0

    neighbor_doms = np.array(neighbor_doms)

    return np.mean(neighbor_doms == query_dom)

# ---------------------------------------------------------------------------
# Diversity metrics (no genre/tag signal)
# ---------------------------------------------------------------------------

def artist_diversity_at_k(neighbor_artists):
    """Fraction of unique artists in the top-k recommendations.

    Range: [1/k, 1]. Higher = more artist variety.

    Args:
        neighbor_artists: array-like of artist name strings for each retrieved track.
    """
    if len(neighbor_artists) == 0:
        return 0.0
    return len(set(neighbor_artists)) / len(neighbor_artists)


def intra_list_diversity_at_k(neighbor_embs):
    """Intra-List Diversity (ILD): 1 - mean pairwise cosine similarity of the
    recommended embeddings.

    Range: [0, 2] in theory (cosine ∈ [-1, 1]), but effectively [0, 1] for
    unit-norm embeddings. Higher = more diverse recommendations in embedding space.

    Args:
        neighbor_embs: (k, D) array of (preferably L2-normalized) embeddings
                       for retrieved tracks.
    """
    embs = torch.as_tensor(neighbor_embs, dtype=torch.float32)
    k = len(embs)
    if k < 2:
        return 0.0

    normed = torch.nn.functional.normalize(embs, dim=1)
    sim_matrix = normed @ normed.T

    # Mean of upper triangle = (total sum - diagonal sum) / k*(k-1)
    # Diagonal entries are all 1.0 for unit-norm vectors, so diagonal sum = k
    mean_sim = (sim_matrix.sum() - k) / (k**2 - k)

    return 1.0 - mean_sim.item()

def novelty_at_k(neighbor_artists, artist_counts):
    """
    Mean self-information (−log₂ p) of the recommended artists, where
    p(artist) is the artist's relative frequency in the full catalog. This
    metric adds more weight to recommendations from artists with fewer tracks
    in the full catalog more than artists with many tracks in the catalog.

    Higher = the model surfaces rarer, less-common artists.

    Args:
        neighbor_artists: array-like of artist name strings for each retrieved track.
        artist_counts:    dict mapping artist name → count in the full catalog.
    """
    if len(neighbor_artists) == 0 or not artist_counts:
        return 0.0

    total = sum(artist_counts.values())
    scores = []
    for artist in neighbor_artists:
        count = artist_counts.get(artist, 1)          # fallback to 1 to avoid log(0)
        prob = count / total
        scores.append(-np.log2(prob))

    return float(np.mean(scores))


def discounted_novelty_at_k(neighbor_artists, artist_counts):
    """Rank-discounted novelty: surprise scores weighted by position so that
    novel tracks appearing earlier contribute more.

    Uses the same logarithmic discount as DCG: weight at rank i = 1 / log2(i+1).
    Normalized by the sum of weights so the result is comparable across different k.

    Higher = model surfaces surprising/rare artists earlier in the ranking.

    Args:
        neighbor_artists: array-like of artist name strings in ranked order.
        artist_counts:    dict mapping artist name → count in the full catalog.
    """
    if len(neighbor_artists) == 0 or not artist_counts:
        return 0.0

    total = sum(artist_counts.values())
    ranks = np.arange(1, len(neighbor_artists) + 1)
    discounts = 1.0 / np.log2(ranks + 1)

    surprises = []
    for artist in neighbor_artists:
        count = artist_counts.get(artist, 1)
        surprises.append(-np.log2(count / total))

    return float(np.dot(surprises, discounts) / discounts.sum())
