import numpy as np

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