import numpy as np
from .metrics import jaccard_similarity

def build_cluster_relevance_vector(
    query_cluster_set,
    neighbor_cluster_sets,
    query_dom,
    neighbor_doms,
    overlap_threshold=0.25
):

    k = len(neighbor_cluster_sets)
    relevance = np.zeros(k, dtype=int)

    for i, (nset, ndom) in enumerate(zip(neighbor_cluster_sets, neighbor_doms)):

        if query_dom == ndom:
            relevance[i] = 1
        else:
            overlap = jaccard_similarity(query_cluster_set, nset)
            relevance[i] = int(overlap >= overlap_threshold)

    return relevance