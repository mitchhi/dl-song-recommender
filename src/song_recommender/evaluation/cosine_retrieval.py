import torch


# OLD TOPK_COSINE FUNCTION

#def topk_cosine(emb: np.ndarray, query: np.ndarray, k: int, exclude_index = None):
#    if not np.isfinite(emb).all():
#        raise ValueError('Embedding matrix contains NaN or inf values. Rebuild embeddings or restart the kernel.')
#    if not np.isfinite(query).all():
#        raise ValueError('Query vector contains NaN or inf values. Rebuild embeddings or restart the kernel.')
#    # Use float64 einsum here to avoid spurious float32 matmul warnings from some NumPy/OpenBLAS builds.
#    scores = np.einsum('ij,j->i', emb.astype(np.float64), query.astype(np.float64), optimize=True)

#    if not np.isfinite(scores).all():
#        raise ValueError('Cosine scores contain NaN or inf values after matmul.')

#    # remove self match when checking queries on training set
#    if exclude_index is not None:
#        scores[exclude_index] = -np.inf   

#    k = int(min(max(k, 1), len(scores)))
#    idx = np.argpartition(-scores, kth=k - 1)[:k]
#    idx = idx[np.argsort(-scores[idx])]

#    return idx, scores[idx]

def topk_cosine(embeddings, query_vec, k=10, exclude_idx=None):
    emb = torch.as_tensor(embeddings, dtype=torch.float32)
    q = torch.as_tensor(query_vec, dtype=torch.float32)
    sims = emb @ q

    if exclude_idx is not None:
        sims[exclude_idx] = float("-inf")

    max_k = emb.shape[0] - (1 if exclude_idx is not None else 0)
    top_vals, top_idx = torch.topk(sims, k=min(int(k), int(max_k)), largest=True, sorted=True)
    return top_idx.cpu().numpy(), top_vals.cpu().numpy()
