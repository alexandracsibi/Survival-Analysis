import numpy as np


def deephit_survival_at_horizons(probs, discretizer, horizons):
    """
    Compute event-free survival S(t) = P(T > t) from DeepHit joint pmf p(k,t_bin).

    probs: [N, K, T] joint distribution, sum_{k,t} = 1 for each sample
    discretizer: must have transform_time(t)->bin index
    horizons: iterable of t values

    Returns:
      S: [N, M] survival probabilities at horizons
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 3:
        raise ValueError(f"Expected probs [N,K,T], got {p.shape}")

    N, K, T = p.shape
    # marginal over time: p_t = sum_k p(k,t)
    p_t = p.sum(axis=1)  # [N, T]

    # survival at bin j: S(j) = sum_{u>j} p_t[u]
    # precompute tail sums
    tail = np.flip(np.cumsum(np.flip(p_t, axis=1), axis=1), axis=1)  # tail[u] = sum_{v>=u} p_t[v]
    # sum_{u>j} = tail[j+1] with boundary
    horizons = list(horizons)
    S = np.zeros((N, len(horizons)), dtype=float)

    for m, t0 in enumerate(horizons):
        j = int(discretizer.transform_time(np.asarray([float(t0)], dtype=float))[0])
        j = max(0, min(T - 1, j))
        if j >= T - 1:
            S[:, m] = 0.0
        else:
            S[:, m] = tail[:, j + 1]

    return np.clip(S, 0.0, 1.0)
