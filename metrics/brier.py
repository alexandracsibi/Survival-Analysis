import numpy as np
from .ipcw import km_censoring_survival, G_hat


def brier_score_ipcw(time, event, surv_prob, t0, eps=1e-12):
    """
    IPCW Brier score at time t0 for survival probability S(t0).

    Target:
      Y_i(t0) = I(T_i > t0)

    Weights (Graf et al. style):
      if time_i > t0: weight = 1 / G(t0)
      if time_i <= t0 and event_i > 0: weight = 1 / G(time_i)
      if time_i <= t0 and event_i == 0: excluded (unknown status at t0)

    Args:
      surv_prob: [N] predicted S(t0|x_i) in [0,1]
    Returns:
      brier_t0, n_eval
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    surv_prob = np.asarray(surv_prob, dtype=float)

    km_t, km_G = km_censoring_survival(time, event)

    # evaluable:
    # - time > t0 (known alive/event-free at t0)
    # - or time <= t0 and event>0 (known failed by t0)
    known = (time > t0) | ((time <= t0) & (event > 0))
    idx = np.where(known)[0]
    if idx.size == 0:
        return float("nan"), 0

    y = (time[idx] > t0).astype(float)  # 1 if survived past t0 else 0
    s = np.clip(surv_prob[idx], 0.0, 1.0)

    w = np.zeros(idx.size, dtype=float)
    G_t0 = max(G_hat(t0, km_t, km_G), eps)

    # controls: time > t0
    ctrl = time[idx] > t0
    w[ctrl] = 1.0 / G_t0

    # cases: failed by t0 (event>0 and time<=t0)
    case = ~ctrl
    for j in np.where(case)[0]:
        tj = time[idx[j]]
        w[j] = 1.0 / max(G_hat(tj, km_t, km_G), eps)

    # Weighted mean squared error
    num = np.sum(w * (y - s) ** 2)
    den = np.sum(w)
    if den <= 0:
        return float("nan"), int(idx.size)
    return float(num / den), int(idx.size)


def integrated_brier_score(time, event, surv_probs_matrix, horizons):
    """
    IBS over horizons using trapezoidal integration of Brier(t).

    Args:
      surv_probs_matrix: shape [N, M], where M=len(horizons), each column is S(t_m|x)
      horizons: array-like [M] increasing

    Returns:
      ibs, brier_by_t
    """
    horizons = np.asarray(horizons, dtype=float)
    if horizons.ndim != 1 or horizons.size < 2:
        raise ValueError("horizons must be 1D with at least 2 points")

    S = np.asarray(surv_probs_matrix, dtype=float)
    if S.shape[1] != horizons.size:
        raise ValueError(f"surv_probs_matrix must have shape [N, {horizons.size}]")

    briers = []
    brier_by_t = {}

    for m, t0 in enumerate(horizons):
        b, n_eval = brier_score_ipcw(time, event, S[:, m], float(t0))
        briers.append(b)
        brier_by_t[float(t0)] = {"brier": b, "n_eval": n_eval}

    briers = np.asarray(briers, dtype=float)
    # integrate ignoring NaNs by masking segments
    mask = ~np.isnan(briers)
    if mask.sum() < 2:
        return float("nan"), brier_by_t

    t = horizons[mask]
    b = briers[mask]
    ibs = np.trapz(b, t) / (t[-1] - t[0])
    return float(ibs), brier_by_t
