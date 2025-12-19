import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


# ---------------------------------------------------------------------
# Generic binary PR-AUC + F1 from continuous scores
# ---------------------------------------------------------------------
def pr_auc_and_f1_from_scores(y_true, scores, threshold: float = None):
    """
    Standard binary classification PR-AUC and F1 from continuous scores.

    Args:
        y_true: array-like [N], 0/1
        scores: array-like [N], higher = more likely positive
        threshold: optional. If None, choose threshold that maximizes F1 on PR curve.

    Returns:
        pr_auc, f1, thr_used
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    # PR-AUC
    if y_true.sum() == 0:
        pr_auc = 0.0
    else:
        pr_auc = float(average_precision_score(y_true, scores))

    # Choose threshold (maximize F1 on PR curve)
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)

        if thresholds.size == 0:
            thr_used = float(np.median(scores))
            y_pred = (scores >= thr_used).astype(int)
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            return pr_auc, f1, thr_used

        p = precision[:-1]
        r = recall[:-1]
        f1s = (2 * p * r) / (p + r + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        thr_used = float(thresholds[best_idx])
    else:
        thr_used = float(threshold)

    y_pred = (scores >= thr_used).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return pr_auc, f1, thr_used


# ---------------------------------------------------------------------
# Survival-aware: build labels at a time horizon t0 (handles censoring)
# ---------------------------------------------------------------------
def survival_horizon_labels(
    time,
    event,
    t0: float,
    event_of_interest: int = 1,
):
    """
    Create a binary label for "event_of_interest happened by t0" with censoring handling.

    Definitions:
      - Positive (y=1): event == event_of_interest AND time <= t0
      - Negative (y=0): time > t0  (known event-free beyond horizon)
      - Excluded: censored (event==0) with time <= t0 (unknown at horizon)

    Returns:
      y:    array [M] of 0/1 labels on evaluable subset
      mask: boolean array [N], True where sample is evaluable at t0
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    pos = (event == event_of_interest) & (time <= t0)
    neg = time > t0
    mask = pos | neg

    y = pos[mask].astype(int)
    return y, mask


# ---------------------------------------------------------------------
# Survival-aware PR-AUC + F1 at horizon t0 (for Cox/DeepSurv style scores)
# ---------------------------------------------------------------------
def pr_auc_f1_survival_at_horizon_from_scores(
    time,
    event,
    scores,
    t0: float,
    event_of_interest: int = 1,
    threshold: float = None,
):
    """
    Compute meaningful PR-AUC and F1 for survival models at a fixed horizon t0.

    For Cox/DeepSurv:
      - pass scores = log_risk (higher => higher hazard)
      - This evaluates how well log_risk separates "event by t0" vs "event-free beyond t0",
        excluding censored before t0.

    Returns:
      pr_auc, f1, thr_used, n_eval, n_pos, pos_rate
    """
    scores = np.asarray(scores, dtype=float)

    y, mask = survival_horizon_labels(time, event, t0, event_of_interest=event_of_interest)
    s = scores[mask]

    n_eval = int(mask.sum())
    n_pos = int(y.sum())
    pos_rate = float(n_pos / max(n_eval, 1))

    if n_eval == 0 or n_pos == 0:
        return np.nan, np.nan, np.nan, n_eval, n_pos, pos_rate

    pr_auc, f1, thr_used = pr_auc_and_f1_from_scores(y, s, threshold=threshold)
    return pr_auc, f1, thr_used, n_eval, n_pos, pos_rate

def _km_survival_censoring(time, cens_event):
    """
    Kaplan–Meier estimator for censoring survival G(t) = P(C > t).
    time: [N] times
    cens_event: [N] 1 if censoring observed, 0 otherwise
                (i.e., "event" in the censoring process)
    Returns:
      unique_times: sorted unique times where events/censoring occur
      G_at_unique:  G(t) evaluated right-continuously at those times
    """
    time = np.asarray(time, dtype=float)
    cens_event = np.asarray(cens_event, dtype=int)

    order = np.argsort(time)
    t = time[order]
    d = cens_event[order]

    uniq = np.unique(t)
    n = len(t)
    G = 1.0

    G_at = []
    at_times = []

    # risk set size just before each uniq time
    for ut in uniq:
        # at risk: those with time >= ut
        n_risk = np.sum(t >= ut)
        if n_risk <= 0:
            break

        # censoring events at ut
        d_c = np.sum((t == ut) & (d == 1))

        # KM step for censoring survival
        if d_c > 0:
            G *= (1.0 - d_c / n_risk)

        at_times.append(ut)
        G_at.append(G)

    return np.asarray(at_times, dtype=float), np.asarray(G_at, dtype=float)

def _G_hat(t_query, km_times, km_G):
    """
    Evaluate G(t_query) using the KM curve.
    Right-continuous step function; returns G(t_query).
    """
    if len(km_times) == 0:
        return 1.0
    idx = np.searchsorted(km_times, t_query, side="right") - 1
    if idx < 0:
        return 1.0
    return float(km_G[idx])

def time_dependent_auc_ipcw(
    time,
    event,
    scores,
    t0: float,
    event_of_interest: int = 1,
    eps: float = 1e-12,
):
    """
    Cumulative/dynamic time-dependent AUC at horizon t0 with IPCW (Uno-style).

    Definitions (cause-specific):
      - Define event'=1 if event==event_of_interest else 0
      - Treat competing events as censored at their time (standard cause-specific evaluation)
      - Cases: event'=1 and time <= t0
      - Controls: time > t0 (event-free of event_of_interest at t0)
      - IPCW weights:
          w_case(i) = 1 / G(T_i)  for cases
          w_ctrl(j) = 1 / G(t0)   for controls
        where G(.) is KM survival of the censoring distribution.

    Returns:
      auc_t0, n_cases, n_controls
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    scores = np.asarray(scores, dtype=float)

    # cause-specific event indicator
    e = (event == event_of_interest).astype(int)

    # censoring indicator for KM of censoring:
    # "censoring observed" if event == 0
    # (competing events are not censoring in raw data; they act like censoring for e,
    #  but for censoring KM we keep the true administrative censoring only)
    cens_obs = (event == 0).astype(int)

    km_t, km_G = _km_survival_censoring(time, cens_obs)

    # cases and controls for AUC(t0)
    is_case = (e == 1) & (time <= t0)
    is_ctrl = (time > t0)

    idx_case = np.where(is_case)[0]
    idx_ctrl = np.where(is_ctrl)[0]

    n_cases = idx_case.size
    n_ctrl = idx_ctrl.size
    if n_cases == 0 or n_ctrl == 0:
        return float("nan"), int(n_cases), int(n_ctrl)

    # weights
    w_case = np.array([1.0 / max(_G_hat(time[i], km_t, km_G), eps) for i in idx_case], dtype=float)
    G_t0 = max(_G_hat(t0, km_t, km_G), eps)
    w_ctrl = np.full(n_ctrl, 1.0 / G_t0, dtype=float)

    s_case = scores[idx_case]
    s_ctrl = scores[idx_ctrl]

    # Weighted AUC: sum_{i,j} w_i w_j I(s_i > s_j) / (sum w_i)(sum w_j)
    # Handle ties as 0.5.
    num = 0.0
    denom = float(w_case.sum() * w_ctrl.sum())
    if denom <= 0:
        return float("nan"), int(n_cases), int(n_ctrl)

    # O(n_cases * n_ctrl) implementation (OK for val/test sizes; if huge, we can optimize)
    for i in range(n_cases):
        wi = w_case[i]
        si = s_case[i]
        gt = (si > s_ctrl).astype(float)
        eq = (si == s_ctrl).astype(float)
        num += wi * np.sum(w_ctrl * (gt + 0.5 * eq))

    auc = num / denom
    return float(auc), int(n_cases), int(n_ctrl)

# ---------------------------------------------------------------------
# DeepHit-specific: get a score = CIF(t0) for event k, then compute PR/F1
# ---------------------------------------------------------------------
def deephit_cif_at_horizon(
    probs,
    t_index: int,
    event_of_interest: int = 1,
    has_censoring_channel: bool = False,
):
    """
    Extract CIF at a given discrete time index for DeepHit probabilities.

    Expected probs shapes (common variants):
      - [N, K, T]  where K = number of events (excluding censoring), T = time bins
      - [N, K+1, T] where index 0 is censoring and events start at 1
      - [N, T, K] (less common) -> not handled here intentionally

    We return CIF_k(t_index) := sum_{j<=t_index} P(event=k at time-bin j).

    Args:
      probs: numpy array or torch-like array convertible via np.asarray
      t_index: int in [0, T-1]
      event_of_interest: event id (1..K) in your label convention

    Returns:
      scores: array [N] in [0,1]
    """
    p = np.asarray(probs)

    if p.ndim != 3:
        raise ValueError(f"Expected probs with 3 dims [N,K,T] (or [N,K+1,T]); got shape {p.shape}")

    N, Kmaybe, T = p.shape

    if not (0 <= t_index < T):
        raise ValueError(f"t_index out of range: {t_index} for T={T}")

    if has_censoring_channel:
        # p[:,0,:] = censoring, p[:,1,:] = event 1, ...
        k_idx = event_of_interest
    else:
        # p[:,0,:] = event 1, p[:,1,:] = event 2, ...
        k_idx = event_of_interest - 1

    if not (0 <= k_idx < Kmaybe):
        raise ValueError(f"event_of_interest={event_of_interest} incompatible with probs shape {p.shape}")

    pmf_k = p[:, k_idx, :]              # [N, T]
    cif_k = np.cumsum(pmf_k, axis=1)    # [N, T]
    return cif_k[:, t_index]            # [N]


def pr_auc_f1_deephit_at_horizon(
    time,
    event,
    probs,
    t0: float,
    discretizer,
    event_of_interest: int = 1,
    threshold: float = None,
):
    """
    Meaningful PR-AUC and F1 for DeepHit at horizon t0.

    Score is CIF_k(t0) extracted from probs using the discretizer.

    Args:
      time, event: ground-truth arrays
      probs: DeepHit predicted probabilities (ideally PMF over time for each event)
      t0: horizon in the same units as time
      discretizer: your TimeDiscretizer instance (must map t0 -> time bin index)
      event_of_interest: which event to treat as positive
      threshold: optional

    Returns:
      pr_auc, f1, thr_used, n_eval, n_pos, pos_rate
    """
    # you need a stable way to map t0 -> bin index
    if not hasattr(discretizer, "transform_time"):
        raise ValueError("discretizer must have a transform_time(t) -> bin_index method")

    t_index = int(discretizer.transform_time(np.asarray([t0], dtype=float))[0])

    scores = deephit_cif_at_horizon(probs, t_index=t_index, event_of_interest=event_of_interest)
    return pr_auc_f1_survival_at_horizon_from_scores(
        time=time,
        event=event,
        scores=scores,
        t0=t0,
        event_of_interest=event_of_interest,
        threshold=threshold,
    )
