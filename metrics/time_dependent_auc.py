import numpy as np
from .ipcw import km_censoring_survival, G_hat


def td_auc_uno_ipcw(time, event, scores, t0, event_of_interest=1, eps=1e-12):
    """
    Time-dependent AUC at horizon t0 using IPCW (Uno-style).

    Cause-specific convention:
      - Cases: event==event_of_interest and time <= t0
      - Controls: time > t0 (event-free of EOI at t0)
      - Censoring weights use the *true censoring* indicator event==0 (KM of censoring).

    Args:
        time: [N]
        event: [N] with 0=censored, 1..K events
        scores:[N] risk score where higher => more likely to be a case by t0
        t0: float horizon
        event_of_interest: int
    Returns:
        auc_t0, n_cases, n_ctrl
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    scores = np.asarray(scores, dtype=float)

    km_t, km_G = km_censoring_survival(time, event)

    is_case = (event == event_of_interest) & (time <= t0)
    is_ctrl = (time > t0)

    idx_case = np.where(is_case)[0]
    idx_ctrl = np.where(is_ctrl)[0]

    n_cases = int(idx_case.size)
    n_ctrl = int(idx_ctrl.size)
    if n_cases == 0 or n_ctrl == 0:
        return float("nan"), n_cases, n_ctrl

    # IPCW weights
    w_case = np.array([1.0 / max(G_hat(time[i], km_t, km_G), eps) for i in idx_case], dtype=float)
    G_t0 = max(G_hat(t0, km_t, km_G), eps)
    w_ctrl = np.full(n_ctrl, 1.0 / G_t0, dtype=float)

    s_case = scores[idx_case]
    s_ctrl = scores[idx_ctrl]

    denom = float(w_case.sum() * w_ctrl.sum())
    if denom <= 0:
        return float("nan"), n_cases, n_ctrl

    num = 0.0
    # O(n_cases*n_ctrl); OK for val/test. If huge, we can optimize later.
    for i in range(n_cases):
        si = s_case[i]
        wi = w_case[i]
        gt = (si > s_ctrl).astype(float)
        eq = (si == s_ctrl).astype(float)
        num += wi * np.sum(w_ctrl * (gt + 0.5 * eq))

    return float(num / denom), n_cases, n_ctrl
