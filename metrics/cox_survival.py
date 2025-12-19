import numpy as np


def breslow_baseline_hazard(time, event, log_risk):
    """
    Breslow estimator of cumulative baseline hazard H0(t).
    event: binary indicator for "any event" (1 if event>0 else 0)

    Returns:
      unique_event_times, H0_at_times  (right-continuous)
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    log_risk = np.asarray(log_risk, dtype=float)

    # risk = exp(log_risk)
    risk = np.exp(log_risk)

    order = np.argsort(time)
    t = time[order]
    e = event[order]
    r = risk[order]

    event_times = np.unique(t[e == 1])
    H0 = 0.0
    H0_at = []
    out_times = []

    for ut in event_times:
        d = np.sum((t == ut) & (e == 1))          # number of events at ut
        risk_set = np.sum(r[t >= ut])             # sum risk in risk set
        if risk_set <= 0:
            continue
        H0 += d / risk_set
        out_times.append(ut)
        H0_at.append(H0)

    return np.asarray(out_times, dtype=float), np.asarray(H0_at, dtype=float)


def H0_hat(t_query, base_times, base_H0):
    if base_times.size == 0:
        return 0.0
    idx = np.searchsorted(base_times, t_query, side="right") - 1
    if idx < 0:
        return 0.0
    return float(base_H0[idx])


def cox_survival_at_horizons(time, event, log_risk, horizons):
    """
    Fit Breslow baseline on (time,event,log_risk) and return S(t|x) for each horizon.
    event should be binary any-event indicator: (event_np > 0)

    Returns:
      S: [N, M] where M=len(horizons)
    """
    horizons = np.asarray(list(horizons), dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    log_risk = np.asarray(log_risk, dtype=float)

    base_t, base_H0 = breslow_baseline_hazard(time, event, log_risk)
    risk = np.exp(log_risk)

    S = np.zeros((time.shape[0], horizons.size), dtype=float)
    for m, t0 in enumerate(horizons):
        H0 = H0_hat(float(t0), base_t, base_H0)
        # S(t|x) = exp(-H0(t) * exp(η))
        S[:, m] = np.exp(-H0 * risk)
    # clamp for numerical safety
    return np.clip(S, 0.0, 1.0)
