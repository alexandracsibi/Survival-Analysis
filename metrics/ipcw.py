import numpy as np


def km_censoring_survival(time, event):
    """
    KM estimate of censoring survival G(t) = P(C > t).

    We define censoring event indicator:
      cens_obs = 1 if event == 0 else 0

    Args:
        time:  [N]
        event: [N] (0=censored, >0=event/competing event)

    Returns:
        km_times: sorted unique times
        km_G:     G(t) right-continuous step values at km_times
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    cens_obs = (event == 0).astype(int)

    order = np.argsort(time)
    t = time[order]
    d = cens_obs[order]

    uniq = np.unique(t)
    G = 1.0
    km_times = []
    km_G = []

    for ut in uniq:
        n_risk = np.sum(t >= ut)
        if n_risk <= 0:
            break
        d_c = np.sum((t == ut) & (d == 1))
        if d_c > 0:
            G *= (1.0 - d_c / n_risk)
        km_times.append(ut)
        km_G.append(G)

    return np.asarray(km_times, dtype=float), np.asarray(km_G, dtype=float)


def G_hat(t_query, km_times, km_G):
    """
    Evaluate KM censoring survival at t_query (right-continuous).
    Returns G(t_query).
    """
    if km_times.size == 0:
        return 1.0
    idx = np.searchsorted(km_times, t_query, side="right") - 1
    if idx < 0:
        return 1.0
    return float(km_G[idx])
