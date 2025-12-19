import numpy as np
from .time_dependent_auc import td_auc_uno_ipcw


def integrated_auc(time, event, scores, horizons, event_of_interest=1):
    """
    Integrated AUC over a list/array of horizons.
    Returns the mean of defined AUC(t).

    Args:
        horizons: iterable of t values (in the same units as time)
    Returns:
        iauc, aucs_dict
    """
    horizons = np.asarray(list(horizons), dtype=float)
    aucs = []
    aucs_dict = {}

    for t0 in horizons:
        auc_t, n_cases, n_ctrl = td_auc_uno_ipcw(
            time=time, event=event, scores=scores, t0=float(t0), event_of_interest=event_of_interest
        )
        aucs_dict[float(t0)] = {"auc": auc_t, "cases": n_cases, "ctrls": n_ctrl}
        if not np.isnan(auc_t):
            aucs.append(auc_t)

    if len(aucs) == 0:
        return float("nan"), aucs_dict
    return float(np.mean(aucs)), aucs_dict
