import torch

"""
Cox loss uses relative risk, not absolute time prediction.
For each event, compares its risk to all patients still at risk at that time.
Works with any model that outputs a scalar log_risk per sample
"""

def cox_ph_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood.

    Args:
        log_risk: [B] log-risk scores from the model
        time:     [B] observed times
        event:    [B] 1 if event occurred, 0 if censored

    Returns:
        scalar loss (to minimize)
    """
    log_risk = log_risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)

    order = torch.argsort(time, descending=True)
    log_risk_sorted = log_risk[order]
    event_sorted = event[order]

    # Cox expects binary event indicator (any event)
    event_sorted = (event_sorted > 0).float()

    log_cumsum = torch.logcumsumexp(log_risk_sorted, dim=0)
    diff = log_risk_sorted - log_cumsum

    loss = -torch.sum(diff * event_sorted) / (event_sorted.sum() + 1e-8)
    return loss