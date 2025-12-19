import torch


def graph_consistency_loss(
    student_risk: torch.Tensor,          # [B]
    batch_idx: torch.Tensor,             # [B] global node ids
    neighbors: list[torch.Tensor],       # adjacency list (global ids)
    teacher_risk_all: torch.Tensor,      # [N] cached, no-grad
    max_neighbors: int = 10,
) -> torch.Tensor:
    """
    Consistency regularization:
      mean_i mean_{j in N(i)} ( r_i - stopgrad(r_j) )^2

    Implemented using teacher_risk_all as stopgrad cache.
    Uses up to max_neighbors per node for speed.

    Returns a scalar tensor.
    """
    if student_risk.ndim != 1:
        student_risk = student_risk.view(-1)

    B = int(batch_idx.shape[0])
    if B == 0:
        return torch.zeros((), device=student_risk.device)

    loss_acc = 0.0
    count = 0

    # Simple loop = shortest, robust code.
    for b in range(B):
        i = int(batch_idx[b].item())
        nbr = neighbors[i]
        if nbr.numel() == 0:
            continue

        if max_neighbors > 0 and nbr.numel() > max_neighbors:
            nbr = nbr[:max_neighbors]

        # teacher_risk_all already no-grad
        rj = teacher_risk_all[nbr.to(teacher_risk_all.device)]
        ri = student_risk[b]
        mu = rj.mean()
        loss_acc = loss_acc + (ri - mu) ** 2
        count += 1

    if count == 0:
        return torch.zeros((), device=student_risk.device)

    return loss_acc / count