from dataclasses import dataclass
from typing import Literal, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TimeDiscretizer:
    """
    Maps continuous times to discrete bins for DeepHit.
    DeepHit assumes discrete time bins t in {0..T-1}.

    """
    n_times: int
    t_min: float
    t_max: float
    bin_edges: torch.Tensor      # [T+1]
    bin_centers: torch.Tensor    # [T]

    @classmethod
    def from_data(
        cls,
        times: Union[np.ndarray, torch.Tensor],
        n_times: int,
        scheme: Literal["linear", "quantile"] = "linear",
    ) -> "TimeDiscretizer":
        """
        Build discretizer from training times.

        scheme="linear": equal-width bins between min and max.
        scheme="quantile": equal-mass bins (based on quantiles).
        """
        if isinstance(times, torch.Tensor):
            times_np = times.detach().cpu().numpy()
        else:
            times_np = np.asarray(times)

        t_min = float(times_np.min())
        t_max = float(times_np.max())

        if scheme == "linear":
            edges = np.linspace(t_min, t_max, n_times + 1)
        elif scheme == "quantile":
            # split into n_times quantile-based bins
            qs = np.linspace(0.0, 1.0, n_times + 1)
            edges = np.quantile(times_np, qs)
            # ensure strictly increasing (avoid identical edges if many ties)
            edges = np.unique(edges)
            # if uniqueness reduces bin count, fall back to linear
            if len(edges) < n_times + 1:
                edges = np.linspace(t_min, t_max, n_times + 1)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        bin_edges = torch.tensor(edges, dtype=torch.float32)
        # centers as midpoints between edges
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return cls(
            n_times=n_times,
            t_min=t_min,
            t_max=t_max,
            bin_edges=bin_edges,
            bin_centers=centers,
        )

    def encode(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map continuous times to integer bin indices [0 .. n_times-1].

        We use torch.bucketize against internal bin_edges.
        """
        # Move edges to same device as t
        edges = self.bin_edges.to(t.device)
        # We want bins [0..T-1]; internal edges[1:-1] act as boundaries
        # This returns indices in [0..T-1]
        idx = torch.bucketize(t, edges[1:-1], right=False)
        return idx.long()  # [B]

    def expected_time_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Expected time under the *marginal* event-time distribution:
            p(t) = sum_k p(k,t)

        logits expected shape: [B, K, T] (pre-softmax).
        """
        p_kt = probs_from_logits(logits)         # [B, K, T]
        p_t = p_kt.sum(dim=1)                    # [B, T]
        centers = self.bin_centers.to(logits.device)  # [T]
        return (p_t * centers.unsqueeze(0)).sum(dim=-1)  # [B]
    
    def transform_time(self, t: Union[np.ndarray, float, list]) -> np.ndarray:
        """
        Map continuous times to integer bin indices [0 .. n_times-1] using the same
        semantics as `encode()` (torch.bucketize with edges[1:-1], right=False).

        Accepts scalar or array-like, returns np.ndarray of dtype int64.
        """
        t_np = np.asarray(t, dtype=np.float64)

        # edges is torch.Tensor [T+1]; convert to numpy on CPU
        edges_np = self.bin_edges.detach().cpu().numpy().astype(np.float64)

        # torch.bucketize(t, edges[1:-1], right=False) is equivalent to:
        # np.searchsorted(edges[1:-1], t, side="left")
        boundaries = edges_np[1:-1]  # length T-1
        idx = np.searchsorted(boundaries, t_np, side="left")

        # clamp to valid range just in case of numerical edge cases
        idx = np.clip(idx, 0, self.n_times - 1).astype(np.int64)
        return idx

def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits [B, K, T] to a joint distribution p(k,t|x) via softmax over K*T.
    This matches the common DeepHit formulation (joint PMF over (k,t)).

    Returns:
        p_kt: [B, K, T], sum_{k,t} p_kt = 1 for each sample.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits [B,K,T], got shape {tuple(logits.shape)}")

    B, K, T = logits.shape
    flat = logits.reshape(B, K * T)
    p_flat = F.softmax(flat, dim=-1)
    return p_flat.reshape(B, K, T)

def deephit_nll_loss(
    logits: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    discretizer: TimeDiscretizer,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    DeepHit negative log-likelihood with right censoring.

    Model defines p(k,t|x) over causes k=1..K (stored 0..K-1) and time bins t=0..T-1.

    For an observed event (event>0):
        L_i = -log p(k_i, t_i)

    For a censored sample (event==0) censored at t_i:
        L_i = -log P(T > t_i) = -log sum_{k} sum_{t > t_i} p(k,t)
    """
    device = logits.device
    p_kt = probs_from_logits(logits)  # [B, K, T]

    t_idx = discretizer.encode(time.to(device).float())  # [B]
    event = event.to(device).long()

    B, K, T = p_kt.shape
    # Precompute survival tail mass S(t) = sum_k sum_{u>t} p(k,u)
    # tail_mass[b] = P(T > t_idx[b])
    cumsum_t = torch.cumsum(p_kt, dim=-1)        # [B, K, T], cumulative over time
    total_mass = cumsum_t[..., -1]               # [B, K] == sum_t p(k,t)
    # mass up to t (inclusive): cumsum_t[..., t_idx]
    # tail per k: total_mass - cumsum_at_t
    idx_expand = t_idx.view(B, 1, 1).expand(B, K, 1)  # [B,K,1]
    cumsum_at_t = torch.gather(cumsum_t, dim=-1, index=idx_expand).squeeze(-1)  # [B,K]
    tail_per_k = total_mass - cumsum_at_t  # [B,K]
    tail_mass = tail_per_k.sum(dim=1).clamp_min(eps)  # [B]

    # Event likelihood: gather p(k, t)
    mask_e = event > 0
    loss = torch.zeros((), device=device)

    if mask_e.any():
        k_idx = (event[mask_e] - 1).clamp(min=0)  # [N_e], 0..K-1
        t_e = t_idx[mask_e]                       # [N_e]
        p_e = p_kt[mask_e, :, :]                  # [N_e,K,T]

        row = torch.arange(p_e.size(0), device=device)
        p_kt_e = p_e[row, k_idx, t_e].clamp_min(eps)
        loss_event = -torch.log(p_kt_e).mean()
        loss = loss + loss_event

    # Censor likelihood
    mask_c = event == 0
    if mask_c.any():
        loss_cens = -torch.log(tail_mass[mask_c]).mean()
        loss = loss + loss_cens

    return loss


def deephit_ranking_loss(
    logits: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    discretizer: TimeDiscretizer,
    max_pairs: int = 8192,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    DeepHit-style ranking loss using cumulative incidence F_k(t).

    For each cause k, for pairs (i,j) where:
      - sample i experienced event k (event_i == k)
      - time_i < time_j
    Encourage:
      F_k(time_i | x_i) > F_k(time_i | x_j)

    We sample up to `max_pairs` comparable pairs to avoid O(B^2).
    """
    device = logits.device
    p_kt = probs_from_logits(logits)  # [B, K, T]

    t_idx = discretizer.encode(time.to(device).float())  # [B]
    event = event.to(device).long()
    B, K, T = p_kt.shape

    # cumulative incidence per cause: F_k(t) = sum_{u<=t} p(k,u)
    F_kt = torch.cumsum(p_kt, dim=-1)  # [B,K,T]

    # Build candidate comparable pairs
    # i must be an event sample
    event_mask = event > 0
    if not event_mask.any():
        return torch.zeros((), device=device)

    # Construct all comparable pairs indices (still can be large, so we sample smartly)
    # We will sample i from event samples, and sample j uniformly, then filter time_i < time_j.
    i_candidates = torch.where(event_mask)[0]
    if i_candidates.numel() == 0:
        return torch.zeros((), device=device)

    # Oversample then filter to reach max_pairs
    # Heuristic: try 4x and then trim
    num_try = min(max_pairs * 4, B * max(1, i_candidates.numel()))
    # sample i indices from event samples
    i_idx = i_candidates[torch.randint(0, i_candidates.numel(), (num_try,), device=device)]
    # sample j indices from all samples
    j_idx = torch.randint(0, B, (num_try,), device=device)

    # comparable if time_i < time_j
    comp = t_idx[i_idx] < t_idx[j_idx]
    i_idx = i_idx[comp]
    j_idx = j_idx[comp]
    if i_idx.numel() == 0:
        return torch.zeros((), device=device)

    # trim to max_pairs
    if i_idx.numel() > max_pairs:
        sel = torch.randperm(i_idx.numel(), device=device)[:max_pairs]
        i_idx = i_idx[sel]
        j_idx = j_idx[sel]

    # event type for i -> k in 0..K-1
    k_idx = (event[i_idx] - 1).clamp(min=0, max=K - 1)  # [P]
    t_i = t_idx[i_idx]                                   # [P]

    # gather F_k(t_i | x_i) and F_k(t_i | x_j)
    # F_kt is [B,K,T]
    P = i_idx.numel()
    row = torch.arange(P, device=device)

    Fi = F_kt[i_idx, :, :]  # [P,K,T]
    Fj = F_kt[j_idx, :, :]  # [P,K,T]

    Fi_kti = Fi[row, k_idx, t_i].clamp_min(eps)  # [P]
    Fj_kti = Fj[row, k_idx, t_i].clamp_min(eps)  # [P]

    # ranking loss: -log(sigmoid(Fi - Fj))
    diff = Fi_kti - Fj_kti
    return -torch.log(torch.sigmoid(diff) + eps).mean()


def deephit_loss(
    logits: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    discretizer: TimeDiscretizer,
    alpha: float = 1.0,
    beta: float = 1.0,
    max_pairs: int = 8192,
) -> torch.Tensor:
    """
    Combined DeepHit objective:
        alpha * NLL + beta * ranking

    Notes:
      - NLL includes censored samples properly via tail mass.
      - Ranking uses sampled pairs to stay efficient.
    """
    nll = deephit_nll_loss(logits, time, event, discretizer)
    rank = deephit_ranking_loss(logits, time, event, discretizer, max_pairs=max_pairs)
    return alpha * nll + beta * rank
