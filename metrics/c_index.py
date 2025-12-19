import numpy as np


class _FenwickTree:
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.zeros(self.n + 1, dtype=np.int64)

    def add(self, i: int, delta: int = 1) -> None:
        n = self.n
        bit = self.bit
        while i <= n:
            bit[i] += delta
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        bit = self.bit
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    def total(self) -> int:
        return int(self.sum(self.n))


def concordance_index(time, risk_score, event) -> float:
    """
    Harrell's C-index (O(N log N)).

    Conventions:
      - higher risk_score => earlier event (worse prognosis)
      - comparable pairs: (i, j) where time_i < time_j and event_i == 1
      - equal times excluded
      - tied risk contributes 0.5
    """
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk_score, dtype=float)
    event = (np.asarray(event) > 0).astype(np.int8)

    n = len(time)
    if n < 2:
        return float("nan")

    # Coordinate compress risk -> ranks 1..M (ties share rank)
    _, inv = np.unique(risk, return_inverse=True)
    ranks = inv.astype(np.int64) + 1
    M = int(ranks.max())

    # Sort by time DESC so BIT accumulates strictly LATER times
    order = np.argsort(time, kind="mergesort")[::-1]
    time_s = time[order]
    event_s = event[order]
    rank_s = ranks[order]

    bit = _FenwickTree(M)
    num = 0.0
    den = 0.0

    i = 0
    while i < n:
        t = time_s[i]
        j = i
        while j < n and time_s[j] == t:
            j += 1

        # At this moment, BIT contains samples with time strictly > t
        total_later = bit.total()
        if total_later > 0:
            # For each event at time t, compare with all later times
            for k in range(i, j):
                if event_s[k] == 1:
                    r = int(rank_s[k])
                    # Concordant: later subjects with LOWER risk (since higher = worse)
                    lower = bit.sum(r - 1)
                    # Ties: later subjects with EXACT same risk
                    leq = bit.sum(r)
                    ties = leq - lower

                    den += total_later
                    num += float(lower) + 0.5 * float(ties)

        # Insert this time block AFTER querying so equal-time are excluded
        for k in range(i, j):
            bit.add(int(rank_s[k]), 1)

        i = j

    return num / den if den > 0 else float("nan")