import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


@dataclass
class KNNGraph:
    """
    Minimal kNN graph container.

    - num_nodes: N
    - neighbors: list of length N, each is LongTensor [deg]
      (deg typically = k, symmetric graph can make it larger)
    """
    num_nodes: int
    neighbors: list[torch.Tensor]

    def to(self, device: str) -> "KNNGraph":
        self.neighbors = [n.to(device) for n in self.neighbors]
        return self


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_cache_path(base_dir: str, split: str, k: int) -> str:
    graph_dir = os.path.join(base_dir, "graph")
    _ensure_dir(graph_dir)
    return os.path.join(graph_dir, f"{split}_knn_k{k}.pt")


def build_knn_graph_from_features(
    X: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    symmetric: bool = False,
) -> KNNGraph:
    """
    Build a kNN graph on CPU using sklearn.

    X: [N, D] numpy float array
    Returns: KNNGraph with adjacency list neighbors[i] -> LongTensor of neighbor ids
    """
    X = np.asarray(X, dtype=np.float32)
    N = int(X.shape[0])

    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric=metric)
    nn.fit(X)
    neigh = nn.kneighbors(return_distance=False)  # [N, k+1] includes self

    # directed edges i -> neigh[i][1:]
    nbrs = [set() for _ in range(N)]
    for i in range(N):
        for j in neigh[i][1:]:
            if i != j:
                nbrs[i].add(int(j))

    neighbors = []
    for i in range(N):
        arr = np.fromiter(nbrs[i], dtype=np.int64)
        neighbors.append(torch.from_numpy(arr).long())

    return KNNGraph(num_nodes=N, neighbors=neighbors)


def extract_features_from_dataset(
    dataset: Dataset,
    feature_key: str = "x",
) -> np.ndarray:
    """
    Extract features in dataset order by iterating dataset[i].

    Assumption (matches your pipeline):
    - dataset[i][feature_key] is a Tensor/array shaped [D]
    - idx returned by dataset is consistent with dataset order (0..N-1).
      If you wrap with Subset, DO NOT use this; use the underlying full train dataset.
    """
    X_list: list[np.ndarray] = []
    for i in range(len(dataset)):
        item = dataset[i]
        x = item[feature_key]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        X_list.append(np.asarray(x, dtype=np.float32))
    return np.vstack(X_list)


def load_or_build_knn_graph(
    base_dir: str,
    split: str,
    k: int,
    X: Optional[np.ndarray] = None,
    dataset: Optional[Dataset] = None,
    feature_key: str = "x",
    metric: str = "euclidean",
    symmetric: bool = False,
    cache_path: Optional[str] = None,
) -> KNNGraph:
    """
    Load cached graph if exists, otherwise build and save.

    Provide either:
    - X (preferred), or
    - dataset (will be iterated to extract X)

    Cache file stores:
    - num_nodes
    - neighbors (list of tensors)
    """
    if cache_path is None:
        cache_path = _default_cache_path(base_dir, split, k)

    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        return KNNGraph(num_nodes=int(obj["num_nodes"]), neighbors=obj["neighbors"])

    if X is None:
        if dataset is None:
            raise ValueError("Provide X or dataset to build the graph.")
        X = extract_features_from_dataset(dataset, feature_key=feature_key)

    graph = build_knn_graph_from_features(X, k=k, metric=metric, symmetric=symmetric)
    torch.save({"num_nodes": graph.num_nodes, "neighbors": graph.neighbors}, cache_path)
    return graph
