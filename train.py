#!/usr/bin/env python3
import argparse
import os
import random
import json
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from datasets import (
    load_synthetic_all,
    load_support2_all,
    load_mnb_comprisk_all,
    load_mimiceye_tabular_all,
    load_mimiceye_multimodal_all,
    UnlabeledSurvivalWrapper
    )
from models import (
    MLPBackbone,
    DeepCoxPH,
    DeepHit,
    MultiModalCox,
    EmbeddedCoxPH,
    )
from trainers.supervised_trainer import SupervisedCoxTrainer
from trainers.deephit_trainer import DeepHitTrainer
from trainers.ssl_deephit_trainer import SSLDeepHitTrainer
from trainers.graphcox_trainer import GraphCoxTrainer
from losses.deephit import TimeDiscretizer
from graph.knn_graph import load_or_build_knn_graph

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def drop_columns_np(X: np.ndarray, drop_idx: list[int]) -> np.ndarray:
    if not drop_idx:
        return X
    mask = np.ones(X.shape[1], dtype=bool)
    mask[np.asarray(drop_idx, dtype=int)] = False
    return X[:, mask]

def get_cat_info_for_mimiceye(
    model_cfg: dict,
    feature_cols: list[str],
) -> Tuple[list[int], list[int]]:
    """
    Get indices and cardinalities for categorical ID columns for MIMIC-Eye.
    Uses category_mappings.json (num_classes) as the source of truth.
    """
    use_cat = bool(model_cfg.get("use_cat_embeddings", False))
    if not use_cat:
        return [], []

    cat_feature_names = model_cfg.get(
        "cat_feature_names",
        ["admission_type_id", "admission_location_id", "race_id"],
    )
    cat_feature_names = list(cat_feature_names)

    mapping_path = model_cfg.get("category_mapping_json")
    if mapping_path is None:
        raise ValueError("use_cat_embeddings=true but model.category_mapping_json is not set")

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Category mapping JSON not found: {mapping_path}")

    with open(mapping_path, "r") as f:
        mappings = json.load(f)

    cat_feature_indices, cat_cardinalities = [], []
    for col_name in cat_feature_names:
        if col_name not in feature_cols:
            raise ValueError(
                f"Categorical feature '{col_name}' not in feature_cols. "
                f"feature_cols={feature_cols}"
            )

        key = col_name.replace("_id", "")
        if key not in mappings:
            raise ValueError(f"Key '{key}' not found in mapping JSON. Keys={list(mappings.keys())}")

        idx = feature_cols.index(col_name)
        card = int(mappings[key]["num_classes"])

        cat_feature_indices.append(idx)
        cat_cardinalities.append(card)

    return cat_feature_indices, cat_cardinalities

def set_all_seeds(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class Datasets:
    train: Any
    val: Any
    test: Any


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ----------------------------------------------------------------------
# Dataset + DataLoader builders
# ----------------------------------------------------------------------

def build_datasets(cfg: Dict[str, Any]) -> Datasets:
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg.get("model", {})

    name = dataset_cfg["name"]
    base_dir = dataset_cfg["base_dir"]

    feature_cols = dataset_cfg.get("feature_cols", None)
    # feature_cols can be null/None in YAML -> infer automatically

    if name == "synthetic":
        train_ds, val_ds, test_ds = load_synthetic_all(base_dir, feature_cols)

    elif name == "support2":
        train_ds, val_ds, test_ds = load_support2_all(base_dir, feature_cols)

    elif name == "mnb_comprisk":
        train_ds, val_ds, test_ds = load_mnb_comprisk_all(base_dir, feature_cols)

    elif name == "mimiceye_tabular":
        # Pass cat-embedding metadata only when enabled
        use_cat = bool(model_cfg.get("use_cat_embeddings", False))
        cat_feature_names = model_cfg.get("cat_feature_names", None) if use_cat else None
        category_mapping_json = (
            model_cfg.get(
                "category_mapping_json",
                os.path.join(base_dir, "category_mappings.json"),
            )
            if use_cat
            else None
        )

        train_ds, val_ds, test_ds = load_mimiceye_tabular_all(
            base_dir,
            feature_cols=feature_cols,
            cat_feature_names=cat_feature_names,
            category_mapping_json=category_mapping_json,
        )

    elif name == "mimiceye_multimodal":
        train_ds, val_ds, test_ds = load_mimiceye_multimodal_all(
            base_dir,
            feature_cols=feature_cols,
            image_root=dataset_cfg.get("image_root", None),
            img_size=dataset_cfg.get("img_size", 224),
        )

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return Datasets(train_ds, val_ds, test_ds)


def build_dataloaders(
    datasets: Datasets,
    loader_cfg: Dict[str, Any],
) -> Loaders:
    train_bs = loader_cfg.get("train_batch_size", 256)
    val_bs = loader_cfg.get("val_batch_size", 512)
    shuffle_train = loader_cfg.get("shuffle_train", True)
    num_workers = loader_cfg.get("num_workers", 4)
    pin_memory = loader_cfg.get("pin_memory", True)

    train_loader = DataLoader(
        datasets.train,
        batch_size=train_bs,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets.val,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return Loaders(train_loader, val_loader, test_loader)

# ----------------------------------------------------------------------
# SSL DataLoader
# ----------------------------------------------------------------------

def build_ssl_train_loaders(
    cfg: Dict[str, Any],
    datasets: Datasets,
    loaders: Loaders,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Build labeled / unlabeled train loaders for SSL.
    - For DeepHit + train.ssl.enabled = True: split datasets.train into labeled/unlabeled.
    - Otherwise: return (loaders.train, None).
    """

    train_cfg = cfg.get("train", {})
    ssl_cfg = train_cfg.get("ssl", {})
    ssl_enabled = bool(ssl_cfg.get("enabled", False))
    model_type = cfg["model"]["type"].lower()

    # Graph-SSL for Cox can also use label-budget splitting.
    graph_cfg = train_cfg.get("graph_ssl", {})
    graph_enabled = bool(graph_cfg.get("enabled", False))

    # Which feature budget knob to use:
    # - DeepHit SSL: train.ssl.unlabeled_fraction
    # - Cox graph SSL: prefer train.graph_ssl.unlabeled_fraction, fallback to train.ssl.unlabeled_fraction
    if model_type == "deephit":
        unlabeled_fraction = float(ssl_cfg.get("unlabeled_fraction", 0.0))
    elif model_type == "deepsurv" and graph_enabled:
        unlabeled_fraction = float(graph_cfg.get("unlabeled_fraction", ssl_cfg.get("unlabeled_fraction", 0.0)))
        # For Cox graph SSL, we still do label-budget splitting even if ssl.enabled is false.
        # (graph regularization itself is controlled by graph_ssl.enabled)
    else:
        return loaders.train, None

    unlabeled_fraction = max(0.0, min(1.0, unlabeled_fraction))

    # If no unlabeled fraction, don't split
    if unlabeled_fraction <= 0.0:
        return loaders.train, None

    base_train_ds = datasets.train
    n = len(base_train_ds)

    n_unlabeled = int(n * unlabeled_fraction)
    n_labeled = n - n_unlabeled

    print(
        f"[DATA] Splitting train set: n={n}, "
        f"labeled={n_labeled}, unlabeled={n_unlabeled} "
        f"(ssl_enabled={ssl_enabled}) "
        f"(graph_enabled={graph_enabled})"
    )

    idx = np.random.permutation(n)
    unlabeled_idx = idx[:n_unlabeled]
    labeled_idx = idx[n_unlabeled:]

    labeled_ds = Subset(base_train_ds, labeled_idx)

    # Reuse train loader hyperparameters
    base_train_loader = loaders.train
    train_bs = base_train_loader.batch_size
    num_workers = base_train_loader.num_workers
    pin_memory = base_train_loader.pin_memory

    train_labeled_loader = DataLoader(
        labeled_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # If SSL disabled for DeepHit: return labeled only (supervised label-budget baseline)
    if model_type == "deephit" and not ssl_enabled:
        return train_labeled_loader, None

    # Otherwise return an unlabeled loader (wrapped)
    unlabeled_raw_ds = Subset(base_train_ds, unlabeled_idx)
    unlabeled_ds = UnlabeledSurvivalWrapper(unlabeled_raw_ds)

    train_unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_labeled_loader, train_unlabeled_loader
    
# ----------------------------------------------------------------------
# Optimizer builder
# ----------------------------------------------------------------------

def build_optimizer(train_cfg: Dict[str, Any], model: torch.nn.Module):
    opt_name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    params = filter(lambda p: p.requires_grad, model.parameters())

    if opt_name == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        opt = torch.optim.SGD(
            params, lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return opt

# ----------------------------------------------------------------------
# Model + Trainer + forward_fn builders
# ----------------------------------------------------------------------

@dataclass
class TrainingObjects:
    model: torch.nn.Module
    trainer: Any
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor]


def build_training_objects(
    cfg: Dict[str, Any],
    datasets: Datasets,
    graph=None,
) -> TrainingObjects:
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    metrics_cfg = cfg.get("metrics", {})
    metrics_horizon = float(metrics_cfg.get("horizon", 30.0))
    event_of_interest = int(metrics_cfg.get("event_of_interest", 1))
    metrics_horizons = metrics_cfg.get("horizons", None)

    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = dataset_cfg["name"]
    model_type = model_cfg["type"].lower()

    # ----------------- DeepSurv-style (Cox) -----------------
    if model_type == "deepsurv":
        # --- MIMIC-Eye multimodal: images + embedded tabular ---
        if dataset_name == "mimiceye_multimodal":
            # multimodal Cox
            cat_feature_indices, cat_cardinalities = get_cat_info_for_mimiceye(
                model_cfg=model_cfg,
                feature_cols=datasets.train.feature_cols,
            )

            num_tab_features = len(datasets.train.feature_cols)

            model = MultiModalCox(
                num_tab_features=num_tab_features,
                img_out_dim=model_cfg.get("img_out_dim", 128),
                tab_hidden_dims=model_cfg.get("tab_mlp_hidden_dims", [64, 32]),
                tab_dropout=float(model_cfg.get("tab_mlp_dropout", 0.1)),
                pretrained=bool(model_cfg.get("pretrained", True)),
                weights_path=model_cfg.get("weights_path", None),
                freeze_cnn=bool(model_cfg.get("freeze_cnn", False)),
                cat_feature_indices=cat_feature_indices,
                cat_cardinalities=cat_cardinalities,
            )
            def count_params(m):
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                return total, trainable

            total, trainable = count_params(model)
            pct = 100.0 * trainable / max(1, total)
            print(f"Model parameters: total={total:,} trainable={trainable:,} ({pct:.2f}%)")

            def forward_fn(m, batch):
                if not hasattr(forward_fn, "_printed"):
                    img = batch["image"]
                    print(
                        "[TRAIN DEBUG] Batch image tensor\n"
                        f"  shape: {tuple(img.shape)}\n"
                        f"  device: {img.device}\n"
                        f"  dtype: {img.dtype}"
                    )
                    forward_fn._printed = True

                return m(batch["x"], batch["image"])


        # --- tabular MIMIC-Eye: embedded CoxPH on tabular only ---
        elif dataset_name == "mimiceye_tabular":
            cat_feature_indices, cat_cardinalities = get_cat_info_for_mimiceye(
                model_cfg=model_cfg,
                feature_cols=datasets.train.feature_cols,
            )

            in_dim = len(datasets.train.feature_cols)

            if len(cat_feature_indices) == 0:
                backbone = MLPBackbone(
                    in_dim=in_dim,
                    hidden_dims=model_cfg.get("mlp_hidden_dims", [128, 64]),
                    dropout=float(model_cfg.get("mlp_dropout", 0.1)),
                )
                model = DeepCoxPH(backbone)
            else:
                model = EmbeddedCoxPH(
                    in_dim=in_dim,
                    cat_feature_indices=cat_feature_indices,
                    cat_cardinalities=cat_cardinalities,
                    hidden_dims=model_cfg.get("mlp_hidden_dims", [128, 64]),
                    dropout=float(model_cfg.get("mlp_dropout", 0.1)),
                )

            def forward_fn(m, batch):
                return m(batch["x"])

        # --- all other datasets: standard DeepSurv with plain MLPBackbone ---
        else:
            in_dim = len(datasets.train.feature_cols)
            backbone = MLPBackbone(
                in_dim=in_dim,
                hidden_dims=model_cfg.get("mlp_hidden_dims", [128, 64]),
                dropout=float(model_cfg.get("mlp_dropout", 0.1)),
            )
            model = DeepCoxPH(backbone)

            def forward_fn(m, batch):
                return m(batch["x"])

        optimizer = build_optimizer(train_cfg, model)

        if graph is not None:
            trainer = GraphCoxTrainer(
                model,
                optimizer,
                graph=graph,
                lambda_graph=float(train_cfg["graph_ssl"].get("lambda", 1.0)),
                max_neighbors=int(train_cfg["graph_ssl"].get("max_neighbors", 10)),
                device=device,
                metrics_horizon=metrics_horizon,
                event_of_interest=event_of_interest,
                metrics_horizons=metrics_horizons,
            )
        else:      
            trainer = SupervisedCoxTrainer(
                model,
                optimizer,
                device=device,
                metrics_horizon=metrics_horizon,
                event_of_interest=event_of_interest,
                metrics_horizons=metrics_horizons,
            )

        return TrainingObjects(model, trainer, forward_fn)

    # ----------------- DeepHit (single or competing risks) -----------------
    elif model_type == "deephit":
        in_dim = len(datasets.train.feature_cols)
        backbone = MLPBackbone(
            in_dim=in_dim,
            hidden_dims=model_cfg.get("mlp_hidden_dims", [128, 64]),
            dropout=float(model_cfg.get("mlp_dropout", 0.1)),
        )

        # n_times from config.dataset
        n_times = int(dataset_cfg.get("n_times", 50))
        # n_events from config or infer from train event column
        if "n_events" in model_cfg:
            n_events = int(model_cfg["n_events"])
        else:
            n_events = int(np.max(datasets.train.event))
            # if binary single-event dataset with labels 0/1
            # => n_events = 1 automatically

        model = DeepHit(
            backbone,
            n_times=n_times,
            n_events=n_events,
        )

        # build discretizer from training times
        disc_scheme = dataset_cfg.get("discretization", "linear")
        discretizer = TimeDiscretizer.from_data(
            datasets.train.time,
            n_times=n_times,
            scheme=disc_scheme,
        )

        optimizer = build_optimizer(train_cfg, model)
        alpha = float(train_cfg.get("deephit_alpha", 1.0))
        beta = float(train_cfg.get("deephit_beta", 1.0))

        # ---- SSL config from train_cfg ----------------------------------------------
        ssl_cfg = train_cfg.get("ssl", None)
        use_ssl = ssl_cfg is not None and bool(ssl_cfg.get("enabled", False))

        if use_ssl:
            trainer = SSLDeepHitTrainer(
                model=model,
                optimizer=optimizer,
                discretizer=discretizer,
                device=device,
                alpha=alpha,
                beta=beta,
                metrics_horizon=metrics_horizon,
                event_of_interest=event_of_interest,
                metrics_horizons=metrics_horizons,
                ssl_cfg=ssl_cfg,
            )
        else:
            trainer = DeepHitTrainer(
                model,
                optimizer,
                discretizer,
                device=device,
                alpha=alpha,
                beta=beta,
                metrics_horizon=metrics_horizon,
                event_of_interest=event_of_interest,
                metrics_horizons=metrics_horizons,
            )
        # ---------------------------------------------------------------------------
        def forward_fn(m, batch):
            probs, logits = m(batch["x"])  # DeepHit returns (probs, logits)
            return probs, logits

        return TrainingObjects(model, trainer, forward_fn)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")



# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def run_experiment(cfg: Dict[str, Any]):
    exp_name = cfg.get("experiment_name", "experiment")
    run_dir = os.path.join("runs", exp_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # seed
    seed = int(cfg.get("train", {}).get("seed", 42))
    deterministic = bool(cfg.get("train", {}).get("deterministic", False))
    set_all_seeds(seed, deterministic=deterministic)

    train_cfg = cfg.get("train", {})
    ssl_cfg = train_cfg.get("ssl", {})
    model_type = cfg["model"]["type"].lower()

    # data
    datasets = build_datasets(cfg)
    loaders = build_dataloaders(datasets, cfg["dataset"].get("loader", {}))

    # SSL: build labeled / unlabeled train loaders
    train_labeled_loader, train_unlabeled_loader = build_ssl_train_loaders(
        cfg, datasets, loaders
    )

    graph_cfg = train_cfg.get("graph_ssl", {})
    use_graph_ssl = bool(graph_cfg.get("enabled", False)) and (model_type == "deepsurv")

    graph = None
    if use_graph_ssl:
        ds_name = cfg["dataset"]["name"]

        # disable for multimodal
        if ds_name == "mimiceye_multimodal":
            use_graph_ssl = False
        else:
            k = int(graph_cfg.get("k", 10))

            if ds_name == "mimiceye_tabular":
                # cat indices used by EmbeddedCoxPH (slice from x)
                cat_feature_indices, _ = get_cat_info_for_mimiceye(
                    model_cfg=cfg["model"],
                    feature_cols=datasets.train.feature_cols,
                )

                # build kNN on numeric-only view of x (drop categorical-ID columns)
                X_full = datasets.train.x  # np.ndarray [N, D]
                X_knn = drop_columns_np(X_full, cat_feature_indices)

                graph = load_or_build_knn_graph(
                    base_dir=cfg["dataset"]["base_dir"],
                    split="train",
                    k=k,
                    X=X_knn,
                    symmetric=bool(graph_cfg.get("symmetric", False)),
                    metric=str(graph_cfg.get("metric", "euclidean")),
                )
            else:
                graph = load_or_build_knn_graph(
                    base_dir=cfg["dataset"]["base_dir"],
                    split="train",
                    k=k,
                    dataset=datasets.train,
                    feature_key="x",
                    symmetric=bool(graph_cfg.get("symmetric", False)),
                    metric=str(graph_cfg.get("metric", "euclidean")),
                )
            if graph is not None:
                degs = [int(n.numel()) for n in graph.neighbors]
                degs_sorted = sorted(degs)
                n = len(degs_sorted)
                p50 = degs_sorted[n // 2]
                p90 = degs_sorted[int(0.9 * (n - 1))]
                print(
                    f"[GRAPH] nodes={graph.num_nodes} k_cfg={k} "
                    f"deg(min/med/p90/max)={min(degs)}/{p50}/{p90}/{max(degs)}"
                )

    # model + trainer + forward_fn
    objs = build_training_objects(cfg, datasets, graph=graph)
    model, trainer, forward_fn = objs.model, objs.trainer, objs.forward_fn

    epochs = int(cfg["train"].get("epochs", 50))

    # early stopping hyperparameters
    patience = int(train_cfg.get("early_stopping_patience", 30))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))

    ssl_enabled_cfg = bool(ssl_cfg.get("enabled", False))
    use_ssl = (
        ssl_enabled_cfg
        and model_type == "deephit"
        and isinstance(trainer, SSLDeepHitTrainer)
        and (train_unlabeled_loader is not None)
    )

    print(f"Starting experiment: {exp_name}")
    print(f"  Dataset: {cfg['dataset']['name']}")
    print(f"  Model:   {cfg['model']['type']}")
    print(f"  Epochs:  {epochs}")
    print(f"  Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"  Seed:    {seed} (deterministic={deterministic})")

    # logging: save learning curves to CSV
    metrics_path = os.path.join(run_dir, f"{exp_name}_metrics.csv")
    header = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_c_index",
        "val_td_auc",
        "val_td_auc_cases",
        "val_td_auc_ctrls",
        "val_iauc",
        "val_ibs",
        "val_horizon",
        "val_event_of_interest",
    ]
    with open(metrics_path, "w") as f:
        f.write(",".join(header) + "\n")

    def _get(m: Dict[str, Any], k: str):
        v = m.get(k, float("nan"))
        return float(v) if v is not None else float("nan")

    best_val_c = -1.0
    best_state = None
    best_snapshot = None
    epochs_no_improve = 0

    # Get SSL hyperparams from trainer (already parsed from ssl_cfg)
    if use_ssl:
        warmup_epochs = int(trainer.warmup_epochs)
        lambda_pseudo = float(trainer.lambda_pseudo)
        print(
            f"[SSL] Using SSLDeepHitTrainer with warmup_epochs={warmup_epochs}, "
            f"ssl_epochs={trainer.ssl_epochs}, "
            f"confidence_threshold_start={trainer.conf_thresh_start}, "
            f"confidence_threshold_end={trainer.conf_thresh_end}, "
            f"lambda_pseudo={lambda_pseudo}"
        )
    else:
        warmup_epochs = 0
        lambda_pseudo = 0.0

    pseudo_loader = None  # will be built lazily on first SSL epoch
    pseudo_refresh_interval = int(ssl_cfg.get("pseudo_refresh_interval", 10))
    pseudo_refresh_min_size = int(ssl_cfg.get("pseudo_refresh_min_size", 0))

    def _should_refresh(epoch: int) -> bool:
        if pseudo_refresh_interval <= 0:
            return False
        # refresh on the first SSL epoch and then every N epochs
        first_ssl_epoch = warmup_epochs + 1
        return (epoch == first_ssl_epoch) or ((epoch - first_ssl_epoch) % pseudo_refresh_interval == 0)

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN STEP ----------------
        # Priority:
        # 1) DeepHit SSL (pseudo-labeling)
        # 2) Cox graph-SSL (graph consistency)
        # 3) Plain supervised
        if use_ssl:
            if epoch <= warmup_epochs:
                # Supervised warmup on labeled subset
                train_loss = float(trainer.train_epoch(train_labeled_loader, forward_fn))
            else:
                    # refresh pseudo labels periodically
                if _should_refresh(epoch):
                    thr_now = trainer.current_conf_threshold()
                    print(
                        f"[SSL] Refreshing pseudo-labels at epoch {epoch} "
                        f"(thr={thr_now:.4f}, step={trainer._ssl_step}/{trainer.conf_schedule_steps-1}, T={trainer.temperature})..."
                    )
                    pseudo_dataset = trainer.generate_pseudo_dataset(train_unlabeled_loader, forward_fn)

                    if len(pseudo_dataset) == 0:
                        print("[SSL] No pseudo-labels met threshold on refresh.")
                        if pseudo_loader is None:
                            print("[SSL] No pseudo set available; falling back to supervised-only this epoch.")
                            train_loss = float(trainer.train_epoch(train_labeled_loader, forward_fn))
                            # do NOT disable SSL permanently; next refresh might succeed
                            val = trainer.eval_epoch(loaders.val, forward_fn)
                            # continue to logging etc. (or just let it flow)
                        else:
                            print("[SSL] Keeping previous pseudo set.")
                    else:
                        if len(pseudo_dataset) < pseudo_refresh_min_size and (pseudo_loader is not None):
                            print(
                                f"[SSL] Refreshed pseudo set too small ({len(pseudo_dataset)} < {pseudo_refresh_min_size}); "
                                "keeping previous pseudo set."
                            )
                        else:
                            print(f"[SSL] Using refreshed pseudo set: {len(pseudo_dataset)} samples")
                            pseudo_loader = DataLoader(
                                pseudo_dataset,
                                batch_size=train_labeled_loader.batch_size,
                                shuffle=True,
                                num_workers=0,
                            )

                # train with current pseudo set if available
                if pseudo_loader is None:
                    train_loss = float(trainer.train_epoch(train_labeled_loader, forward_fn))
                else:
                    train_loss = float(
                        trainer.train_ssl_epoch(
                            train_labeled_loader,
                            pseudo_loader,
                            forward_fn,
                            lambda_pseudo=lambda_pseudo,
                        )
                    )
        elif graph is not None and isinstance(trainer, GraphCoxTrainer):
            # Cox graph-SSL: ensure always provide an "unlabeled" loader to build the teacher cache.
            # If not simulating label budget (unlabeled_fraction=0), we just reuse the labeled loader.
            unl = train_unlabeled_loader if train_unlabeled_loader is not None else train_labeled_loader
            train_loss = float(trainer.train_epoch(train_labeled_loader, unl, forward_fn))
        else:
            # Plain supervised training
            train_loss = float(trainer.train_epoch(train_labeled_loader, forward_fn))

        val = trainer.eval_epoch(loaders.val, forward_fn)
        val_loss = _get(val, "loss")
        val_c = _get(val, "c_index")
        val_td_auc = _get(val, "td_auc")
        val_cases = _get(val, "td_auc_cases")
        val_ctrls = _get(val, "td_auc_ctrls")
        val_iauc = _get(val, "iauc")
        val_ibs = _get(val, "ibs")
        val_h = _get(val, "horizon")
        val_eoi = _get(val, "event_of_interest")

        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_c:.6f},"
                f"{val_td_auc:.6f},{val_cases:.0f},{val_ctrls:.0f},"
                f"{val_iauc:.6f},{val_ibs:.6f},{val_h:.6f},{val_eoi:.0f}\n"
            )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_c_index={val_c:.4f} "
            f"val_td_auc={val_td_auc:.4f} "
            f"val_iauc={val_iauc:.4f} "
            f"val_ibs={val_ibs:.4f}"
        )
        # --- early stopping on val_c_index (maximization) ---
        improved = (not np.isnan(val_c)) and (val_c > best_val_c + min_delta)

        if improved:
            best_val_c = val_c
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_c_index": best_val_c,
                "seed": int(seed),
            }
            best_snapshot = {
                "seed": int(seed),
                "best_epoch": int(epoch),
                "val_loss": float(val_loss),
                "val_c_index": float(val_c),
                "val_td_auc": float(val_td_auc),
                "val_iauc": float(val_iauc),
                "val_ibs": float(val_ibs),
                "val_horizon": float(val_h),
                "val_event_of_interest": float(val_eoi),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {patience} epochs)."
            )
            break

    # final test evaluation with best model (if available)
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        print(
            f"Loaded best model from epoch {best_state['epoch']} "
            f"(val_c_index={best_state['best_val_c_index']:.4f})"
        )
    else:
        print("Warning: no valid best_state found, using last-epoch model.")
        # Create a fallback snapshot so multi-seed aggregation doesn't crash
        best_snapshot = {
            "seed": int(seed),
            "best_epoch": int(epoch),
            "val_loss": float("nan"),
            "val_c_index": float("nan"),
            "val_td_auc": float("nan"),
            "val_iauc": float("nan"),
            "val_ibs": float("nan"),
            "val_horizon": float(cfg.get("metrics", {}).get("horizon", float("nan"))),
            "val_event_of_interest": float(cfg.get("metrics", {}).get("event_of_interest", float("nan"))),
        }

    # test evaluation
    test = trainer.eval_epoch(loaders.test, forward_fn)
    test_metrics = {
        "seed": int(seed),
        "test_loss": _get(test, "loss"),
        "test_c_index": _get(test, "c_index"),
        "test_td_auc": _get(test, "td_auc"),
        "test_iauc": _get(test, "iauc"),
        "test_ibs": _get(test, "ibs"),
        "test_horizon": _get(test, "horizon"),
        "test_event_of_interest": _get(test, "event_of_interest"),
    }

    print("[TEST RESULTS]")
    print(f"  loss    : {test_metrics['test_loss']:.4f}")
    print(f"  c_index : {test_metrics['test_c_index']:.4f}")
    print(f"  td_auc  : {test_metrics['test_td_auc']:.4f}")
    print(f"  iauc    : {test_metrics['test_iauc']:.4f}")
    print(f"  ibs     : {test_metrics['test_ibs']:.4f}")

    # save checkpoint
    ckpt_path = os.path.join(run_dir, f"{exp_name}_best.pt")
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint to: {ckpt_path}")

    # save best + test metrics for this seed
    with open(os.path.join(run_dir, "best_val_metrics.json"), "w") as f:
        json.dump(best_snapshot, f, indent=2)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)




# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file, e.g. configs/support2_deepsurv.yaml",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override train.seed from YAML (useful for Slurm array jobs).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("train", {})

    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
        base_name = cfg.get("experiment_name", "experiment")
        cfg["experiment_name"] = f"{base_name}_seed{int(args.seed)}"

    run_experiment(cfg)

if __name__ == "__main__":
    main()
