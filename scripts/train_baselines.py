#!/usr/bin/env python3
"""
Baseline benchmarks: NIH ChestX-ray14 full single-label subset.

Trains DenseNet121 and Swin-Tiny with frozen backbone + fine-tuned last blocks.
Optimised for 16 GB VRAM (e.g. RTX 5060 Ti / RTX 4080).

Usage:
    python scripts/train_baselines.py
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

try:
    import timm
except ImportError:
    timm = None
    print("timm is not installed – Swin-Tiny will be skipped.  pip install timm")


# ── 1. Configuration ────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    data_dir: Path = ROOT_DIR / "data_single_label_full"
    output_dir: Path = ROOT_DIR / "outputs" / "advanced_baselines_full"

    # ── VRAM-optimised for 16 GB ────────────────────────────────────────
    image_size: int = 512                 # 1024 → 512  (fits 16 GB with AMP)
    batch_size: int = 64                  # default; overridden per model below
    accumulation_steps: int = 1           # default; overridden per model below
    num_workers: int = 4                  # 8   → 4     (less host RAM)

    epochs: int = 50
    seed: int = 42

    # Transfer-learning setup – no image augmentation for the baseline.
    fine_tune_mode: str = "last_blocks"   # head_only | last_blocks | full
    use_class_weights: bool = True
    label_smoothing: float = 0.0

    # Differential LR: small for pre-trained backbone, larger for head.
    backbone_lr: float = 5e-5
    head_lr: float = 3e-4
    weight_decay: float = 1e-4

    # Training stabilisers.
    use_amp: bool = True
    grad_clip_norm: float | None = 1.0
    monitor_metric: str = "val_macro_f1"
    min_delta: float = 1e-4
    early_stopping_patience: int = 6

    # Scheduler.
    scheduler: str = "reduce_on_plateau"  # reduce_on_plateau | cosine | none
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Debug: set to a small int to run only a few batches per epoch.
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


# Per-model VRAM-tuned batch sizes for 16 GB GPU.
# DenseNet121 is lightweight → large batch.  Swin-Tiny heavier → medium batch.
MODEL_CONFIGS: dict[str, dict] = {
    "densenet121": {"batch_size": 128, "accumulation_steps": 1},   # ~8 GB est.
    "swin_tiny":   {"batch_size":  64, "accumulation_steps": 1},   # ~8 GB est.
}


# ── 2. Helpers ───────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ── 3. Patient-aware split ───────────────────────────────────────────────────

def make_patient_split(
    data: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/20 train+val / test, then 80/20 train / val, grouped by patient_id."""
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(
            sgkf.split(data, y=data["target"], groups=data["patient_id"])
        )
        train_val = data.iloc[train_val_idx].copy()
        test = data.iloc[test_idx].copy()

        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
        train_idx, val_idx = next(
            sgkf_val.split(train_val, y=train_val["target"], groups=train_val["patient_id"])
        )
        train = train_val.iloc[train_idx].copy()
        val = train_val.iloc[val_idx].copy()
        return train, val, test
    except Exception as exc:
        print("StratifiedGroupKFold failed, falling back to GroupShuffleSplit:", repr(exc))
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(data, groups=data["patient_id"]))
        train_val = data.iloc[train_val_idx].copy()
        test = data.iloc[test_idx].copy()
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
        train_idx, val_idx = next(splitter_val.split(train_val, groups=train_val["patient_id"]))
        train = train_val.iloc[train_idx].copy()
        val = train_val.iloc[val_idx].copy()
        return train, val, test


# ── 4. Dataset ───────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = int(row["target"])
        return image, target


# ── 5. Model builders ───────────────────────────────────────────────────────

def set_trainable(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def build_densenet121(num_classes: int, fine_tune_mode: str) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    if fine_tune_mode == "head_only":
        set_trainable(model, False)
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif fine_tune_mode == "last_blocks":
        set_trainable(model, False)
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in ("features.denseblock4", "features.norm5", "classifier")):
                param.requires_grad = True
    elif fine_tune_mode == "full":
        set_trainable(model, True)
    else:
        raise ValueError(f"Unknown fine_tune_mode: {fine_tune_mode}")
    return model


def build_swin_tiny(num_classes: int, fine_tune_mode: str, image_size: int = 224) -> nn.Module:
    if timm is None:
        raise ImportError("timm is required for Swin-Tiny")
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
        img_size=image_size,
    )

    if fine_tune_mode == "head_only":
        set_trainable(model, False)
        for name, param in model.named_parameters():
            if name.startswith("head"):
                param.requires_grad = True
    elif fine_tune_mode == "last_blocks":
        set_trainable(model, False)
        for name, param in model.named_parameters():
            if name.startswith("head") or name.startswith("norm") or name.startswith("layers.3") or name.startswith("stages.3"):
                param.requires_grad = True
    elif fine_tune_mode == "full":
        set_trainable(model, True)
    else:
        raise ValueError(f"Unknown fine_tune_mode: {fine_tune_mode}")
    return model


def build_model(model_name: str, num_classes: int, fine_tune_mode: str, image_size: int = 224) -> nn.Module:
    if model_name == "densenet121":
        return build_densenet121(num_classes, fine_tune_mode)
    if model_name == "swin_tiny":
        return build_swin_tiny(num_classes, fine_tune_mode, image_size=image_size)
    raise ValueError(f"Unknown model: {model_name}")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_parameter_prefixes(model: nn.Module, max_items: int = 20) -> list[str]:
    prefixes: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parts = name.split(".")
            prefix = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            if prefix not in prefixes:
                prefixes.append(prefix)
    return prefixes[:max_items]


# ── 6. Optimizer / Scheduler / Scaler ────────────────────────────────────────

def make_class_weights(
    train_frame: pd.DataFrame, num_classes: int
) -> torch.Tensor:
    counts = (
        train_frame["target"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .sort_index()
    )
    weights = len(train_frame) / (num_classes * counts.clip(lower=1))
    return torch.tensor(weights.values, dtype=torch.float32)


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    head_params: list[torch.Tensor] = []
    backbone_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier") or name.startswith("head"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups: list[dict] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.head_lr, "name": "head"})
    if not param_groups:
        raise ValueError("No trainable parameters found.")

    print("optimizer groups:", [(g["name"], g["lr"], len(g["params"])) for g in param_groups])
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_lr,
        )
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.min_lr,
        )
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        group.get("name", f"group_{idx}"): group["lr"]
        for idx, group in enumerate(optimizer.param_groups)
    }


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ── 7. Training / evaluation loops ──────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    max_batches: int | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    accum = cfg.accumulation_steps if training else 1

    progress = tqdm(loader, leave=False, desc="train" if training else "eval")
    for batch_idx, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
                if training:
                    loss = loss / accum          # scale for accumulation

        if training:
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(loader):
                if cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum * images.size(0)   # undo scaling for logging
        total_samples += images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(targets.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        progress.set_postfix(loss=loss.item() * accum)

    avg_loss = total_loss / max(1, total_samples)
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> tuple[list[int], list[int]]:
    model.eval()
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in tqdm(loader, leave=False, desc="predict"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        y_true.extend(targets.numpy().tolist())
        y_pred.extend(preds)
    return y_true, y_pred


# ── 8. Full training run for one model ───────────────────────────────────────

def train_model(
    model_name: str,
    num_classes: int,
    classes: list[str],
    label_to_idx: dict[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor | None,
    device: torch.device,
    cfg: Config,
) -> tuple[nn.Module, pd.DataFrame]:
    seed_everything(cfg.seed)
    run_name = f"{model_name}_{cfg.fine_tune_mode}"
    model_dir = cfg.output_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, num_classes, cfg.fine_tune_mode, image_size=cfg.image_size).to(device)
    print(f"{model_name}  fine_tune_mode={cfg.fine_tune_mode}")
    print(f"  trainable params : {count_trainable_params(model):,}")
    print(f"  trainable groups : {trainable_parameter_prefixes(model)}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = make_optimizer(model, cfg)
    optimizer.zero_grad(set_to_none=True)        # init for gradient accumulation
    scheduler = make_scheduler(optimizer, cfg)
    scaler = make_grad_scaler(enabled=bool(cfg.use_amp and device.type == "cuda"))

    best_metric = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, cfg,
            optimizer=optimizer, scaler=scaler,
            max_batches=cfg.max_train_batches,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, cfg,
            optimizer=None, scaler=None,
            max_batches=cfg.max_eval_batches,
        )
        elapsed = time.time() - start

        row = {
            "model": model_name,
            "run_name": run_name,
            "fine_tune_mode": cfg.fine_tune_mode,
            "epoch": epoch,
            "seconds": elapsed,
            "lr_backbone": current_lrs(optimizer).get("backbone"),
            "lr_head": current_lrs(optimizer).get("head"),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_macro_recall": train_metrics["macro_recall"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)

        monitor_value = row[cfg.monitor_metric]
        improved = monitor_value > best_metric + cfg.min_delta

        tag = " ★" if improved else ""
        print(
            f"  [{epoch:02d}/{cfg.epochs}]  "
            f"train_loss={row['train_loss']:.4f}  "
            f"val_f1={row['val_macro_f1']:.4f}  "
            f"val_bal_acc={row['val_balanced_accuracy']:.4f}  "
            f"{elapsed:.0f}s{tag}"
        )

        if improved:
            best_metric = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": model_name,
                    "run_name": run_name,
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                    "epoch": epoch,
                    "monitor_metric": cfg.monitor_metric,
                    "best_metric": best_metric,
                    "val_metrics": {k: v for k, v in row.items() if k.startswith("val_")},
                },
                model_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if cfg.scheduler == "reduce_on_plateau":
                scheduler.step(monitor_value)
            else:
                scheduler.step()

        # Save incremental history
        pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"  Early stopping at epoch {epoch}. "
                f"Best {cfg.monitor_metric}={best_metric:.4f} at epoch {best_epoch}."
            )
            break

    # Save last checkpoint
    torch.save(
        {
            "model_name": model_name,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "epoch": history[-1]["epoch"],
        },
        model_dir / "last.pt",
    )

    best_summary = {
        "model": model_name,
        "run_name": run_name,
        "best_epoch": best_epoch,
        "monitor_metric": cfg.monitor_metric,
        "best_metric": best_metric,
    }
    (model_dir / "best_summary.json").write_text(json.dumps(best_summary, indent=2))
    return model, pd.DataFrame(history)


# ── 9. Test-set evaluation ───────────────────────────────────────────────────

def evaluate_on_test(
    model_name: str,
    num_classes: int,
    classes: list[str],
    label_to_idx: dict[str, int],
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
):
    run_name = f"{model_name}_{cfg.fine_tune_mode}"
    ckpt_path = cfg.output_dir / run_name / "best.pt"
    if not ckpt_path.exists():
        print(f"  Skipping test eval – checkpoint missing: {ckpt_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_cfg = checkpoint.get("config", {})
    fine_tune_mode = checkpoint_cfg.get("fine_tune_mode", cfg.fine_tune_mode)
    model = build_model(model_name, num_classes, fine_tune_mode, image_size=int(checkpoint_cfg.get('image_size', 224))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = predict(model, test_loader, device, cfg)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    out_dir = cfg.output_dir / run_name
    report_df.to_csv(out_dir / "test_classification_report.csv")

    pred_df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    summary = {
        "model": model_name,
        "run_name": run_name,
        "best_epoch": checkpoint.get("epoch"),
        "best_val_metric": checkpoint.get("best_metric"),
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    print(f"  TEST  acc={summary['test_accuracy']:.4f}  "
          f"bal_acc={summary['test_balanced_accuracy']:.4f}  "
          f"f1={summary['test_macro_f1']:.4f}")
    return summary


# ── 10. Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device        : {device}")
    print(f"image_size    : {cfg.image_size}")
    print(f"batch_size    : {cfg.batch_size}")
    print(f"accum_steps   : {cfg.accumulation_steps}")
    print(f"effective_bs  : {cfg.batch_size * cfg.accumulation_steps}")
    print(f"use_amp       : {cfg.use_amp}")

    labels_csv = cfg.data_dir / "labels.csv"
    images_dir = cfg.data_dir / "images"
    print(f"labels_csv    : {labels_csv.resolve()}")
    print(f"images_dir    : {images_dir.resolve()}")
    print(f"output_dir    : {cfg.output_dir.resolve()}")

    seed_everything(cfg.seed)

    # ── Load labels ──────────────────────────────────────────────────────
    df = pd.read_csv(labels_csv)
    df["image_path"] = df["image"].map(lambda name: images_dir / name)
    df["exists"] = df["image_path"].map(lambda p: p.exists())
    print(f"rows          : {len(df)}")
    print(f"images on disk: {df['exists'].sum()}")
    assert df["exists"].all(), "Some images listed in labels.csv are missing."

    classes = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    df["target"] = df["label"].map(label_to_idx)
    num_classes = len(classes)
    print(f"num_classes   : {num_classes}")

    with (cfg.output_dir / "label_mapping.json").open("w") as f:
        json.dump(label_to_idx, f, indent=2)

    # ── Patient split ────────────────────────────────────────────────────
    train_df, val_df, test_df = make_patient_split(df, cfg.seed)
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:5s}: {len(sdf):6d} samples, {sdf['patient_id'].nunique():5d} patients")
    assert set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"]))
    assert set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))
    assert set(val_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))

    train_df.to_csv(cfg.output_dir / "train_split.csv", index=False)
    val_df.to_csv(cfg.output_dir / "val_split.csv", index=False)
    test_df.to_csv(cfg.output_dir / "test_split.csv", index=False)

    # ── Transforms (no augmentation for baseline) ────────────────────────
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ChestXrayDataset(train_df, transform)
    val_ds = ChestXrayDataset(val_df, transform)
    test_ds = ChestXrayDataset(test_df, transform)

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)

    print(f"  train batches: {len(train_loader)}")
    print(f"  val   batches: {len(val_loader)}")
    print(f"  test  batches: {len(test_loader)}")

    # ── Class weights ────────────────────────────────────────────────────
    class_weights = (
        make_class_weights(train_df, num_classes).to(device)
        if cfg.use_class_weights else None
    )

    # ── Models to train ──────────────────────────────────────────────────
    models_to_run = ["densenet121"]
    if timm is not None:
        models_to_run.append("swin_tiny")
    else:
        print("⚠  timm not installed, skipping swin_tiny")

    histories: list[pd.DataFrame] = []
    for model_name in models_to_run:
        # Apply per-model batch-size overrides
        overrides = MODEL_CONFIGS.get(model_name, {})
        model_cfg = Config(**{**asdict(cfg), **overrides})

        m_train_loader = DataLoader(
            train_ds, batch_size=model_cfg.batch_size, shuffle=True, **loader_kwargs
        )
        m_val_loader = DataLoader(
            val_ds, batch_size=model_cfg.batch_size, shuffle=False, **loader_kwargs
        )

        print("=" * 72)
        print(f"Training: {model_name}  (batch_size={model_cfg.batch_size}, "
              f"accum={model_cfg.accumulation_steps}, "
              f"effective_bs={model_cfg.batch_size * model_cfg.accumulation_steps})")

        model, history_df = train_model(
            model_name, num_classes, classes, label_to_idx,
            m_train_loader, m_val_loader, class_weights, device, model_cfg,
        )
        histories.append(history_df)
        del model
        torch.cuda.empty_cache()

    all_history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    all_history.to_csv(cfg.output_dir / "all_history.csv", index=False)

    # ── Test evaluation ──────────────────────────────────────────────────
    print("=" * 72)
    print("Test-set evaluation")
    test_summaries = []
    m_test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs
    )
    for model_name in models_to_run:
        summary = evaluate_on_test(
            model_name, num_classes, classes, label_to_idx,
            m_test_loader, device, cfg,
        )
        if summary is not None:
            test_summaries.append(summary)

    if test_summaries:
        summary_df = pd.DataFrame(test_summaries)
        summary_df.to_csv(cfg.output_dir / "test_summary.csv", index=False)
        print("\n", summary_df.to_string(index=False))

    print("\nDone. Results saved to:", cfg.output_dir.resolve())


if __name__ == "__main__":
    main()
