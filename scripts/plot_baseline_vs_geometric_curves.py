#!/usr/bin/env python3
"""Plot method-comparison curves for the experiment report.

Each training plot compares experiment variants on one metric. Missing variants
are skipped, so the same script can be reused after StyleGAN/Diffusion runs.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures" / "learning_curves"
TEST_FIGURE_DIR = ROOT / "figures" / "test_metrics"
PER_CLASS_FIGURE_DIR = ROOT / "figures" / "per_class"
MAX_EPOCH = 15

MODELS = {
    "densenet121": "DenseNet121",
    "swin_tiny": "Swin Tiny",
}

VARIANTS = [
    ("Baseline", ROOT / "outputs" / "advanced_baselines_full", "#1f77b4"),
    ("Aug. geometryczna", ROOT / "outputs" / "geometric_aug_train_3000_full15", "#d62728"),
    ("StyleGAN-ADA", ROOT / "outputs" / "stylegan2_all_minority_256px_aug_train_full15", "#2ca02c"),
    ("LDM/Diffusion", ROOT / "outputs" / "diffusion_aug_train_full15", "#9467bd"),
]

TRAINING_METRICS = [
    ("train_loss", "Train loss", "Loss", "train_loss"),
    ("val_loss", "Validation loss", "Loss", "val_loss"),
    ("train_macro_f1", "Train macro F1", "Macro F1", "train_macro_f1"),
    ("val_macro_f1", "Validation macro F1", "Macro F1", "val_macro_f1"),
    ("val_accuracy", "Validation accuracy", "Accuracy", "val_accuracy"),
    ("val_balanced_accuracy", "Validation balanced accuracy", "Balanced accuracy", "val_balanced_accuracy"),
    ("val_macro_recall", "Validation macro recall", "Macro recall", "val_macro_recall"),
]

TEST_METRICS = [
    ("test_accuracy", "Test accuracy", "Accuracy", "test_accuracy"),
    ("test_balanced_accuracy", "Test balanced accuracy", "Balanced accuracy", "test_balanced_accuracy"),
    ("test_macro_f1", "Test macro F1", "Macro F1", "test_macro_f1"),
    ("test_macro_recall", "Test macro recall", "Macro recall", "test_macro_recall"),
]

CLASS_ORDER = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

RARE_CLASSES = [
    "Hernia",
    "Pneumonia",
    "Edema",
    "Fibrosis",
    "Emphysema",
    "Cardiomegaly",
    "Pleural_Thickening",
]


def available_variants() -> list[tuple[str, Path, str]]:
    return [variant for variant in VARIANTS if variant[1].exists()]


def read_history(root: Path, model: str) -> pd.DataFrame | None:
    path = root / f"{model}_last_blocks" / "history.csv"
    if not path.exists():
        return None
    history = pd.read_csv(path)
    history = history[history["epoch"] <= MAX_EPOCH].copy()
    return history.sort_values("epoch")


def read_test_summary(root: Path, model: str) -> pd.Series | None:
    path = root / "test_summary.csv"
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    rows = summary[summary["model"] == model]
    if rows.empty:
        return None
    return rows.iloc[0]


def read_classification_report(root: Path, model: str) -> pd.DataFrame | None:
    path = root / f"{model}_last_blocks" / "test_classification_report.csv"
    if not path.exists():
        return None
    report = pd.read_csv(path, index_col=0)
    report = report.loc[[label for label in CLASS_ORDER if label in report.index]].copy()
    return report


def style_epoch_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_xlabel("Epoka")
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(range(1, MAX_EPOCH + 1)))
    axis.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_metric(model: str, display_name: str, metric: tuple[str, str, str, str]) -> Path | None:
    suffix, title, ylabel, column = metric
    fig, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    plotted = 0

    for variant_name, root, color in available_variants():
        history = read_history(root, model)
        if history is None or column not in history:
            continue
        axis.plot(
            history["epoch"],
            history[column],
            color=color,
            linewidth=2.2,
            label=variant_name,
        )
        plotted += 1

    if plotted < 2:
        plt.close(fig)
        return None

    axis.set_title(f"{display_name}: {title}, epoki 1-{MAX_EPOCH}")
    style_epoch_axis(axis, ylabel)
    axis.legend(fontsize=8)
    return save_figure(fig, FIGURE_DIR / f"{model}_methods_{suffix}.png")


def plot_test_metric(model: str, display_name: str, metric: tuple[str, str, str, str]) -> Path | None:
    suffix, title, ylabel, column = metric
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for variant_name, root, color in available_variants():
        row = read_test_summary(root, model)
        if row is None or column not in row:
            continue
        labels.append(variant_name)
        values.append(float(row[column]))
        colors.append(color)

    if len(values) < 2:
        return None

    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors, width=0.62)
    axis.set_title(f"{display_name}: {title}")
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_ylim(0, max(values) * 1.18)
    axis.tick_params(axis="x", labelrotation=12)
    return save_figure(fig, TEST_FIGURE_DIR / f"{model}_{suffix}_comparison.png")


def plot_per_class_metric(
    model: str,
    display_name: str,
    metric_column: str,
    title: str,
    classes: list[str],
    output_name: str,
) -> Path | None:
    reports: list[tuple[str, pd.DataFrame, str]] = []
    for variant_name, root, color in available_variants():
        report = read_classification_report(root, model)
        if report is None or metric_column not in report:
            continue
        reports.append((variant_name, report, color))

    if len(reports) < 2:
        return None

    labels = [label for label in classes if all(label in report.index for _, report, _ in reports)]
    if not labels:
        return None

    x = range(len(labels))
    width = min(0.22, 0.72 / len(reports))
    offsets = [(i - (len(reports) - 1) / 2) * width for i in range(len(reports))]

    fig_width = 12.5 if len(labels) > 8 else 8.2
    fig, axis = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    for offset, (variant_name, report, color) in zip(offsets, reports):
        values = [float(report.loc[label, metric_column]) for label in labels]
        axis.bar([pos + offset for pos in x], values, width=width, color=color, label=variant_name)

    axis.set_title(f"{display_name}: {title}")
    axis.set_ylabel(title)
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8)
    axis.set_ylim(0, min(1.0, axis.get_ylim()[1] * 1.1))
    return save_figure(fig, PER_CLASS_FIGURE_DIR / output_name)


def main() -> None:
    output_paths: list[Path] = []

    for model, display_name in MODELS.items():
        for metric in TRAINING_METRICS:
            output_path = plot_training_metric(model, display_name, metric)
            if output_path is not None:
                output_paths.append(output_path)
        for metric in TEST_METRICS:
            output_path = plot_test_metric(model, display_name, metric)
            if output_path is not None:
                output_paths.append(output_path)
        output_path = plot_per_class_metric(
            model,
            display_name,
            "f1-score",
            "F1 per klasa",
            CLASS_ORDER,
            f"{model}_per_class_f1_comparison.png",
        )
        if output_path is not None:
            output_paths.append(output_path)
        output_path = plot_per_class_metric(
            model,
            display_name,
            "recall",
            "Recall dla rzadkich klas",
            RARE_CLASSES,
            f"{model}_rare_classes_recall_comparison.png",
        )
        if output_path is not None:
            output_paths.append(output_path)

    for output_path in output_paths:
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
