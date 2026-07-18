#!/usr/bin/env python3
"""汇总 CNN、神经风格迁移与拓扑审查结果，并生成可提交的定量图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PROJECT_TITLE = "Entering Las Meninas: How a Painting Changed the Way I Look at Art"
HEX_SUFFIX_PATTERN = re.compile(r"-[0-9a-f]{10}$")

# Okabe–Ito 色盲友好色板。类别还同时使用不同点形与线型，避免只依赖颜色。
CATEGORY_STYLES: dict[str, dict[str, Any]] = {
    "deterministic-style": {"color": "#0072B2", "marker": "o", "label": "Deterministic style"},
    "geometry": {"color": "#E69F00", "marker": "s", "label": "Geometry"},
    "topology": {"color": "#6B7280", "marker": "D", "label": "Topology controls"},
    "neural-style": {"color": "#CC79A7", "marker": "^", "label": "Neural style"},
    "combined": {"color": "#009E73", "marker": "X", "label": "Combined"},
    "final-artwork": {"color": "#D55E00", "marker": "*", "label": "Final artwork"},
    "picasso": {"color": "#111827", "marker": "P", "label": "Picasso comparison"},
}

INK = "#18212B"
MUTED = "#5F6B76"
GRID = "#D9E0E6"
BLUE = "#0072B2"
ORANGE = "#E69F00"
PINK = "#CC79A7"
GREEN = "#009E73"
NEUTRAL = "#6B7280"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cnn-json",
        type=Path,
        default=Path("outputs/cnn-analysis/cnn-analysis.json"),
        help="CNN 汇总 JSON",
    )
    parser.add_argument(
        "--style-csv",
        type=Path,
        default=Path("outputs/cnn-analysis/style-distances.csv"),
        help="逐层风格距离 CSV",
    )
    parser.add_argument(
        "--spatial-csv",
        type=Path,
        default=Path("outputs/cnn-analysis/spatial-distances.csv"),
        help="逐层空间距离 CSV",
    )
    parser.add_argument(
        "--style-transfer-manifest",
        type=Path,
        default=Path("outputs/neural-style-transfer/manifest.json"),
        help="神经风格迁移清单",
    )
    parser.add_argument(
        "--combined-style-transfer-manifest",
        type=Path,
        default=Path("outputs/neural-style-transfer-combined/manifest.json"),
        help="以 100%% 几何变换为内容图的神经风格迁移清单",
    )
    parser.add_argument(
        "--topology-csv",
        type=Path,
        default=Path("data/topology-relations.csv"),
        help="人工拓扑关系审查 CSV",
    )
    parser.add_argument(
        "--transform-manifest",
        type=Path,
        default=Path("outputs/transformations/transformations-manifest.json"),
        help="确定性变换清单",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="结果表和图的输出目录",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG 输出分辨率")
    return parser.parse_args()


def loadJson(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def loadCsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256File(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseSampleName(candidateId: str) -> str:
    """移除 CNN 缓存键中的内容哈希，保留人可读样本名。"""

    return HEX_SUFFIX_PATTERN.sub("", candidateId)


def classifySample(candidateId: str, candidatePath: str) -> str:
    """按实验角色分类；先判断 combined，避免被 geometry 前缀误分。"""

    sampleName = baseSampleName(candidateId)
    normalizedPath = candidatePath.replace("\\", "/")
    if "combined" in normalizedPath or "combined" in sampleName:
        return "combined"
    if sampleName.startswith("style-baseline-"):
        return "deterministic-style"
    if sampleName.startswith("geometry-"):
        return "geometry"
    if sampleName.startswith("topology-"):
        return "topology"
    if sampleName.startswith("neural-style-"):
        return "neural-style"
    if sampleName.startswith("entering-las-meninas-final"):
        return "final-artwork"
    if sampleName.startswith("picasso"):
        return "picasso"
    raise ValueError(f"无法分类样本：{candidateId} ({candidatePath})")


def shortLabel(sampleName: str, category: str, candidatePath: str = "") -> str:
    replacements = {
        "style-baseline-": "Style ",
        "geometry-": "Geometry ",
        "topology-": "Topology: ",
        "neural-style-strength-": "NST ",
        "entering-las-meninas-final": "Final artwork",
        "picasso": "Picasso",
    }
    label = sampleName
    for prefix, replacement in replacements.items():
        if label.startswith(prefix):
            label = replacement + label[len(prefix) :]
            break
    label = label.replace("0p25", "0.25×").replace("0p5", "0.5×")
    if category == "neural-style" and label.endswith(" 1"):
        label += ".0×"
    if category == "combined":
        if "neural-style-transfer-combined" in candidatePath:
            return "NST 1.0× + geometry 100%"
        return "Deterministic style + geometry 100%"
    label = label.replace("-", " ")
    if category in {"deterministic-style", "geometry"}:
        label = re.sub(r"(\d{3})$", lambda match: f"{int(match.group(1))}%", label)
    return label


def extractIntensity(sampleName: str, category: str) -> float | None:
    if category in {"deterministic-style", "geometry"}:
        match = re.search(r"-(\d{3})$", sampleName)
        return float(match.group(1)) if match else None
    if category == "topology":
        return 0.0 if sampleName == "topology-original" else 100.0
    if category == "combined":
        return 100.0
    return None


def topologyVariant(sampleName: str, category: str, candidatePath: str = "") -> str | None:
    mapping = {
        "topology-original": "original",
        "topology-matched-control": "matched-control",
        "topology-mirror-deletion": "mirror-deletion",
        "geometry-100": "geometry-100",
        "neural-style-strength-1": "neural-style-1",
        "entering-las-meninas-final": "final-artwork",
    }
    if category == "combined" and "neural-style-transfer-combined" in candidatePath:
        return "combined-neural-geometry"
    return mapping.get(sampleName)


def configureMatplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def styleAxis(axis: Axes, *, gridAxis: str = "both") -> None:
    axis.grid(True, axis=gridAxis, color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AAB4BE")
    axis.spines["bottom"].set_color("#AAB4BE")


def addFigureHeader(
    figure: Figure,
    title: str,
    subtitle: str,
    *,
    footer: str,
) -> None:
    figure.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=20, fontweight="bold", color=INK)
    figure.text(0.06, 0.945, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)
    figure.text(0.06, 0.018, footer, ha="left", va="bottom", fontsize=8.3, color=MUTED)


def saveFigure(figure: Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.22,
        metadata={"Title": path.stem, "Author": "Las Meninas CNN project"},
    )
    plt.close(figure)


def validateCsvAgainstJson(
    cnnData: dict[str, Any],
    styleRows: list[dict[str, str]],
    spatialRows: list[dict[str, str]],
) -> None:
    styleLookup = {
        (row["candidate_image_id"], row["layer"]): float(row["relative_frobenius"])
        for row in styleRows
    }
    spatialLookup = {
        (row["candidate_image_id"], row["layer"]): float(row["relative_rms"])
        for row in spatialRows
    }
    for comparison in cnnData["comparisons"]:
        candidateId = comparison["candidate_image_id"]
        for layer, metrics in comparison["style"].items():
            csvValue = styleLookup[(candidateId, layer)]
            if not math.isclose(csvValue, metrics["relative_frobenius"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"style CSV 与 JSON 不一致：{candidateId}/{layer}")
        for layer, metrics in comparison["spatial"].items():
            csvValue = spatialLookup[(candidateId, layer)]
            if not math.isclose(csvValue, metrics["relative_rms"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"spatial CSV 与 JSON 不一致：{candidateId}/{layer}")


def validateZeroBaselines(cnnData: dict[str, Any]) -> None:
    requiredPaths = {
        "outputs/transformations/style/style-baseline-000.png",
        "outputs/transformations/geometry/geometry-000.png",
        "outputs/transformations/topology/topology-original.png",
    }
    observedPaths: set[str] = set()
    for comparison in cnnData["comparisons"]:
        candidatePath = comparison["candidate_path"]
        if candidatePath not in requiredPaths:
            continue
        observedPaths.add(candidatePath)
        styleValue = float(comparison["style"]["aggregate"]["relative_frobenius"])
        spatialValue = float(comparison["spatial"]["aggregate"]["relative_rms"])
        if abs(styleValue) > 1e-12 or abs(spatialValue) > 1e-12:
            raise ValueError(
                f"0% 基线没有严格对齐 working proxy：{candidatePath} "
                f"(style={styleValue}, spatial={spatialValue})"
            )
    missingPaths = requiredPaths - observedPaths
    if missingPaths:
        raise ValueError(f"CNN 输出缺少必要基线：{sorted(missingPaths)}")


def buildRows(
    cnnData: dict[str, Any],
    styleRows: list[dict[str, str]],
    spatialRows: list[dict[str, str]],
    styleTransferData: dict[str, Any],
    combinedStyleTransferData: dict[str, Any],
    topologyRows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topologyLookup = {row["variant"]: row for row in topologyRows}
    styleTransferLookup = {
        record["image"]: record
        for manifest in (styleTransferData, combinedStyleTransferData)
        for record in manifest.get("outputs", [])
    }
    imageRows: list[dict[str, Any]] = []

    for comparison in cnnData["comparisons"]:
        candidateId = comparison["candidate_image_id"]
        candidatePath = comparison["candidate_path"]
        sampleName = baseSampleName(candidateId)
        category = classifySample(candidateId, candidatePath)
        relationVariant = topologyVariant(sampleName, category, candidatePath)
        relationRecord = topologyLookup.get(relationVariant or "")
        transferRecord = styleTransferLookup.get(candidatePath)
        finalLoss = transferRecord.get("final_loss", {}) if transferRecord else {}
        imageRows.append(
            {
                "candidate_image_id": candidateId,
                "sample_name": sampleName,
                "display_label": shortLabel(sampleName, category, candidatePath),
                "category": category,
                "candidate_path": candidatePath,
                "intensity_percent": extractIntensity(sampleName, category),
                "neural_style_strength": (
                    float(transferRecord["style_strength"]) if transferRecord else None
                ),
                "nst_content_condition": (
                    "geometry-100"
                    if transferRecord and "neural-style-transfer-combined" in candidatePath
                    else "working-proxy"
                    if transferRecord
                    else None
                ),
                "style_relative_frobenius": float(
                    comparison["style"]["aggregate"]["relative_frobenius"]
                ),
                "style_absolute_frobenius": float(
                    comparison["style"]["aggregate"]["absolute_frobenius"]
                ),
                "spatial_relative_rms": float(
                    comparison["spatial"]["aggregate"]["relative_rms"]
                ),
                "spatial_mean_cosine_distance": float(
                    comparison["spatial"]["aggregate"]["mean_cosine_distance"]
                ),
                "topology_variant": relationVariant,
                "relations_preserved": (
                    int(relationRecord["relations_preserved"]) if relationRecord else None
                ),
                "total_relations": (
                    int(relationRecord["total_relations"]) if relationRecord else None
                ),
                "relation_fraction": (
                    int(relationRecord["relations_preserved"])
                    / int(relationRecord["total_relations"])
                    if relationRecord
                    else None
                ),
                "relation_review_method": (
                    relationRecord["review_method"] if relationRecord else None
                ),
                "nst_final_total_loss": (
                    float(finalLoss["total_loss"]) if "total_loss" in finalLoss else None
                ),
                "nst_final_content_loss": (
                    float(finalLoss["content_loss"]) if "content_loss" in finalLoss else None
                ),
                "nst_final_style_loss": (
                    float(finalLoss["style_loss"]) if "style_loss" in finalLoss else None
                ),
                "nst_final_tv_loss": (
                    float(finalLoss["tv_loss"]) if "tv_loss" in finalLoss else None
                ),
            }
        )

    metadataLookup = {
        row["candidate_image_id"]: {
            "sample_name": baseSampleName(row["candidate_image_id"]),
            "category": classifySample(row["candidate_image_id"], row["candidate_path"]),
            "candidate_path": row["candidate_path"],
        }
        for row in styleRows
    }
    layerRows: list[dict[str, Any]] = []
    for row in styleRows:
        metadata = metadataLookup[row["candidate_image_id"]]
        layerRows.append(
            {
                "candidate_image_id": row["candidate_image_id"],
                **metadata,
                "metric_family": "style",
                "layer": row["layer"],
                "primary_metric": "relative_frobenius",
                "primary_value": float(row["relative_frobenius"]),
                "secondary_metric": "absolute_frobenius",
                "secondary_value": float(row["absolute_frobenius"]),
            }
        )
    for row in spatialRows:
        metadata = metadataLookup[row["candidate_image_id"]]
        layerRows.append(
            {
                "candidate_image_id": row["candidate_image_id"],
                **metadata,
                "metric_family": "spatial",
                "layer": row["layer"],
                "primary_metric": "relative_rms",
                "primary_value": float(row["relative_rms"]),
                "secondary_metric": "mean_cosine_distance",
                "secondary_value": float(row["mean_cosine_distance"]),
            }
        )
    return imageRows, layerRows


def writeCsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregateCategoryMetrics(imageRows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in imageRows:
        grouped[row["category"]].append(row)
    result: dict[str, Any] = {}
    for category, rows in grouped.items():
        styleValues = [row["style_relative_frobenius"] for row in rows]
        spatialValues = [row["spatial_relative_rms"] for row in rows]
        result[category] = {
            "sample_count": len(rows),
            "style_relative_frobenius": {
                "min": min(styleValues),
                "median": float(np.median(styleValues)),
                "max": max(styleValues),
            },
            "spatial_relative_rms": {
                "min": min(spatialValues),
                "median": float(np.median(spatialValues)),
                "max": max(spatialValues),
            },
        }
    return result


def buildSummary(
    inputPaths: dict[str, Path],
    cnnData: dict[str, Any],
    styleTransferData: dict[str, Any],
    combinedStyleTransferData: dict[str, Any],
    imageRows: list[dict[str, Any]],
    layerRows: list[dict[str, Any]],
) -> dict[str, Any]:
    byName = {row["sample_name"]: row for row in imageRows}
    topologyOrder = {"original": 0, "matched-control": 1, "mirror-deletion": 2}
    topologySamples = sorted(
        (
            row
            for row in imageRows
            if row["topology_variant"] in topologyOrder
        ),
        key=lambda row: topologyOrder[row["topology_variant"]],
    )
    neuralSamples = [
        row for row in imageRows if row["neural_style_strength"] is not None
    ]

    chartContracts = {
        "style-vs-spatial-scatter.png": {
            "question": "How do experiment families differ in VGG19 style and spatial-content distance?",
            "chart": "labeled scatter",
            "grain": "one point per candidate image",
            "scales": "raw aggregate relative distances; no rescaling",
        },
        "transformation-trajectories.png": {
            "question": "How do deterministic style and geometry distances change with intensity?",
            "chart": "two-panel multi-series line",
            "grain": "five registered intensity levels per family",
            "scales": "raw aggregate relative distances; zero baseline retained",
        },
        "selected-layer-heatmap.png": {
            "question": "At which VGG19 layers do candidate images diverge from the reference?",
            "chart": "two-panel annotated heatmap",
            "grain": "candidate image × CNN layer",
            "scales": "separate raw scales for different metric definitions",
        },
        "topology-ablation-metrics.png": {
            "question": "Does mirror-relation deletion differ from pixel-matched and resampling controls?",
            "chart": "manual-audit bars plus grouped CNN-distance bars",
            "grain": "one bar group per topology control",
            "scales": "relation fraction and raw CNN distances kept in separate panels",
        },
        "neural-style-loss-curves.png": {
            "question": "How did each neural-style objective evolve during 500 Adam steps?",
            "chart": "three aligned loss-curve panels",
            "grain": "one record per optimization step and strength",
            "scales": "shared linear y-axis; raw weighted objective components",
        },
    }

    return {
        "format_version": 1,
        "project": PROJECT_TITLE,
        "source_integrity": {
            name: {"path": str(path), "sha256": sha256File(path)}
            for name, path in inputPaths.items()
        },
        "cnn_reference": {
            "path": cnnData["reference_path"],
            "image_id": cnnData["reference_image_id"],
            "model_signature": cnnData["model_signature"],
            "device": cnnData["device"],
        },
        "record_counts": {
            "candidate_images": len(imageRows),
            "layer_metric_rows": len(layerRows),
            "categories": dict(sorted(Counter(row["category"] for row in imageRows).items())),
            "neural_style_optimization_records": sum(
                int(record["loss_records"])
                for manifest in (styleTransferData, combinedStyleTransferData)
                for record in manifest.get("outputs", [])
            ),
        },
        "category_metrics": aggregateCategoryMetrics(imageRows),
        "registered_trajectory_endpoints": {
            "deterministic_style_100_percent": {
                "style_relative_frobenius": byName["style-baseline-100"][
                    "style_relative_frobenius"
                ],
                "spatial_relative_rms": byName["style-baseline-100"]["spatial_relative_rms"],
            },
            "geometry_100_percent": {
                "style_relative_frobenius": byName["geometry-100"][
                    "style_relative_frobenius"
                ],
                "spatial_relative_rms": byName["geometry-100"]["spatial_relative_rms"],
            },
        },
        "topology_ablation": [
            {
                key: row[key]
                for key in (
                    "topology_variant",
                    "relations_preserved",
                    "total_relations",
                    "relation_fraction",
                    "style_relative_frobenius",
                    "spatial_relative_rms",
                )
            }
            for row in topologySamples
        ],
        "neural_style_final_losses": [
            {
                key: row[key]
                for key in (
                    "candidate_path",
                    "nst_content_condition",
                    "neural_style_strength",
                    "nst_final_total_loss",
                    "nst_final_content_loss",
                    "nst_final_style_loss",
                    "nst_final_tv_loss",
                )
            }
            for row in neuralSamples
        ],
        "chart_contracts": chartContracts,
        "interpretation_limits": [
            "VGG19 distances are descriptive model outputs, not human aesthetic judgments.",
            "Style and spatial distances use different definitions and are never merged into one score.",
            "Topology preservation is a manual audit of five predefined relations, not a CNN prediction.",
            "The topology audit has no repeated observers or inferential statistics.",
            "Neural-style objectives differ by style-strength coefficient, so their losses are not model rankings.",
            "Picasso is included as a numerical comparison only; the source image is not redistributed.",
        ],
    }


def plotStyleSpatialScatter(imageFrame: pd.DataFrame, outputPath: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(12.8, 7.4))
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.15, top=0.86)
    addFigureHeader(
        figure,
        "VGG19 style and spatial-content distances",
        "Each point is one candidate image. Axes show raw aggregate distances from the same working-proxy reference.",
        footer="Source: outputs/cnn-analysis/cnn-analysis.json · Frozen ImageNet VGG19 · No axis normalization",
    )

    for category, categoryFrame in imageFrame.groupby("category", sort=False):
        style = CATEGORY_STYLES[category]
        markerSize = 150 if category == "final-artwork" else 76
        axis.scatter(
            categoryFrame["style_relative_frobenius"],
            categoryFrame["spatial_relative_rms"],
            s=markerSize,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            label=style["label"],
            zorder=3,
        )

    labelNames = {
        "style-baseline-100",
        "geometry-100",
        "neural-style-strength-0p25",
        "neural-style-strength-0p5",
        "neural-style-strength-1",
        "entering-las-meninas-final",
        "picasso",
    }
    for _, row in imageFrame.iterrows():
        if row["sample_name"] not in labelNames and row["category"] != "combined":
            continue
        xOffset = 7
        yOffset = 4
        axis.annotate(
            row["display_label"],
            (row["style_relative_frobenius"], row["spatial_relative_rms"]),
            xytext=(xOffset, yOffset),
            textcoords="offset points",
            fontsize=8.2,
            color=INK,
        )

    axis.set_xlabel("Style distance — aggregate relative Frobenius norm")
    axis.set_ylabel("Spatial-content distance — aggregate relative RMS")
    axis.set_xlim(left=-0.035)
    axis.set_ylim(bottom=-0.035)
    styleAxis(axis)
    axis.legend(ncol=4, loc="upper left", fontsize=8.8, handletextpad=0.4, columnspacing=1.1)

    # 主图保留完整尺度；嵌入图只放大原点附近，不移动数据点。
    inset = axis.inset_axes([0.655, 0.08, 0.30, 0.32])
    topologyFrame = imageFrame[imageFrame["category"] == "topology"]
    inset.scatter(
        topologyFrame["style_relative_frobenius"],
        topologyFrame["spatial_relative_rms"],
        s=42,
        marker=CATEGORY_STYLES["topology"]["marker"],
        color=CATEGORY_STYLES["topology"]["color"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    insetOffsets = {
        "topology-original": (4, 4),
        "topology-sham-warp": (4, -11),
        "topology-matched-control": (-4, -13),
        "topology-mirror-deletion": (-4, 5),
    }
    insetLabels = {
        "topology-original": "3 baselines at (0, 0)",
        "topology-sham-warp": "Sham warp",
        "topology-matched-control": "Matched control",
        "topology-mirror-deletion": "Mirror deletion",
    }
    for _, row in topologyFrame.iterrows():
        xOffset, yOffset = insetOffsets[row["sample_name"]]
        inset.annotate(
            insetLabels[row["sample_name"]],
            (row["style_relative_frobenius"], row["spatial_relative_rms"]),
            xytext=(xOffset, yOffset),
            textcoords="offset points",
            fontsize=6.8,
            ha="right" if xOffset < 0 else "left",
            color=INK,
        )
    inset.set_title("Origin detail · same raw axes", fontsize=8.2, loc="left", pad=4)
    inset.set_xlim(-0.003, 0.046)
    inset.set_ylim(-0.006, 0.108)
    inset.tick_params(labelsize=6.5)
    inset.grid(True, color=GRID, linewidth=0.5)
    inset.set_axisbelow(True)
    for spine in inset.spines.values():
        spine.set_color("#AAB4BE")
    saveFigure(figure, outputPath, dpi)


def plotTransformationTrajectories(imageFrame: pd.DataFrame, outputPath: Path, dpi: int) -> None:
    sequenceFrame = imageFrame[
        imageFrame["category"].isin(["deterministic-style", "geometry"])
    ].copy()
    sequenceFrame = sequenceFrame.sort_values(["category", "intensity_percent"])
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.5), sharex=True)
    figure.subplots_adjust(left=0.075, right=0.97, bottom=0.17, top=0.80, wspace=0.22)
    addFigureHeader(
        figure,
        "Registered transformation trajectories",
        "Five fixed intensity levels per family; values are raw distances from the identical 0% working-proxy baseline.",
        footer="Source: CNN aggregate descriptors · Lines connect registered levels only; they do not imply measurements between levels",
    )
    panels = [
        ("style_relative_frobenius", "Style descriptor change", "Aggregate relative Frobenius norm"),
        ("spatial_relative_rms", "Spatial feature change", "Aggregate relative RMS"),
    ]
    lineStyles = {"deterministic-style": "-", "geometry": "--"}
    for axis, (metric, title, yLabel) in zip(axes, panels, strict=True):
        for category in ["deterministic-style", "geometry"]:
            categoryFrame = sequenceFrame[sequenceFrame["category"] == category]
            style = CATEGORY_STYLES[category]
            axis.plot(
                categoryFrame["intensity_percent"],
                categoryFrame[metric],
                color=style["color"],
                linestyle=lineStyles[category],
                marker=style["marker"],
                markersize=6.5,
                linewidth=2.2,
                label=style["label"],
            )
            endpoint = categoryFrame.iloc[-1]
            axis.annotate(
                f"{endpoint[metric]:.3f}",
                (endpoint["intensity_percent"], endpoint[metric]),
                xytext=(-8, 8),
                textcoords="offset points",
                ha="right",
                fontsize=8.4,
                fontweight="bold",
                color=style["color"],
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Transformation intensity (%)")
        axis.set_ylabel(yLabel)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlim(-2, 102)
        axis.set_ylim(bottom=0)
        styleAxis(axis)
    axes[0].legend(loc="upper left", fontsize=9)
    saveFigure(figure, outputPath, dpi)


def heatmapPanel(
    axis: Axes,
    matrix: np.ndarray,
    rowLabels: list[str],
    columnLabels: list[str],
    title: str,
    colorbarLabel: str,
    figure: Figure,
) -> None:
    image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=0)
    axis.set_title(title, loc="left", pad=12)
    axis.set_xticks(np.arange(len(columnLabels)), labels=columnLabels)
    axis.set_yticks(np.arange(len(rowLabels)), labels=rowLabels)
    axis.tick_params(axis="x", rotation=35, labelsize=8.5)
    axis.tick_params(axis="y", labelsize=8.1)
    axis.spines[:].set_visible(False)
    maximum = float(np.nanmax(matrix)) if matrix.size else 0
    for rowIndex in range(matrix.shape[0]):
        for columnIndex in range(matrix.shape[1]):
            value = matrix[rowIndex, columnIndex]
            axis.text(
                columnIndex,
                rowIndex,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if maximum and value / maximum < 0.48 else INK,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.018)
    colorbar.set_label(colorbarLabel, color=MUTED, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5, colors=MUTED)


def plotSelectedLayerHeatmap(
    imageRows: list[dict[str, Any]],
    layerRows: list[dict[str, Any]],
    outputPath: Path,
    dpi: int,
) -> None:
    orderedImageRows = imageRows
    rowLabels = [row["display_label"] for row in orderedImageRows]
    orderedIds = [row["candidate_image_id"] for row in orderedImageRows]
    layerFrame = pd.DataFrame(layerRows)
    styleLayers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
    spatialLayers = ["relu3_1", "relu4_2", "relu5_1"]

    def buildMatrix(metricFamily: str, layers: list[str]) -> np.ndarray:
        metricFrame = layerFrame[
            (layerFrame["metric_family"] == metricFamily)
            & (layerFrame["layer"].isin(layers))
        ]
        pivot = metricFrame.pivot(
            index="candidate_image_id", columns="layer", values="primary_value"
        )
        pivot = pivot.reindex(index=orderedIds, columns=layers)
        if pivot.isna().any().any():
            raise ValueError(f"逐层矩阵存在缺失值：{metricFamily}")
        return pivot.to_numpy(dtype=float)

    styleMatrix = buildMatrix("style", styleLayers)
    spatialMatrix = buildMatrix("spatial", spatialLayers)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 12.6),
        gridspec_kw={"width_ratios": [1.42, 1.0]},
    )
    figure.subplots_adjust(left=0.19, right=0.96, bottom=0.08, top=0.87, wspace=0.20)
    addFigureHeader(
        figure,
        "Layer-wise VGG19 distances",
        "Raw relative metrics. Separate color scales are required because style and spatial panels use different distance definitions.",
        footer="Source: style-distances.csv and spatial-distances.csv · Cell labels report unrounded source values rounded to 2 decimals",
    )
    heatmapPanel(
        axes[0],
        styleMatrix,
        rowLabels,
        styleLayers,
        "A  Style Gram-matrix distances",
        "Relative Frobenius norm",
        figure,
    )
    heatmapPanel(
        axes[1],
        spatialMatrix,
        rowLabels,
        spatialLayers,
        "B  Spatial activation distances",
        "Relative RMS",
        figure,
    )
    axes[1].tick_params(labelleft=False)
    saveFigure(figure, outputPath, dpi)


def plotTopologyAblation(imageFrame: pd.DataFrame, outputPath: Path, dpi: int) -> None:
    auditOrder = ["original", "matched-control", "mirror-deletion"]
    cnnOrder = ["original", "sham-warp", "matched-control", "mirror-deletion"]
    topologyFrame = imageFrame[imageFrame["category"] == "topology"].copy()
    topologyFrame["control_name"] = topologyFrame["sample_name"].str.replace(
        "topology-", "", regex=False
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.7), gridspec_kw={"width_ratios": [0.8, 1.35]})
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.80, wspace=0.28)
    addFigureHeader(
        figure,
        "Mirror-topology ablation metrics",
        "Manual relation audit and CNN distances are shown separately; neither panel is an inferential test.",
        footer="Manual audit: 5 predefined relations, one project author · CNN: raw VGG19 aggregate distances from working proxy",
    )

    auditFrame = topologyFrame[
        topologyFrame["control_name"].isin(auditOrder)
    ].set_index("control_name").reindex(auditOrder)
    auditValues = auditFrame["relation_fraction"].astype(float).to_numpy()
    auditColors = [NEUTRAL, BLUE, ORANGE]
    bars = axes[0].bar(
        np.arange(len(auditOrder)),
        auditValues,
        color=auditColors,
        edgecolor=INK,
        linewidth=0.7,
        width=0.68,
    )
    axes[0].bar_label(
        bars,
        labels=[
            f"{int(row.relations_preserved)}/{int(row.total_relations)}"
            for row in auditFrame.itertuples()
        ],
        padding=4,
        fontsize=9,
        fontweight="bold",
    )
    axes[0].set_title("A  Manual relation preservation", loc="left")
    axes[0].set_ylabel("Preserved fraction of 5 relations")
    axes[0].set_xticks(np.arange(len(auditOrder)), ["Original", "Matched\ncontrol", "Mirror\ndeletion"])
    axes[0].set_ylim(0, 1.12)
    styleAxis(axes[0], gridAxis="y")

    cnnFrame = topologyFrame.set_index("control_name").reindex(cnnOrder)
    positions = np.arange(len(cnnOrder))
    width = 0.34
    styleBars = axes[1].bar(
        positions - width / 2,
        cnnFrame["style_relative_frobenius"].astype(float),
        width,
        label="Style relative Frobenius",
        color=BLUE,
        edgecolor=INK,
        linewidth=0.7,
    )
    spatialBars = axes[1].bar(
        positions + width / 2,
        cnnFrame["spatial_relative_rms"].astype(float),
        width,
        label="Spatial relative RMS",
        color=ORANGE,
        hatch="//",
        edgecolor=INK,
        linewidth=0.7,
    )
    axes[1].bar_label(styleBars, fmt="%.3f", padding=3, fontsize=7.5, rotation=90)
    axes[1].bar_label(spatialBars, fmt="%.3f", padding=3, fontsize=7.5, rotation=90)
    axes[1].set_title("B  VGG19 response to local controls", loc="left")
    axes[1].set_ylabel("Raw aggregate distance")
    axes[1].set_xticks(
        positions,
        ["Original", "Sham\nwarp", "Matched\ncontrol", "Mirror\ndeletion"],
    )
    axes[1].set_ylim(0, max(cnnFrame["spatial_relative_rms"].max() * 1.32, 0.12))
    axes[1].legend(loc="upper left", fontsize=8.6)
    styleAxis(axes[1], gridAxis="y")
    saveFigure(figure, outputPath, dpi)


def loadLossFrames(
    styleTransferData: dict[str, Any], combinedStyleTransferData: dict[str, Any]
) -> list[pd.DataFrame]:
    lossFrames: list[pd.DataFrame] = []
    manifestSpecs = [
        (styleTransferData, "Original content"),
        (combinedStyleTransferData, "Geometry 100% content"),
    ]
    for manifest, contentLabel in manifestSpecs:
        for record in manifest.get("outputs", []):
            lossPath = Path(record["loss_csv"])
            lossFrame = pd.read_csv(lossPath)
            expectedColumns = {
                "step",
                "style_strength",
                "weighted_content_loss",
                "weighted_style_loss",
                "weighted_tv_loss",
                "total_loss",
            }
            missingColumns = expectedColumns - set(lossFrame.columns)
            if missingColumns:
                raise ValueError(f"损失 CSV 缺列 {sorted(missingColumns)}：{lossPath}")
            finalRecord = record["final_loss"]
            finalRow = lossFrame.iloc[-1]
            if int(finalRow["step"]) != int(finalRecord["step"]):
                raise ValueError(f"损失 CSV 与 manifest 最终步不一致：{lossPath}")
            if not math.isclose(
                float(finalRow["total_loss"]),
                float(finalRecord["total_loss"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"损失 CSV 与 manifest 最终损失不一致：{lossPath}")
            lossFrame["content_label"] = contentLabel
            lossFrames.append(lossFrame)
    return lossFrames


def plotNeuralStyleLossCurves(
    lossFrames: list[pd.DataFrame], outputPath: Path, dpi: int
) -> None:
    if len(lossFrames) != 4:
        raise ValueError(f"预期 3 个原作强度和 1 个 combined 强度，实际 {len(lossFrames)}")
    globalMaximum = max(float(frame["total_loss"].max()) for frame in lossFrames)
    figure, axesGrid = plt.subplots(2, 2, figsize=(13.8, 9.2), sharex=True, sharey=True)
    axes = axesGrid.ravel()
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.84, wspace=0.14, hspace=0.28)
    addFigureHeader(
        figure,
        "Neural style-transfer optimization losses",
        "Raw weighted objective components over 500 Adam steps. Coefficients and content inputs differ, so curves are not model rankings.",
        footer="Source: outputs/neural-style-transfer*/loss-strength-*.csv · Shared linear y-axis · No smoothing",
    )
    for axis, lossFrame in zip(axes, lossFrames, strict=True):
        strength = float(lossFrame["style_strength"].iloc[0])
        contentLabel = str(lossFrame["content_label"].iloc[0])
        axis.plot(
            lossFrame["step"],
            lossFrame["total_loss"],
            color=INK,
            linewidth=2.0,
            label="Total",
            zorder=3,
        )
        axis.plot(
            lossFrame["step"],
            lossFrame["weighted_content_loss"],
            color=BLUE,
            linewidth=1.4,
            label="Weighted content",
        )
        axis.plot(
            lossFrame["step"],
            lossFrame["weighted_style_loss"],
            color=PINK,
            linestyle="--",
            linewidth=1.4,
            label="Weighted style",
        )
        finalRow = lossFrame.iloc[-1]
        axis.scatter(
            [finalRow["step"]],
            [finalRow["total_loss"]],
            color=INK,
            edgecolor="white",
            s=40,
            zorder=4,
        )
        axis.annotate(
            f"final {finalRow['total_loss']:.3f}",
            (finalRow["step"], finalRow["total_loss"]),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=INK,
        )
        axis.set_title(f"{contentLabel} · strength {strength:g}×", loc="left")
        axis.set_xlabel("Optimization step")
        axis.set_xlim(0, 500)
        axis.set_ylim(0, globalMaximum * 1.05)
        styleAxis(axis)
    axes[0].set_ylabel("Weighted loss")
    axes[2].set_ylabel("Weighted loss")
    axes[0].legend(loc="upper right", fontsize=8.4)
    saveFigure(figure, outputPath, dpi)


def main() -> None:
    args = parseArgs()
    inputPaths = {
        "cnn_json": args.cnn_json,
        "style_csv": args.style_csv,
        "spatial_csv": args.spatial_csv,
        "style_transfer_manifest": args.style_transfer_manifest,
        "combined_style_transfer_manifest": args.combined_style_transfer_manifest,
        "topology_csv": args.topology_csv,
        "transform_manifest": args.transform_manifest,
    }
    missingPaths = [str(path) for path in inputPaths.values() if not path.is_file()]
    if missingPaths:
        raise FileNotFoundError(f"缺少输入文件：{missingPaths}")

    cnnData = loadJson(args.cnn_json)
    styleRows = loadCsv(args.style_csv)
    spatialRows = loadCsv(args.spatial_csv)
    styleTransferData = loadJson(args.style_transfer_manifest)
    combinedStyleTransferData = loadJson(args.combined_style_transfer_manifest)
    topologyRows = loadCsv(args.topology_csv)
    # 读取变换清单并确认其格式；样本分类仍以 CNN 实际 candidate_path 为准。
    transformData = loadJson(args.transform_manifest)
    if "records" not in transformData:
        raise ValueError("变换清单缺少 records")

    validateCsvAgainstJson(cnnData, styleRows, spatialRows)
    validateZeroBaselines(cnnData)
    imageRows, layerRows = buildRows(
        cnnData,
        styleRows,
        spatialRows,
        styleTransferData,
        combinedStyleTransferData,
        topologyRows,
    )
    lossFrames = loadLossFrames(styleTransferData, combinedStyleTransferData)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    writeCsv(args.output_dir / "image-metrics.csv", imageRows)
    writeCsv(args.output_dir / "layer-metrics.csv", layerRows)
    summary = buildSummary(
        inputPaths,
        cnnData,
        styleTransferData,
        combinedStyleTransferData,
        imageRows,
        layerRows,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    configureMatplotlib()
    imageFrame = pd.DataFrame(imageRows)
    plotStyleSpatialScatter(
        imageFrame, args.output_dir / "style-vs-spatial-scatter.png", args.dpi
    )
    plotTransformationTrajectories(
        imageFrame, args.output_dir / "transformation-trajectories.png", args.dpi
    )
    plotSelectedLayerHeatmap(
        imageRows, layerRows, args.output_dir / "selected-layer-heatmap.png", args.dpi
    )
    plotTopologyAblation(
        imageFrame, args.output_dir / "topology-ablation-metrics.png", args.dpi
    )
    plotNeuralStyleLossCurves(
        lossFrames, args.output_dir / "neural-style-loss-curves.png", args.dpi
    )
    print(f"已生成 {len(imageRows)} 个样本、{len(layerRows)} 条逐层指标和 5 张定量图：{args.output_dir}")


if __name__ == "__main__":
    main()
