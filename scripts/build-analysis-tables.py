#!/usr/bin/env python3
"""把 CNN 与几何分析 JSON 汇总为可审计、可直接制图的 CSV。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def projectRelativePath(path: str | Path) -> str:
    """优先记录项目相对路径，避免公开清单泄露本机目录。"""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class CandidateIdentity:
    """从受控输出路径解析出的实验身份。"""

    label: str
    group: str
    intensity: float | None


@dataclass(frozen=True)
class DistanceRecord:
    """一个候选图像相对原作的聚合 CNN 距离。"""

    path: str
    label: str
    group: str
    intensity: float | None
    styleDistance: float
    spatialDistance: float
    spatialRms: float


def loadJson(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON：{path}")
    with path.open(encoding="utf-8") as fileHandle:
        payload = json.load(fileHandle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def finiteMean(values: Iterable[float], fieldName: str) -> float:
    numericValues = [float(value) for value in values]
    if not numericValues:
        raise ValueError(f"{fieldName} 没有任何数值。")
    if not all(math.isfinite(value) for value in numericValues):
        raise ValueError(f"{fieldName} 包含非有限数值。")
    return sum(numericValues) / len(numericValues)


def parsePercent(stem: str) -> float | None:
    match = re.search(r"-(\d{3})$", stem)
    return float(int(match.group(1))) if match else None


def parseStrength(stem: str) -> float | None:
    prefix = "neural-style-strength-"
    if not stem.startswith(prefix):
        return None
    rawValue = stem.removeprefix(prefix).replace("m", "-").replace("p", ".")
    try:
        return float(rawValue)
    except ValueError:
        return None


def identifyCandidate(candidatePath: str) -> CandidateIdentity:
    normalizedPath = candidatePath.replace("\\", "/")
    path = Path(candidatePath)
    stem = path.stem

    if "/transformations/geometry/" in normalizedPath:
        intensity = parsePercent(stem)
        suffix = "?" if intensity is None else f"{intensity:g}%"
        return CandidateIdentity(f"Geometry {suffix}", "geometry", intensity)

    if "/transformations/style/" in normalizedPath:
        intensity = parsePercent(stem)
        suffix = "?" if intensity is None else f"{intensity:g}%"
        return CandidateIdentity(
            f"Non-neural baseline {suffix}",
            "non-neural-style-baseline",
            intensity,
        )

    if "/neural-style-transfer-combined/" in normalizedPath:
        strength = parseStrength(stem)
        intensity = None if strength is None else strength * 100.0
        return CandidateIdentity(
            "Geometry 100% + neural style 1.0",
            "geometry-plus-neural-style",
            intensity,
        )

    if "/combined-neural/" in normalizedPath:
        return CandidateIdentity(
            "Neural style + geometry 100%",
            "geometry-plus-neural-style",
            100.0,
        )

    if "/neural-style-transfer/" in normalizedPath:
        strength = parseStrength(stem)
        intensity = None if strength is None else strength * 100.0
        suffix = "?" if strength is None else f"{strength:g}"
        return CandidateIdentity(f"Neural style {suffix}", "neural-style", intensity)

    if "/transformations/topology/" in normalizedPath:
        topologyLabels = {
            "topology-original": "Topology original",
            "topology-mirror-deletion": "Mirror deleted",
            "topology-matched-control": "Matched-patch control",
            "topology-sham-warp": "Sham-warp control",
        }
        return CandidateIdentity(topologyLabels.get(stem, stem), "topology-control", None)

    if stem == "combined-geometry-style-100":
        return CandidateIdentity(
            "Geometry + non-neural baseline",
            "geometry-plus-non-neural-baseline",
            100.0,
        )

    if stem == "entering-las-meninas-final":
        return CandidateIdentity("Final personal artwork", "personal-artwork", None)

    if stem == "picasso":
        return CandidateIdentity(
            "Picasso comparison (local; image not redistributed)",
            "external-historical-comparison",
            None,
        )

    return CandidateIdentity(stem.replace("-", " ").title(), "other", None)


def extractLayerValues(
    layerPayload: Mapping[str, Any],
    valueName: str,
    context: str,
) -> list[float]:
    values: list[float] = []
    for layerName, metrics in layerPayload.items():
        if not isinstance(metrics, Mapping) or valueName not in metrics:
            raise ValueError(f"{context}.{layerName} 缺少 {valueName}。")
        values.append(float(metrics[valueName]))
    return values


def extractDistanceRecords(cnnPayload: Mapping[str, Any]) -> list[DistanceRecord]:
    comparisons = cnnPayload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("CNN JSON 必须含有非空 comparisons 列表。")

    records: list[DistanceRecord] = []
    for comparisonIndex, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            raise ValueError(f"comparisons[{comparisonIndex}] 必须是对象。")
        candidatePath = comparison.get("candidate_path")
        stylePayload = comparison.get("style")
        spatialPayload = comparison.get("spatial")
        if not isinstance(candidatePath, str):
            raise ValueError(f"comparisons[{comparisonIndex}] 缺少 candidate_path。")
        if not isinstance(stylePayload, Mapping) or not isinstance(spatialPayload, Mapping):
            raise ValueError(f"comparisons[{comparisonIndex}] 缺少 style 或 spatial。")

        identity = identifyCandidate(candidatePath)
        records.append(
            DistanceRecord(
                path=candidatePath,
                label=identity.label,
                group=identity.group,
                intensity=identity.intensity,
                styleDistance=finiteMean(
                    extractLayerValues(
                        stylePayload,
                        "relative_frobenius",
                        f"comparisons[{comparisonIndex}].style",
                    ),
                    "style relative_frobenius",
                ),
                spatialDistance=finiteMean(
                    extractLayerValues(
                        spatialPayload,
                        "mean_cosine_distance",
                        f"comparisons[{comparisonIndex}].spatial",
                    ),
                    "spatial mean_cosine_distance",
                ),
                spatialRms=finiteMean(
                    extractLayerValues(
                        spatialPayload,
                        "relative_rms",
                        f"comparisons[{comparisonIndex}].spatial",
                    ),
                    "spatial relative_rms",
                ),
            )
        )
    return records


def writeCsv(path: Path, fieldNames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fileHandle:
        writer = csv.DictWriter(fileHandle, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(rows)


def writeFeatureSpace(records: Sequence[DistanceRecord], outputPath: Path) -> None:
    rows = [
        {
            "x": f"{record.styleDistance:.10g}",
            "y": f"{record.spatialDistance:.10g}",
            "label": record.label,
            "group": record.group,
            "intensity": "" if record.intensity is None else f"{record.intensity:g}",
            "candidate_path": record.path,
            "spatial_relative_rms": f"{record.spatialRms:.10g}",
        }
        for record in records
    ]
    writeCsv(
        outputPath,
        ("x", "y", "label", "group", "intensity", "candidate_path", "spatial_relative_rms"),
        rows,
    )


def writeTrajectory(
    records: Sequence[DistanceRecord],
    group: str,
    outputPath: Path,
    *,
    includeReferenceOrigin: bool = False,
) -> int:
    selected = [record for record in records if record.group == group and record.intensity is not None]
    if includeReferenceOrigin and not any(record.intensity == 0.0 for record in selected):
        rows: list[dict[str, str]] = [
            {
                "intensity": "0",
                "style-distance": "0",
                "spatial-cosine-distance": "0",
                "spatial-relative-rms": "0",
            }
        ]
    else:
        rows = []
    rows.extend(
        {
            "intensity": f"{record.intensity:g}",
            "style-distance": f"{record.styleDistance:.10g}",
            "spatial-cosine-distance": f"{record.spatialDistance:.10g}",
            "spatial-relative-rms": f"{record.spatialRms:.10g}",
        }
        for record in sorted(selected, key=lambda item: float(item.intensity or 0.0))
    )
    if not rows:
        raise ValueError(f"CNN 比较中没有 {group} 轨迹记录。")
    writeCsv(
        outputPath,
        ("intensity", "style-distance", "spatial-cosine-distance", "spatial-relative-rms"),
        rows,
    )
    return len(rows)


def finiteOrNan(value: Any) -> str:
    if value is None:
        return "nan"
    numericValue = float(value)
    return f"{numericValue:.10g}" if math.isfinite(numericValue) else "nan"


def writeGeometryTrajectory(geometryRoot: Path, outputPath: Path) -> int:
    rows: list[dict[str, str]] = []
    expectedRuns = (("original", 0.0), ("geometry-025", 25.0), ("geometry-050", 50.0),
                    ("geometry-075", 75.0), ("geometry-100", 100.0))
    for directoryName, intensity in expectedRuns:
        payload = loadJson(geometryRoot / directoryName / "geometry-analysis.json")
        primaryGeometry = payload.get("primary_geometry")
        if not isinstance(primaryGeometry, Mapping):
            raise ValueError(f"{directoryName} 缺少 primary_geometry。")
        vanishingPoint = primaryGeometry.get("vanishing_point")
        if vanishingPoint is None:
            vanishingPoint = {}
        if not isinstance(vanishingPoint, Mapping):
            raise ValueError(f"{directoryName}.vanishing_point 必须是对象或 null。")
        rows.append(
            {
                "intensity": f"{intensity:g}",
                "line-count": finiteOrNan(primaryGeometry.get("line_count")),
                "vanishing-x": finiteOrNan(vanishingPoint.get("normalized_x")),
                "vanishing-y": finiteOrNan(vanishingPoint.get("normalized_y")),
                "weighted-inlier-ratio": finiteOrNan(vanishingPoint.get("weighted_inlier_ratio")),
                "cnn-canny-vp-distance": finiteOrNan(
                    payload.get("cnn_canny_vanishing_point_distance_normalized")
                ),
            }
        )
    writeCsv(
        outputPath,
        (
            "intensity",
            "line-count",
            "vanishing-x",
            "vanishing-y",
            "weighted-inlier-ratio",
            "cnn-canny-vp-distance",
        ),
        rows,
    )
    return len(rows)


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 cnn-analysis.json 与逐条件 geometry-analysis.json 汇总为 CSV。"
    )
    parser.add_argument("--cnn-json", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = buildParser().parse_args(argv)
    cnnPayload = loadJson(arguments.cnn_json)
    records = extractDistanceRecords(cnnPayload)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    featureSpacePath = arguments.output_dir / "cnn-feature-space.csv"
    geometryCnnPath = arguments.output_dir / "geometry-cnn-metrics.csv"
    baselineCnnPath = arguments.output_dir / "non-neural-style-cnn-metrics.csv"
    neuralStyleCnnPath = arguments.output_dir / "neural-style-cnn-metrics.csv"
    geometryDetectionPath = arguments.output_dir / "geometry-detection-metrics.csv"

    writeFeatureSpace(records, featureSpacePath)
    trajectoryCounts = {
        "geometry": writeTrajectory(records, "geometry", geometryCnnPath),
        "non-neural-style-baseline": writeTrajectory(
            records,
            "non-neural-style-baseline",
            baselineCnnPath,
        ),
        "neural-style": writeTrajectory(
            records,
            "neural-style",
            neuralStyleCnnPath,
            includeReferenceOrigin=True,
        ),
        "geometry-detection": writeGeometryTrajectory(
            arguments.geometry_root,
            geometryDetectionPath,
        ),
    }

    summary = {
        "source_cnn_json": projectRelativePath(arguments.cnn_json),
        "source_geometry_root": projectRelativePath(arguments.geometry_root),
        "comparison_count": len(records),
        "trajectory_row_counts": trajectoryCounts,
        "outputs": [
            projectRelativePath(path)
            for path in (
                featureSpacePath,
                geometryCnnPath,
                baselineCnnPath,
                neuralStyleCnnPath,
                geometryDetectionPath,
            )
        ],
        "aggregation": {
            "style_distance": "unweighted mean of per-layer relative Frobenius Gram distances",
            "spatial_distance": "unweighted mean of per-layer mean cosine distances",
            "spatial_rms": "unweighted mean of per-layer relative RMS distances",
        },
    }
    summaryPath = arguments.output_dir / "analysis-tables.json"
    with summaryPath.open("w", encoding="utf-8") as fileHandle:
        json.dump(summary, fileHandle, ensure_ascii=False, indent=2)
        fileHandle.write("\n")
    print(json.dumps({"output": str(summaryPath), "comparisons": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
