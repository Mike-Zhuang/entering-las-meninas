"""结果汇总、分类、原始量纲与定量图输出的回归测试。"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "summarize-results.py"


def loadModule() -> ModuleType:
    moduleName = "las_meninas_summarize_results"
    specification = importlib.util.spec_from_file_location(moduleName, MODULE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"无法加载模块：{MODULE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[moduleName] = module
    specification.loader.exec_module(module)
    return module


summaryModule = loadModule()


def test_cliHelpFormatsPercentSigns() -> None:
    completedProcess = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completedProcess.returncode == 0
    assert "100% 几何变换" in completedProcess.stdout


def loadProjectInputs() -> tuple[dict, list[dict[str, str]], list[dict[str, str]], dict, dict, list[dict[str, str]]]:
    cnnData = summaryModule.loadJson(
        PROJECT_ROOT / "outputs" / "cnn-analysis" / "cnn-analysis.json"
    )
    styleRows = summaryModule.loadCsv(
        PROJECT_ROOT / "outputs" / "cnn-analysis" / "style-distances.csv"
    )
    spatialRows = summaryModule.loadCsv(
        PROJECT_ROOT / "outputs" / "cnn-analysis" / "spatial-distances.csv"
    )
    styleTransferData = summaryModule.loadJson(
        PROJECT_ROOT / "outputs" / "neural-style-transfer" / "manifest.json"
    )
    combinedStyleTransferData = summaryModule.loadJson(
        PROJECT_ROOT / "outputs" / "neural-style-transfer-combined" / "manifest.json"
    )
    topologyRows = summaryModule.loadCsv(PROJECT_ROOT / "data" / "topology-relations.csv")
    return (
        cnnData,
        styleRows,
        spatialRows,
        styleTransferData,
        combinedStyleTransferData,
        topologyRows,
    )


@pytest.mark.parametrize(
    ("candidateId", "candidatePath", "expected"),
    [
        ("style-baseline-100-65957c81fc", "outputs/transformations/style/style-baseline-100.png", "deterministic-style"),
        ("geometry-100-a19aaa76aa", "outputs/transformations/geometry/geometry-100.png", "geometry"),
        ("topology-mirror-deletion-35ba0b3128", "outputs/transformations/topology/topology-mirror-deletion.png", "topology"),
        ("neural-style-strength-1-4c716da10d", "outputs/neural-style-transfer/neural-style-strength-1.png", "neural-style"),
        ("combined-geometry-style-100-27ede2c6e0", "outputs/transformations/combined-geometry-style-100.png", "combined"),
        ("neural-style-strength-1-e731e19ea5", "outputs/neural-style-transfer-combined/neural-style-strength-1.png", "combined"),
        ("entering-las-meninas-final-6b697a5de5", "outputs/artwork/entering-las-meninas-final.png", "final-artwork"),
        ("picasso-fd3932138d", "pics/picasso.png", "picasso"),
    ],
)
def test_classifySampleUsesExperimentalRole(candidateId: str, candidatePath: str, expected: str) -> None:
    assert summaryModule.classifySample(candidateId, candidatePath) == expected


def test_currentCsvFilesExactlyMatchCurrentJson() -> None:
    cnnData, styleRows, spatialRows, *_ = loadProjectInputs()
    summaryModule.validateCsvAgainstJson(cnnData, styleRows, spatialRows)


def test_registeredZeroBaselinesAreExactlyZero() -> None:
    cnnData, *_ = loadProjectInputs()
    summaryModule.validateZeroBaselines(cnnData)
    baselinePaths = {
        "outputs/transformations/style/style-baseline-000.png",
        "outputs/transformations/geometry/geometry-000.png",
        "outputs/transformations/topology/topology-original.png",
    }
    baselineComparisons = [
        row for row in cnnData["comparisons"] if row["candidate_path"] in baselinePaths
    ]
    assert len(baselineComparisons) == 3
    for comparison in baselineComparisons:
        assert comparison["style"]["aggregate"]["relative_frobenius"] == 0.0
        assert comparison["spatial"]["aggregate"]["relative_rms"] == 0.0


def test_zeroBaselineValidatorRejectsStaleReferenceDistances() -> None:
    cnnData, *_ = loadProjectInputs()
    staleData = deepcopy(cnnData)
    for comparison in staleData["comparisons"]:
        if comparison["candidate_path"].endswith("geometry-000.png"):
            comparison["spatial"]["aggregate"]["relative_rms"] = 0.083
            break
    with pytest.raises(ValueError, match="没有严格对齐"):
        summaryModule.validateZeroBaselines(staleData)


def test_buildRowsKeepsRawMetricsAndAllRequiredCategories() -> None:
    (
        cnnData,
        styleRows,
        spatialRows,
        styleTransferData,
        combinedStyleTransferData,
        topologyRows,
    ) = loadProjectInputs()
    imageRows, layerRows = summaryModule.buildRows(
        cnnData,
        styleRows,
        spatialRows,
        styleTransferData,
        combinedStyleTransferData,
        topologyRows,
    )
    picassoCount = sum(row["category"] == "picasso" for row in imageRows)
    assert picassoCount in {0, 1}
    assert len(imageRows) == 20 + picassoCount
    assert len(layerRows) == len(styleRows) + len(spatialRows) == 200 + 10 * picassoCount
    expectedCategories = {
        "geometry": 5,
        "deterministic-style": 5,
        "topology": 4,
        "combined": 2,
        "neural-style": 3,
        "final-artwork": 1,
    }
    if picassoCount:
        expectedCategories["picasso"] = 1
    assert Counter(row["category"] for row in imageRows) == expectedCategories

    jsonLookup = {
        row["candidate_image_id"]: row for row in cnnData["comparisons"]
    }
    for row in imageRows:
        source = jsonLookup[row["candidate_image_id"]]
        assert row["style_relative_frobenius"] == source["style"]["aggregate"][
            "relative_frobenius"
        ]
        assert row["spatial_relative_rms"] == source["spatial"]["aggregate"][
            "relative_rms"
        ]

    combinedRows = [row for row in imageRows if row["category"] == "combined"]
    assert {row["display_label"] for row in combinedRows} == {
        "Deterministic style + geometry 100%",
        "NST 1.0× + geometry 100%",
    }
    assert sum(row["nst_final_total_loss"] is not None for row in combinedRows) == 1


def test_topologyAuditIsKeptSeparateFromCnnDistance() -> None:
    imageMetricsPath = PROJECT_ROOT / "outputs" / "metrics" / "image-metrics.csv"
    with imageMetricsPath.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    byName = {row["sample_name"]: row for row in rows}
    mirrorDeletion = byName["topology-mirror-deletion"]
    matchedControl = byName["topology-matched-control"]
    assert mirrorDeletion["relations_preserved"] == "4"
    assert matchedControl["relations_preserved"] == "5"
    assert float(mirrorDeletion["spatial_relative_rms"]) == pytest.approx(
        0.09276891071491822
    )
    assert float(matchedControl["spatial_relative_rms"]) == pytest.approx(
        0.08429480155559115
    )


def test_lossFramesIncludeOriginalAndCombinedContent() -> None:
    _, _, _, styleTransferData, combinedStyleTransferData, _ = loadProjectInputs()
    frames = summaryModule.loadLossFrames(styleTransferData, combinedStyleTransferData)
    assert len(frames) == 4
    assert [len(frame) for frame in frames] == [501, 501, 501, 501]
    assert [str(frame["content_label"].iloc[0]) for frame in frames] == [
        "Original content",
        "Original content",
        "Original content",
        "Geometry 100% content",
    ]
    assert float(frames[-1]["total_loss"].iloc[-1]) == pytest.approx(
        1.1978967189788818
    )


def test_summaryAndTablesMatchFrozenResultSet() -> None:
    metricsDir = PROJECT_ROOT / "outputs" / "metrics"
    summary = json.loads((metricsDir / "summary.json").read_text(encoding="utf-8"))
    candidateCount = summary["record_counts"]["candidate_images"]
    assert candidateCount in {20, 21}
    assert summary["record_counts"]["layer_metric_rows"] == candidateCount * 10
    assert summary["record_counts"]["neural_style_optimization_records"] == 2004
    assert summary["cnn_reference"]["path"] == "outputs/transformations/working-proxy.png"
    assert summary["registered_trajectory_endpoints"]["geometry_100_percent"][
        "spatial_relative_rms"
    ] == pytest.approx(0.9188680604193241)
    assert len(summary["chart_contracts"]) == 5
    assert all(
        "normalization" not in contract.get("scales", "").lower()
        or "no" in contract.get("scales", "").lower()
        for contract in summary["chart_contracts"].values()
    )


def test_quantitativeFiguresAreHighResolutionAndNonblank() -> None:
    metricsDir = PROJECT_ROOT / "outputs" / "metrics"
    figureNames = [
        "style-vs-spatial-scatter.png",
        "transformation-trajectories.png",
        "selected-layer-heatmap.png",
        "topology-ablation-metrics.png",
        "neural-style-loss-curves.png",
    ]
    for figureName in figureNames:
        with Image.open(metricsDir / figureName) as image:
            assert image.format == "PNG"
            assert image.width >= 3000
            assert image.height >= 1800
            statistics = ImageStat.Stat(image.convert("RGB"))
            assert max(statistics.stddev) > 20
