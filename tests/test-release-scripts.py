from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def runCommand(command: list[str], workingDirectory: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=workingDirectory,
        check=True,
        capture_output=True,
        text=True,
    )


def testFullPipelineShellSyntaxAndDryRun() -> None:
    scriptPath = PROJECT_ROOT / "scripts" / "run-full-pipeline.sh"
    runCommand(["bash", "-n", str(scriptPath)])
    completed = runCommand(
        [
            "bash",
            str(scriptPath),
            "--dry-run",
            "--skip-download",
            "--python",
            sys.executable,
        ]
    )
    for requiredCommand in (
        "transformations.py",
        "neural-style-transfer.py",
        "cnn-analysis.py",
        "geometry-analysis.py",
        "build-analysis-tables.py",
        "summarize-results.py",
        "visualization.py",
        "validate-artifacts.py",
    ):
        assert requiredCommand in completed.stdout
    assert '"--weights" "default"' in completed.stdout
    assert '"--heuristic-depth"' in completed.stdout
    assert '"--vanishing-roi" "0.5,0.3,0.65,0.48"' in completed.stdout
    assert '"--vanishing-roi" "0.5,0.495,0.645,0.62"' in completed.stdout


def comparisonRecord(candidatePath: str, scale: float) -> dict[str, object]:
    return {
        "candidate_path": candidatePath,
        "style": {
            "relu1_1": {"relative_frobenius": 0.1 * scale},
            "relu2_1": {"relative_frobenius": 0.2 * scale},
        },
        "spatial": {
            "relu3_2": {"mean_cosine_distance": 0.03 * scale, "relative_rms": 0.4 * scale},
            "relu4_2": {"mean_cosine_distance": 0.05 * scale, "relative_rms": 0.6 * scale},
        },
    }


def writeGeometryManifest(path: Path, intensity: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend_used": "hed",
        "cnn_canny_vanishing_point_distance_normalized": 0.01 + intensity / 10_000,
        "primary_geometry": {
            "line_count": 100 + int(intensity),
            "vanishing_point": {
                "normalized_x": 0.55 + intensity / 10_000,
                "normalized_y": 0.50,
                "weighted_inlier_ratio": 0.2,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def testBuildAnalysisTablesFromSyntheticManifests(tmp_path: Path) -> None:
    cnnJson = tmp_path / "cnn-analysis.json"
    geometryRoot = tmp_path / "geometry-analysis"
    outputDirectory = tmp_path / "tables"

    comparisons: list[dict[str, object]] = []
    for index, intensity in enumerate((0, 25, 50, 75, 100), start=1):
        comparisons.append(
            comparisonRecord(
                f"/project/outputs/transformations/geometry/geometry-{intensity:03d}.png",
                float(index),
            )
        )
        comparisons.append(
            comparisonRecord(
                f"/project/outputs/transformations/style/style-baseline-{intensity:03d}.png",
                float(index) * 0.5,
            )
        )
    for strength in ("0p25", "0p5", "1"):
        comparisons.append(
            comparisonRecord(
                f"/project/outputs/neural-style-transfer/neural-style-strength-{strength}.png",
                2.0,
            )
        )
    cnnJson.write_text(json.dumps({"comparisons": comparisons}), encoding="utf-8")

    for directoryName, intensity in (
        ("original", 0.0),
        ("geometry-025", 25.0),
        ("geometry-050", 50.0),
        ("geometry-075", 75.0),
        ("geometry-100", 100.0),
    ):
        writeGeometryManifest(
            geometryRoot / directoryName / "geometry-analysis.json",
            intensity,
        )

    completed = runCommand(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build-analysis-tables.py"),
            "--cnn-json",
            str(cnnJson),
            "--geometry-root",
            str(geometryRoot),
            "--output-dir",
            str(outputDirectory),
        ]
    )
    assert '"comparisons": 13' in completed.stdout

    with (outputDirectory / "geometry-cnn-metrics.csv").open(newline="", encoding="utf-8") as fileHandle:
        geometryRows = list(csv.DictReader(fileHandle))
    with (outputDirectory / "neural-style-cnn-metrics.csv").open(
        newline="", encoding="utf-8"
    ) as fileHandle:
        neuralRows = list(csv.DictReader(fileHandle))
    with (outputDirectory / "cnn-feature-space.csv").open(newline="", encoding="utf-8") as fileHandle:
        featureRows = list(csv.DictReader(fileHandle))

    assert [float(row["intensity"]) for row in geometryRows] == [0, 25, 50, 75, 100]
    assert [float(row["intensity"]) for row in neuralRows] == [0, 25, 50, 100]
    assert len(featureRows) == 13
    assert all(float(row["style-distance"]) >= 0 for row in geometryRows)


def testValidationScriptHelpIsAvailable() -> None:
    completed = runCommand(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate-artifacts.py"), "--help"]
    )
    assert "验证《宫娥》正式流水线" in completed.stdout
