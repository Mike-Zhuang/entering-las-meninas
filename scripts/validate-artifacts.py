#!/usr/bin/env python3
"""验证正式流水线的模型身份、清单、文件完整性与可提交图像。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class ValidationCollector:
    """累积全部验证结果，避免首个错误掩盖后续缺失项。"""

    projectRoot: Path
    checks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})

    def resolveRecordedPath(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.projectRoot / path

    def requireFile(self, path: Path, name: str, minimumBytes: int = 1) -> bool:
        passed = path.is_file() and path.stat().st_size >= minimumBytes
        if passed:
            relativePath = relativeOrAbsolute(path, self.projectRoot)
            self.artifacts[relativePath] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256File(path),
            }
            self.record(name, True, f"{relativePath} · {path.stat().st_size} bytes")
        else:
            self.record(name, False, f"缺失或小于 {minimumBytes} bytes：{path}")
        return passed

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if not check["passed"]]


def relativeOrAbsolute(path: Path, projectRoot: Path) -> str:
    try:
        return str(path.resolve().relative_to(projectRoot.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256File(path: Path, chunkSize: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fileHandle:
        while block := fileHandle.read(chunkSize):
            digest.update(block)
    return digest.hexdigest()


def loadJson(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fileHandle:
        payload = json.load(fileHandle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return payload


def loadJsonSafely(
    collector: ValidationCollector,
    path: Path,
    name: str,
) -> dict[str, Any] | None:
    if not collector.requireFile(path, name, minimumBytes=20):
        return None
    try:
        return loadJson(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        collector.record(f"{name} 可解析", False, str(exc))
        return None


def validateImage(
    collector: ValidationCollector,
    path: Path,
    name: str,
    minimumSide: int = 256,
) -> None:
    if not collector.requireFile(path, name, minimumBytes=1024):
        return
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            valid = min(width, height) >= minimumSide
            collector.record(
                f"{name} 图像尺寸",
                valid,
                f"{width}×{height}，最短边要求 ≥ {minimumSide}",
            )
    except (OSError, ValueError) as exc:
        collector.record(f"{name} 可解码", False, str(exc))


def validateCsv(
    collector: ValidationCollector,
    path: Path,
    name: str,
    minimumRows: int,
) -> None:
    if not collector.requireFile(path, name, minimumBytes=10):
        return
    try:
        with path.open(encoding="utf-8-sig", newline="") as fileHandle:
            reader = csv.DictReader(fileHandle)
            rows = list(reader)
            hasHeader = bool(reader.fieldnames)
        collector.record(
            f"{name} 数据行",
            hasHeader and len(rows) >= minimumRows,
            f"表头={reader.fieldnames}，数据行={len(rows)}，要求 ≥ {minimumRows}",
        )
    except (OSError, UnicodeError, csv.Error) as exc:
        collector.record(f"{name} 可解析", False, str(exc))


def validateTransformations(
    collector: ValidationCollector,
    outputRoot: Path,
) -> None:
    transformationsRoot = outputRoot / "transformations"
    manifestPath = transformationsRoot / "transformations-manifest.json"
    manifest = loadJsonSafely(collector, manifestPath, "受控变换清单")
    if manifest is None:
        return

    levels = manifest.get("levels")
    collector.record(
        "受控强度完整",
        levels == [0, 25, 50, 75, 100],
        f"levels={levels}",
    )
    methodNote = str(manifest.get("method_note", ""))
    collector.record(
        "非神经基线已披露",
        "not neural style transfer" in methodNote.lower(),
        methodNote or "method_note 缺失",
    )

    records = manifest.get("records")
    recordList = records if isinstance(records, list) else []
    collector.record(
        "受控条件数量",
        len(recordList) >= 15,
        f"records={len(recordList)}，要求 ≥ 15",
    )
    for recordIndex, record in enumerate(recordList):
        if not isinstance(record, Mapping) or not isinstance(record.get("output_path"), str):
            collector.record(
                f"受控条件 {recordIndex} 结构",
                False,
                "缺少字符串 output_path",
            )
            continue
        outputPath = collector.resolveRecordedPath(record["output_path"])
        collector.requireFile(outputPath, f"受控条件 {record.get('name', recordIndex)}", 1024)

    validateImage(collector, transformationsRoot / "working-proxy.png", "统一工作代理图")


def validateStyleManifest(
    collector: ValidationCollector,
    manifestPath: Path,
    name: str,
    expectedStrengths: Sequence[float],
) -> None:
    manifest = loadJsonSafely(collector, manifestPath, f"{name}清单")
    if manifest is None:
        return
    parameters = manifest.get("parameters")
    implementation = manifest.get("implementation")
    outputs = manifest.get("outputs")
    parametersMap = parameters if isinstance(parameters, Mapping) else {}
    implementationMap = implementation if isinstance(implementation, Mapping) else {}
    outputList = outputs if isinstance(outputs, list) else []

    collector.record(
        f"{name}使用正式 VGG19 权重",
        parametersMap.get("weights") == "default"
        and "vgg19" in str(implementationMap.get("model_signature", "")).lower(),
        f"weights={parametersMap.get('weights')}，model={implementationMap.get('model_signature')}",
    )
    strengths = parametersMap.get("style_strengths")
    collector.record(
        f"{name}强度完整",
        strengths == list(expectedStrengths),
        f"style_strengths={strengths}",
    )
    collector.record(
        f"{name}输出数量",
        len(outputList) == len(expectedStrengths),
        f"outputs={len(outputList)}，要求={len(expectedStrengths)}",
    )

    for outputIndex, output in enumerate(outputList):
        if not isinstance(output, Mapping) or not isinstance(output.get("image"), str):
            collector.record(f"{name}输出 {outputIndex}", False, "缺少 image 路径")
            continue
        imagePath = collector.resolveRecordedPath(output["image"])
        validateImage(collector, imagePath, f"{name}图像 {outputIndex}")
        expectedHash = output.get("image_sha256")
        if imagePath.is_file() and isinstance(expectedHash, str):
            actualHash = sha256File(imagePath)
            collector.record(
                f"{name}图像 {outputIndex} 哈希",
                actualHash == expectedHash,
                f"expected={expectedHash}，actual={actualHash}",
            )


def validateCnnAnalysis(collector: ValidationCollector, outputRoot: Path) -> None:
    cnnRoot = outputRoot / "cnn-analysis"
    manifest = loadJsonSafely(collector, cnnRoot / "cnn-analysis.json", "CNN 分析清单")
    if manifest is None:
        return
    config = manifest.get("config")
    configMap = config if isinstance(config, Mapping) else {}
    comparisons = manifest.get("comparisons")
    images = manifest.get("images")
    comparisonList = comparisons if isinstance(comparisons, list) else []
    imageList = images if isinstance(images, list) else []

    collector.record(
        "CNN 使用正式 VGG19 权重",
        configMap.get("weights") == "default"
        and "vgg19" in str(manifest.get("model_signature", "")).lower(),
        f"weights={configMap.get('weights')}，model={manifest.get('model_signature')}",
    )
    collector.record(
        "CNN 候选条件覆盖",
        len(comparisonList) >= 20 and len(imageList) >= 21,
        f"comparisons={len(comparisonList)}，images={len(imageList)}",
    )
    limits = manifest.get("interpretation_limits")
    collector.record(
        "CNN 解释限制已披露",
        isinstance(limits, list) and len(limits) >= 3,
        f"interpretation_limits={0 if not isinstance(limits, list) else len(limits)}",
    )
    validateCsv(collector, cnnRoot / "style-distances.csv", "逐层 style 距离", 20)
    validateCsv(collector, cnnRoot / "spatial-distances.csv", "逐层 spatial 距离", 20)


def validateGeometryAnalysis(collector: ValidationCollector, outputRoot: Path) -> None:
    geometryRoot = outputRoot / "geometry-analysis"
    runNames = (
        "original",
        "geometry-025",
        "geometry-050",
        "geometry-075",
        "geometry-100",
        "topology-original",
        "topology-mirror-deletion",
        "topology-matched-control",
        "topology-sham-warp",
        "art-history-roi",
        "door-roi",
    )
    for runName in runNames:
        manifest = loadJsonSafely(
            collector,
            geometryRoot / runName / "geometry-analysis.json",
            f"几何分析 {runName}",
        )
        if manifest is None:
            continue
        backendUsed = manifest.get("backend_used")
        collector.record(
            f"几何分析 {runName} 使用 CNN 边缘",
            backendUsed in {"hed", "vgg"},
            f"backend_used={backendUsed}；正式结果不接受纯 Canny 主后端",
        )
        primaryGeometry = manifest.get("primary_geometry")
        lineCount = primaryGeometry.get("line_count") if isinstance(primaryGeometry, Mapping) else 0
        collector.record(
            f"几何分析 {runName} 检出线段",
            isinstance(lineCount, int) and lineCount > 0,
            f"line_count={lineCount}",
        )


def validateSubmissionArtifacts(collector: ValidationCollector, outputRoot: Path) -> None:
    tableRequirements = (
        ("cnn-feature-space.csv", 20),
        ("geometry-cnn-metrics.csv", 5),
        ("non-neural-style-cnn-metrics.csv", 5),
        ("neural-style-cnn-metrics.csv", 4),
        ("geometry-detection-metrics.csv", 5),
    )
    for filename, minimumRows in tableRequirements:
        validateCsv(
            collector,
            outputRoot / "tables" / filename,
            f"汇总表 {filename}",
            minimumRows,
        )
    collector.requireFile(outputRoot / "tables" / "analysis-tables.json", "汇总表清单", 100)

    validateCsv(
        collector,
        outputRoot / "metrics" / "image-metrics.csv",
        "定量汇总 image-metrics.csv",
        20,
    )
    validateCsv(
        collector,
        outputRoot / "metrics" / "layer-metrics.csv",
        "定量汇总 layer-metrics.csv",
        200,
    )
    collector.requireFile(outputRoot / "metrics" / "summary.json", "定量汇总清单", 500)

    figureNames = (
        "controlled-neural-matrix.png",
        "geometry-sequence.png",
        "neural-style-sequence.png",
        "topology-controls.png",
        "geometry-method.png",
        "cnn-style-spatial-feature-space.png",
        "geometry-cnn-trajectories.png",
        "neural-style-cnn-trajectories.png",
        "geometry-detection-trajectories.png",
    )
    for figureName in figureNames:
        validateImage(collector, outputRoot / "figures" / figureName, f"提交图表 {figureName}")

    metricFigureNames = (
        "style-vs-spatial-scatter.png",
        "transformation-trajectories.png",
        "selected-layer-heatmap.png",
        "topology-ablation-metrics.png",
        "neural-style-loss-curves.png",
    )
    for figureName in metricFigureNames:
        validateImage(
            collector,
            outputRoot / "metrics" / figureName,
            f"定量图表 {figureName}",
        )

    collector.requireFile(
        outputRoot / "animation" / "entering-las-meninas-heuristic-parallax.mp4",
        "视差视频",
        minimumBytes=10_000,
    )
    validateImage(
        collector,
        outputRoot / "artwork" / "entering-las-meninas-final.png",
        "最终个人作品",
        minimumSide=512,
    )


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证《宫娥》正式流水线全部可提交产物。")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = buildParser().parse_args(argv)
    collector = ValidationCollector(arguments.project_root.resolve())
    outputRoot = arguments.output_root.resolve()

    validateTransformations(collector, outputRoot)
    validateStyleManifest(
        collector,
        outputRoot / "neural-style-transfer" / "manifest.json",
        "原作神经风格迁移",
        (0.25, 0.5, 1.0),
    )
    validateStyleManifest(
        collector,
        outputRoot / "neural-style-transfer-combined" / "manifest.json",
        "几何组合神经风格迁移",
        (1.0,),
    )
    validateCnnAnalysis(collector, outputRoot)
    validateGeometryAnalysis(collector, outputRoot)
    validateSubmissionArtifacts(collector, outputRoot)

    report = {
        "status": "passed" if not collector.failures else "failed",
        # 公开验证报告只记录可移植相对路径，不暴露执行机器的用户名与目录。
        "project_root": ".",
        "output_root": relativeOrAbsolute(outputRoot, arguments.project_root),
        "check_count": len(collector.checks),
        "failure_count": len(collector.failures),
        "checks": collector.checks,
        "artifacts": collector.artifacts,
    }
    arguments.write_report.parent.mkdir(parents=True, exist_ok=True)
    with arguments.write_report.open("w", encoding="utf-8") as fileHandle:
        json.dump(report, fileHandle, ensure_ascii=False, indent=2, sort_keys=True)
        fileHandle.write("\n")

    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": report["check_count"],
                "failures": report["failure_count"],
                "report": str(arguments.write_report.resolve()),
            },
            ensure_ascii=False,
        )
    )
    if collector.failures:
        for failure in collector.failures:
            print(f"[失败] {failure['name']}：{failure['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
