"""把组合图中的原始面板与重建对比材料分门别类导出。"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn


ROOT_DIR = Path(__file__).resolve().parent
MODEL_CODE_PATH = ROOT_DIR / "autoencoder-experiment.py"
VISUALIZATION_CODE_PATH = ROOT_DIR / "network-visualization.py"
MODEL_PATH = ROOT_DIR / "model" / "geometric-autoencoder.pt"
OUTPUT_DIR = ROOT_DIR / "outputs"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
BACKGROUND_COLOR = "#f6f3ed"
TEXT_COLOR = "#202326"
MUTED_TEXT_COLOR = "#625f59"


def loadNamespaces() -> tuple[dict[str, Any], dict[str, Any]]:
    """复用主实验和网络可视化中的真实模型定义与前向传播函数。"""

    experimentNamespace = runpy.run_path(str(MODEL_CODE_PATH))
    visualizationNamespace = runpy.run_path(str(VISUALIZATION_CODE_PATH))
    return experimentNamespace, visualizationNamespace


def loadModel(
    experimentNamespace: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    modelClass = experimentNamespace["GeometricAutoencoder"]
    model = modelClass()
    model.load_state_dict(checkpoint["modelState"])
    model.eval()
    return model, checkpoint


def createDirectories() -> dict[str, Path]:
    directories = {
        "resolution": INDIVIDUAL_DIR / "01-resolution",
        "trainingAll": INDIVIDUAL_DIR / "02-training-patterns" / "all-800",
        "trainingDisplayed": (
            INDIVIDUAL_DIR / "02-training-patterns" / "displayed-samples"
        ),
        "reconstruction": INDIVIDUAL_DIR / "03-reconstruction-comparison",
        "earlyMaps": INDIVIDUAL_DIR / "04-feature-maps" / "early-layer",
        "bottleneckMaps": INDIVIDUAL_DIR / "04-feature-maps" / "bottleneck",
        "ownComparison": INDIVIDUAL_DIR / "05-artwork-comparison" / "own-work",
        "referenceComparison": (
            INDIVIDUAL_DIR / "05-artwork-comparison" / "las-meninas"
        ),
        "forwardPass": INDIVIDUAL_DIR / "06-forward-pass",
        "networkStructure": INDIVIDUAL_DIR / "07-network-structure",
        "training": INDIVIDUAL_DIR / "08-training",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def tensorToImage(imageTensor: Tensor) -> np.ndarray:
    imageArray = imageTensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(imageArray, 0.0, 1.0)


def normalizeMap(featureMap: np.ndarray) -> np.ndarray:
    minimum = float(featureMap.min())
    maximum = float(featureMap.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(featureMap)
    return (featureMap - minimum) / (maximum - minimum)


def floatRgbToPillow(imageArray: np.ndarray) -> Image.Image:
    byteArray = np.round(np.clip(imageArray, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(byteArray, mode="RGB")


def saveRgbImage(imageArray: np.ndarray, outputPath: Path) -> None:
    floatRgbToPillow(imageArray).save(outputPath, format="PNG", compress_level=6)


def saveNormalizedHeatmap(
    featureMap: np.ndarray,
    outputPath: Path,
    colorMap: str,
) -> None:
    normalizedMap = normalizeMap(featureMap)
    plt.imsave(outputPath, normalizedMap, cmap=colorMap, vmin=0.0, vmax=1.0)


def saveResolutionImages(
    experimentNamespace: dict[str, Any],
    outputDirectory: Path,
) -> None:
    ownWorkPath = experimentNamespace["OWN_WORK_PATH"]
    with Image.open(ownWorkPath) as sourceImage:
        sourceImage = sourceImage.convert("RGB")
    for resolution in (256, 128, 64, 32):
        resizedImage = sourceImage.resize(
            (resolution, resolution),
            Image.Resampling.LANCZOS,
        )
        resizedImage.save(
            outputDirectory / f"own-work-{resolution:04d}x{resolution:04d}.png",
            format="PNG",
            compress_level=6,
        )


def saveTrainingPatterns(
    experimentNamespace: dict[str, Any],
    checkpoint: dict[str, Any],
    allDirectory: Path,
    displayedDirectory: Path,
) -> None:
    sampleCount = int(checkpoint["config"]["trainSamples"])
    seed = int(checkpoint["config"]["seed"])
    dataset = experimentNamespace["buildDataset"](sampleCount, seed)
    displayedIndices = set(
        np.linspace(0, sampleCount - 1, num=12, dtype=int).tolist()
    )
    manifest = []

    for sampleIndex in range(sampleCount):
        sampleSeed = seed + sampleIndex * 17
        imageArray = dataset.samples[sampleIndex].permute(1, 2, 0).numpy()
        image = Image.fromarray(imageArray.astype(np.uint8), mode="RGB")
        fileName = f"pattern-{sampleIndex:04d}-seed-{sampleSeed:05d}.png"
        image.save(allDirectory / fileName, format="PNG", compress_level=6)
        isDisplayed = sampleIndex in displayedIndices
        if isDisplayed:
            image.save(displayedDirectory / fileName, format="PNG", compress_level=6)
        manifest.append(
            {
                "index": sampleIndex,
                "seed": sampleSeed,
                "file": f"all-800/{fileName}",
                "shownInCompositeFigure": isDisplayed,
            }
        )

    manifestPath = allDirectory.parent / "manifest.json"
    manifestPath.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def createLabeledGifFrame(imageArray: np.ndarray, label: str) -> Image.Image:
    image = floatRgbToPillow(imageArray).resize((512, 512), Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (512, 562), (246, 243, 237))
    frame.paste(image, (0, 50))
    fontPath = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    try:
        font = ImageFont.truetype(str(fontPath), 26)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(frame)
    textBounds = draw.textbbox((0, 0), label, font=font)
    textWidth = textBounds[2] - textBounds[0]
    draw.text(
        ((512 - textWidth) / 2, 10),
        label,
        fill=(32, 35, 38),
        font=font,
    )
    return frame


def saveSideBySideComparison(
    inputImage: np.ndarray,
    reconstructionImage: np.ndarray,
    metrics: dict[str, float],
    outputPath: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.8), facecolor=BACKGROUND_COLOR)
    for axis, imageArray, title in zip(
        axes,
        (inputImage, reconstructionImage),
        ("Input at 128 × 128", "Reconstruction"),
        strict=True,
    ):
        axis.imshow(imageArray)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.axis("off")
    figure.suptitle(
        "Direct Comparison: Input vs. Reconstruction",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.5,
        0.035,
        f"MSE {metrics['meanSquaredError']:.6f}  •  "
        f"MAE {metrics['meanAbsoluteError']:.6f}",
        ha="center",
        fontsize=10.5,
        color=MUTED_TEXT_COLOR,
    )
    figure.tight_layout(rect=(0.01, 0.08, 0.99, 0.91))
    figure.savefig(outputPath, dpi=190, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    plt.close(figure)


def saveDifferenceFigure(
    differenceMap: np.ndarray,
    metrics: dict[str, float],
    outputPath: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.5), facecolor=BACKGROUND_COLOR)
    image = axis.imshow(
        differenceMap,
        cmap="magma",
        vmin=0.0,
        vmax=max(0.25, float(differenceMap.max())),
    )
    axis.set_title("Absolute Reconstruction Difference", fontsize=15, fontweight="bold")
    axis.axis("off")
    colorBar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorBar.set_label("Mean absolute RGB difference (0–1)")
    figure.text(
        0.5,
        0.025,
        f"MSE {metrics['meanSquaredError']:.6f}  •  "
        f"MAE {metrics['meanAbsoluteError']:.6f}",
        ha="center",
        fontsize=10.5,
        color=MUTED_TEXT_COLOR,
    )
    figure.tight_layout(rect=(0.01, 0.07, 0.99, 0.98))
    figure.savefig(outputPath, dpi=190, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    plt.close(figure)


def saveReconstructionComparison(
    experimentNamespace: dict[str, Any],
    inputTensor: Tensor,
    reconstruction: Tensor,
    outputDirectory: Path,
) -> dict[str, float]:
    inputImage = tensorToImage(inputTensor)
    reconstructionImage = tensorToImage(reconstruction)
    differenceRgb = np.abs(inputImage - reconstructionImage)
    differenceMap = differenceRgb.mean(axis=2)
    overlayImage = inputImage * 0.5 + reconstructionImage * 0.5
    redInput = np.clip(
        inputImage[:, :, 0] - inputImage[:, :, 1:].mean(axis=2),
        0.0,
        1.0,
    )
    redOutput = np.clip(
        reconstructionImage[:, :, 0]
        - reconstructionImage[:, :, 1:].mean(axis=2),
        0.0,
        1.0,
    )
    redDifference = redInput - redOutput
    redMaximum = max(0.05, float(np.abs(redDifference).max()))
    metrics = experimentNamespace["calculateMetrics"](inputTensor, reconstruction)

    saveRgbImage(inputImage, outputDirectory / "own-work-input-128x128.png")
    saveRgbImage(
        reconstructionImage,
        outputDirectory / "own-work-reconstruction-128x128.png",
    )
    saveRgbImage(
        np.clip(differenceRgb * 4.0, 0.0, 1.0),
        outputDirectory / "own-work-absolute-difference-rgb-amplified-4x.png",
    )
    saveRgbImage(
        overlayImage,
        outputDirectory / "own-work-overlay-input-50-reconstruction-50.png",
    )
    plt.imsave(
        outputDirectory / "own-work-absolute-difference-heatmap.png",
        differenceMap,
        cmap="magma",
        vmin=0.0,
        vmax=max(0.25, float(differenceMap.max())),
    )
    plt.imsave(
        outputDirectory / "own-work-red-dominance-input-minus-output.png",
        redDifference,
        cmap="coolwarm",
        vmin=-redMaximum,
        vmax=redMaximum,
    )
    saveSideBySideComparison(
        inputImage,
        reconstructionImage,
        metrics,
        outputDirectory / "own-work-input-vs-reconstruction-side-by-side.png",
    )
    saveDifferenceFigure(
        differenceMap,
        metrics,
        outputDirectory / "own-work-difference-with-color-scale.png",
    )

    inputFrame = createLabeledGifFrame(inputImage, "INPUT")
    reconstructionFrame = createLabeledGifFrame(reconstructionImage, "RECONSTRUCTION")
    inputFrame.save(
        outputDirectory / "own-work-input-reconstruction-blink.gif",
        save_all=True,
        append_images=[reconstructionFrame],
        duration=[900, 900],
        loop=0,
        optimize=False,
    )
    metricsPath = outputDirectory / "comparison-metrics.json"
    metricsPath.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def saveFeatureMaps(
    earlyFeatures: Tensor,
    bottleneck: Tensor,
    earlyDirectory: Path,
    bottleneckDirectory: Path,
) -> None:
    for channelIndex in range(earlyFeatures.shape[1]):
        fileName = (
            f"own-work-early-map-{channelIndex + 1:02d}-of-"
            f"{earlyFeatures.shape[1]:02d}.png"
        )
        saveNormalizedHeatmap(
            earlyFeatures[0, channelIndex].numpy(),
            earlyDirectory / fileName,
            colorMap="bone",
        )

    featureVariances = earlyFeatures.squeeze(0).flatten(1).var(dim=1)
    selectedChannels = torch.argsort(featureVariances, descending=True)[:8]
    selection = {
        "selectionRule": "Eight channels with the highest spatial variance.",
        "displayOrderOneBased": [int(index) + 1 for index in selectedChannels],
    }
    (earlyDirectory / "selected-high-variance-maps.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for channelIndex in range(bottleneck.shape[1]):
        fileName = (
            f"own-work-bottleneck-channel-{channelIndex + 1:02d}-of-"
            f"{bottleneck.shape[1]:02d}.png"
        )
        saveNormalizedHeatmap(
            bottleneck[0, channelIndex].numpy(),
            bottleneckDirectory / fileName,
            colorMap="inferno",
        )
    saveNormalizedHeatmap(
        bottleneck[0].mean(dim=0).numpy(),
        bottleneckDirectory / "own-work-bottleneck-channel-mean.png",
        colorMap="inferno",
    )


def saveArtworkComparisonPanels(
    model: nn.Module,
    experimentNamespace: dict[str, Any],
    device: torch.device,
    ownInput: Tensor,
    ownDirectory: Path,
    referenceDirectory: Path,
) -> None:
    referencePath = experimentNamespace["REFERENCE_PATH"]
    _, referenceInput = experimentNamespace["loadImageForModel"](
        referencePath,
        targetSize=(112, 128),
    )
    pairs = [
        ("own-work", ownInput, ownDirectory),
        ("las-meninas", referenceInput, referenceDirectory),
    ]
    for prefix, inputTensor, outputDirectory in pairs:
        earlyFeatures, bottleneck, reconstruction = experimentNamespace[
            "getFeatureMaps"
        ](model, inputTensor, device)
        saveRgbImage(
            tensorToImage(inputTensor),
            outputDirectory / f"{prefix}-input.png",
        )
        saveRgbImage(
            tensorToImage(reconstruction),
            outputDirectory / f"{prefix}-reconstruction.png",
        )
        saveNormalizedHeatmap(
            earlyFeatures[0].mean(dim=0).numpy(),
            outputDirectory / f"{prefix}-early-response-channel-mean.png",
            colorMap="inferno",
        )
        saveNormalizedHeatmap(
            bottleneck[0].mean(dim=0).numpy(),
            outputDirectory / f"{prefix}-bottleneck-channel-mean.png",
            colorMap="inferno",
        )


def saveForwardPassPanels(
    activations: dict[str, Tensor],
    inputTensor: Tensor,
    reconstruction: Tensor,
    outputDirectory: Path,
) -> None:
    saveRgbImage(
        tensorToImage(inputTensor),
        outputDirectory / "stage-00-input-3x128x128.png",
    )
    stages = [
        ("stage-01-encoder-12x64x64-channel-mean.png", "encoder.conv1"),
        ("stage-02-encoder-24x32x32-channel-mean.png", "encoder.conv2"),
        ("stage-03-bottleneck-32x16x16-channel-mean.png", "encoder.conv3"),
        ("stage-04-decoder-24x32x32-channel-mean.png", "decoder.tconv1"),
        ("stage-05-decoder-12x64x64-channel-mean.png", "decoder.tconv2"),
    ]
    for fileName, activationName in stages:
        saveNormalizedHeatmap(
            activations[activationName][0].mean(dim=0).numpy(),
            outputDirectory / fileName,
            colorMap="magma",
        )
    saveRgbImage(
        tensorToImage(reconstruction),
        outputDirectory / "stage-06-output-3x128x128.png",
    )


def copySingleFigures(directories: dict[str, Path]) -> None:
    copies = [
        (
            OUTPUT_DIR / "06-torchview-network.png",
            directories["networkStructure"] / "torchview-exact-network.png",
        ),
        (
            OUTPUT_DIR / "06-torchview-network.svg",
            directories["networkStructure"] / "torchview-exact-network.svg",
        ),
        (
            OUTPUT_DIR / "07-paper-style-architecture.png",
            directories["networkStructure"] / "paper-style-autoencoder.png",
        ),
        (
            OUTPUT_DIR / "07-paper-style-architecture.svg",
            directories["networkStructure"] / "paper-style-autoencoder.svg",
        ),
        (
            OUTPUT_DIR / "training-loss.png",
            directories["training"] / "training-and-validation-loss.png",
        ),
    ]
    for sourcePath, destinationPath in copies:
        shutil.copy2(sourcePath, destinationPath)


def validateExports(
    directories: dict[str, Path],
    expectedMetrics: dict[str, float],
) -> None:
    expectedCounts = {
        "resolution": 4,
        "trainingAll": 800,
        "trainingDisplayed": 12,
        "reconstruction": 9,
        "earlyMaps": 12,
        "bottleneckMaps": 33,
        "ownComparison": 4,
        "referenceComparison": 4,
        "forwardPass": 7,
        "networkStructure": 4,
        "training": 1,
    }
    for directoryName, expectedCount in expectedCounts.items():
        directory = directories[directoryName]
        imageFiles = [
            path
            for path in directory.iterdir()
            if path.suffix.lower() in {".png", ".gif", ".svg"}
        ]
        if len(imageFiles) != expectedCount:
            raise RuntimeError(
                f"Unexpected image count in {directory}: "
                f"expected {expectedCount}, found {len(imageFiles)}"
            )

    savedMetrics = json.loads(
        (directories["reconstruction"] / "comparison-metrics.json").read_text()
    )
    for metricName, expectedValue in expectedMetrics.items():
        if not np.isclose(savedMetrics[metricName], expectedValue, atol=1e-10):
            raise RuntimeError(f"Metric mismatch for {metricName}")

    pngAndGifFiles = list(INDIVIDUAL_DIR.rglob("*.png")) + list(
        INDIVIDUAL_DIR.rglob("*.gif")
    )
    for imagePath in pngAndGifFiles:
        with Image.open(imagePath) as image:
            image.verify()


def main() -> None:
    directories = createDirectories()
    experimentNamespace, visualizationNamespace = loadNamespaces()
    model, checkpoint = loadModel(experimentNamespace)
    # 与主实验使用同一设备选择逻辑，尽量保证单图和组合图数值完全一致。
    device = experimentNamespace["chooseDevice"]("auto")
    model.to(device)

    saveResolutionImages(experimentNamespace, directories["resolution"])
    saveTrainingPatterns(
        experimentNamespace,
        checkpoint,
        directories["trainingAll"],
        directories["trainingDisplayed"],
    )

    _, ownInput = experimentNamespace["loadImageForModel"](
        experimentNamespace["OWN_WORK_PATH"]
    )
    earlyFeatures, bottleneck, reconstruction = experimentNamespace[
        "getFeatureMaps"
    ](model, ownInput, device)
    metrics = saveReconstructionComparison(
        experimentNamespace,
        ownInput,
        reconstruction,
        directories["reconstruction"],
    )
    saveFeatureMaps(
        earlyFeatures,
        bottleneck,
        directories["earlyMaps"],
        directories["bottleneckMaps"],
    )
    saveArtworkComparisonPanels(
        model,
        experimentNamespace,
        device,
        ownInput,
        directories["ownComparison"],
        directories["referenceComparison"],
    )
    model.cpu()
    _, activations, forwardReconstruction = visualizationNamespace[
        "collectArchitecture"
    ](model, ownInput)
    saveForwardPassPanels(
        activations,
        ownInput,
        forwardReconstruction,
        directories["forwardPass"],
    )
    copySingleFigures(directories)
    validateExports(directories, metrics)
    print(f"Individual images exported to: {INDIVIDUAL_DIR}")


if __name__ == "__main__":
    main()
