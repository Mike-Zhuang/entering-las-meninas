"""用一个小型卷积自编码器观察《被系住的观看者》中哪些结构能穿过视觉瓶颈。"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split


ROOT_DIR = Path(__file__).resolve().parent
OWN_WORK_PATH = ROOT_DIR / "My_Own_Work.png"
REFERENCE_PATH = ROOT_DIR / "inputs" / "las-meninas-reference.jpg"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "model"
SEED = 139
IMAGE_SIZE = 128
BACKGROUND_COLOR = (246, 243, 237)
INK_COLORS = ((25, 26, 28), (60, 63, 68), (105, 103, 98))
RED_COLORS = ((170, 39, 37), (198, 55, 46), (143, 32, 35))


@dataclass(frozen=True)
class ExperimentConfig:
    """集中保存实验参数，使模型文件和结果 JSON 都能记录同一组设定。"""

    imageSize: int = IMAGE_SIZE
    trainSamples: int = 800
    epochs: int = 18
    batchSize: int = 32
    learningRate: float = 0.001
    seed: int = SEED


class GeometryDataset(Dataset[Tensor]):
    """在内存中保存由程序生成的简单几何图案。"""

    def __init__(self, samples: Tensor) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> Tensor:
        return self.samples[index].float().div(255.0)


class GeometricAutoencoder(nn.Module):
    """三次降采样、三次上采样的小型卷积自编码器。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputTensor: Tensor) -> Tensor:
        return self.decoder(self.encoder(inputTensor))


def setRandomSeed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机数，便于重现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validateConfig(config: ExperimentConfig) -> None:
    """尽早拒绝无法完成训练或出图的参数，给出明确错误而非中途越界。"""

    if config.trainSamples < 12:
        raise ValueError("--samples must be at least 12.")
    if config.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if config.batchSize < 1:
        raise ValueError("--batch-size must be at least 1.")
    if config.learningRate <= 0:
        raise ValueError("--learning-rate must be greater than 0.")


def chooseDevice(requestedDevice: str) -> torch.device:
    """优先使用 Apple MPS；不可用时自动回退到 CPU。"""

    if requestedDevice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requestedDevice == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def randomPoint(rng: random.Random, margin: int = 8) -> tuple[int, int]:
    return (
        rng.randint(margin, IMAGE_SIZE - margin),
        rng.randint(margin, IMAGE_SIZE - margin),
    )


def drawPerspectiveRoom(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """生成类似房间、门框和消失点的线性结构。"""

    vanishX = rng.randint(45, 83)
    vanishY = rng.randint(40, 76)
    inkColor = rng.choice(INK_COLORS)
    lineWidth = rng.randint(2, 5)
    corners = [(5, 5), (IMAGE_SIZE - 6, 5), (5, IMAGE_SIZE - 6), (IMAGE_SIZE - 6, IMAGE_SIZE - 6)]
    for corner in corners:
        draw.line([corner, (vanishX, vanishY)], fill=inkColor, width=lineWidth)

    frameWidth = rng.randint(22, 48)
    frameHeight = rng.randint(35, 72)
    left = max(5, vanishX - frameWidth // 2 + rng.randint(-12, 12))
    top = max(5, vanishY - frameHeight // 2 + rng.randint(-10, 8))
    right = min(IMAGE_SIZE - 5, left + frameWidth)
    bottom = min(IMAGE_SIZE - 5, top + frameHeight)
    draw.rectangle((left, top, right, bottom), outline=inkColor, width=lineWidth)
    if rng.random() < 0.75:
        inset = rng.randint(4, 9)
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            outline=rng.choice(INK_COLORS),
            width=max(1, lineWidth - 1),
        )


def drawNestedFrames(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """生成画框、镜面或门洞式的嵌套矩形。"""

    centerX, centerY = randomPoint(rng, margin=32)
    maxWidth = rng.randint(44, 98)
    maxHeight = rng.randint(42, 106)
    frameCount = rng.randint(2, 5)
    inkColor = rng.choice(INK_COLORS)
    for frameIndex in range(frameCount):
        scale = 1.0 - frameIndex * rng.uniform(0.12, 0.2)
        halfWidth = int(maxWidth * scale / 2)
        halfHeight = int(maxHeight * scale / 2)
        draw.rectangle(
            (
                centerX - halfWidth,
                centerY - halfHeight,
                centerX + halfWidth,
                centerY + halfHeight,
            ),
            outline=inkColor,
            width=rng.randint(2, 5),
        )


def drawRadialStructure(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """生成从中心向外连接的拓扑结构。"""

    center = randomPoint(rng, margin=36)
    armCount = rng.randint(5, 11)
    inkColor = rng.choice(INK_COLORS)
    for armIndex in range(armCount):
        angle = (2 * math.pi * armIndex / armCount) + rng.uniform(-0.18, 0.18)
        radius = rng.randint(38, 70)
        endpoint = (
            int(center[0] + math.cos(angle) * radius),
            int(center[1] + math.sin(angle) * radius),
        )
        draw.line([center, endpoint], fill=inkColor, width=rng.randint(2, 5))
        nodeRadius = rng.randint(3, 9)
        draw.ellipse(
            (
                endpoint[0] - nodeRadius,
                endpoint[1] - nodeRadius,
                endpoint[0] + nodeRadius,
                endpoint[1] + nodeRadius,
            ),
            outline=inkColor,
            width=rng.randint(2, 4),
        )


def drawGridAndArches(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """生成窗格、地板网格与弧线，增加方向和曲率的变化。"""

    inkColor = rng.choice(INK_COLORS)
    lineWidth = rng.randint(2, 4)
    spacing = rng.randint(14, 28)
    offsetX = rng.randint(-spacing, spacing)
    offsetY = rng.randint(-spacing, spacing)
    for xPosition in range(offsetX, IMAGE_SIZE + spacing, spacing):
        draw.line([(xPosition, 0), (xPosition, IMAGE_SIZE)], fill=inkColor, width=lineWidth)
    for yPosition in range(offsetY, IMAGE_SIZE + spacing, spacing):
        draw.line([(0, yPosition), (IMAGE_SIZE, yPosition)], fill=inkColor, width=lineWidth)
    if rng.random() < 0.8:
        bounds = (
            rng.randint(4, 24),
            rng.randint(8, 38),
            rng.randint(88, 124),
            rng.randint(78, 124),
        )
        draw.arc(
            bounds,
            start=rng.randint(160, 205),
            end=rng.randint(335, 380),
            fill=inkColor,
            width=lineWidth + 1,
        )


def drawRedConnector(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """加入少量红色连接线，使训练图案能回应作品中的红色系绳。"""

    pointCount = rng.randint(3, 6)
    points = [randomPoint(rng, margin=10) for _ in range(pointCount)]
    redColor = rng.choice(RED_COLORS)
    draw.line(points, fill=redColor, width=rng.randint(3, 7), joint="curve")
    for point in (points[0], points[-1]):
        radius = rng.randint(3, 7)
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=redColor,
        )


def generatePattern(seed: int) -> Image.Image:
    """根据索引种子生成可重复的几何图案。"""

    rng = random.Random(seed)
    backgroundShift = rng.randint(-5, 6)
    background = tuple(max(0, min(255, value + backgroundShift)) for value in BACKGROUND_COLOR)
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), background)
    draw = ImageDraw.Draw(image)

    patternFunctions = [
        drawPerspectiveRoom,
        drawNestedFrames,
        drawRadialStructure,
        drawGridAndArches,
    ]
    primaryFunction = rng.choice(patternFunctions)
    primaryFunction(draw, rng)
    if rng.random() < 0.72:
        rng.choice(patternFunctions)(draw, rng)
    if rng.random() < 0.82:
        drawRedConnector(draw, rng)

    # 轻微纸张噪声避免模型只记住完全均匀的数字背景。
    imageArray = np.asarray(image, dtype=np.int16)
    noise = np.random.default_rng(seed).normal(0.0, 2.2, imageArray.shape[:2] + (1,))
    imageArray = np.clip(imageArray + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(imageArray, mode="RGB")


def buildDataset(sampleCount: int, seed: int) -> GeometryDataset:
    """预先生成 uint8 数据，兼顾训练速度与内存占用。"""

    samples = []
    for sampleIndex in range(sampleCount):
        image = generatePattern(seed + sampleIndex * 17)
        imageArray = np.asarray(image, dtype=np.uint8).copy()
        samples.append(torch.from_numpy(imageArray).permute(2, 0, 1))
    return GeometryDataset(torch.stack(samples))


def createDataLoaders(
    dataset: GeometryDataset,
    batchSize: int,
    seed: int,
) -> tuple[DataLoader[Tensor], DataLoader[Tensor]]:
    """使用固定划分创建训练集和验证集。"""

    validationSize = max(1, int(len(dataset) * 0.1))
    trainSize = len(dataset) - validationSize
    generator = torch.Generator().manual_seed(seed)
    trainDataset, validationDataset = random_split(
        dataset,
        [trainSize, validationSize],
        generator=generator,
    )
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, generator=generator)
    validationLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False)
    return trainLoader, validationLoader


def calculateLoss(
    model: nn.Module,
    dataLoader: DataLoader[Tensor],
    lossFunction: nn.Module,
    device: torch.device,
) -> float:
    """计算完整验证集平均损失。"""

    model.eval()
    totalLoss = 0.0
    totalItems = 0
    with torch.inference_mode():
        for batch in dataLoader:
            batch = batch.to(device)
            reconstruction = model(batch)
            batchLoss = lossFunction(reconstruction, batch)
            totalLoss += float(batchLoss.item()) * batch.shape[0]
            totalItems += int(batch.shape[0])
    return totalLoss / max(1, totalItems)


def trainModel(
    model: GeometricAutoencoder,
    trainLoader: DataLoader[Tensor],
    validationLoader: DataLoader[Tensor],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    """只使用均方误差训练自编码器，保持方法易于解释。"""

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningRate)
    lossFunction = nn.MSELoss()
    history = {"train": [], "validation": []}
    model.to(device)

    for epochIndex in range(config.epochs):
        model.train()
        runningLoss = 0.0
        totalItems = 0
        for batch in trainLoader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = lossFunction(reconstruction, batch)
            loss.backward()
            optimizer.step()
            runningLoss += float(loss.item()) * batch.shape[0]
            totalItems += int(batch.shape[0])

        trainLoss = runningLoss / max(1, totalItems)
        validationLoss = calculateLoss(model, validationLoader, lossFunction, device)
        history["train"].append(trainLoss)
        history["validation"].append(validationLoss)
        print(
            f"Epoch {epochIndex + 1:02d}/{config.epochs:02d} | "
            f"train={trainLoss:.6f} | validation={validationLoss:.6f}"
        )

    return history


def loadImageForModel(
    imagePath: Path,
    targetSize: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> tuple[Image.Image, Tensor]:
    """按目标尺寸缩放图像，并转成模型需要的 NCHW 张量。"""

    with Image.open(imagePath) as sourceImage:
        image = ImageOps.exif_transpose(sourceImage).convert("RGB")
    image = image.resize(targetSize, Image.Resampling.LANCZOS)
    imageArray = np.asarray(image, dtype=np.float32) / 255.0
    imageTensor = torch.from_numpy(imageArray).permute(2, 0, 1).unsqueeze(0)
    return image, imageTensor


def tensorToImage(imageTensor: Tensor) -> np.ndarray:
    """把模型张量转换为 Matplotlib 可显示的 RGB 数组。"""

    imageArray = imageTensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(imageArray, 0.0, 1.0)


def normalizeMap(featureMap: np.ndarray) -> np.ndarray:
    minimum = float(featureMap.min())
    maximum = float(featureMap.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(featureMap)
    return (featureMap - minimum) / (maximum - minimum)


def getFeatureMaps(
    model: GeometricAutoencoder,
    inputTensor: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """取第一层卷积、瓶颈与最终重建，展示网络内部信息流。"""

    model.eval()
    with torch.inference_mode():
        deviceInput = inputTensor.to(device)
        earlyFeatures = model.encoder[1](model.encoder[0](deviceInput))
        bottleneck = model.encoder(deviceInput)
        reconstruction = model.decoder(bottleneck)
    return earlyFeatures.cpu(), bottleneck.cpu(), reconstruction.cpu()


def saveResolutionLadder(imagePath: Path, outputPath: Path) -> None:
    """把同一作品降到四种分辨率，再用最近邻放大以显示信息损失。"""

    with Image.open(imagePath) as sourceImage:
        sourceImage = ImageOps.exif_transpose(sourceImage).convert("RGB")

    resolutions = [256, 128, 64, 32]
    figure, axes = plt.subplots(1, len(resolutions), figsize=(14, 4.1), facecolor="#f6f3ed")
    for axis, resolution in zip(axes, resolutions, strict=True):
        lowResolution = sourceImage.resize((resolution, resolution), Image.Resampling.LANCZOS)
        displayImage = lowResolution.resize((512, 512), Image.Resampling.NEAREST)
        axis.imshow(displayImage)
        axis.set_title(f"{resolution} × {resolution}", fontsize=13, fontweight="bold")
        axis.axis("off")
    figure.suptitle(
        "Resolution Ladder — what disappears before the CNN?",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        "At 128 × 128 the room, mirror, doorway, and red tether remain visible; "
        "most handwriting does not.",
        ha="center",
        fontsize=10.5,
        color="#4a4844",
    )
    figure.tight_layout(rect=(0.01, 0.07, 0.99, 0.92))
    figure.savefig(outputPath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def savePatternSamples(dataset: GeometryDataset, outputPath: Path) -> None:
    """展示模型真正见过的训练材料，避免把网络描述成神秘黑箱。"""

    if len(dataset) < 12:
        raise ValueError("At least 12 samples are required to create the training-set figure.")
    sampleIndices = np.linspace(0, len(dataset) - 1, num=12, dtype=int).tolist()
    figure, axes = plt.subplots(3, 4, figsize=(10, 8), facecolor="#f6f3ed")
    for axis, sampleIndex in zip(axes.flat, sampleIndices, strict=True):
        sample = dataset[sampleIndex].permute(1, 2, 0).numpy()
        axis.imshow(sample)
        axis.axis("off")
    figure.suptitle(
        "Synthetic Training Set — frames, perspective, curves, and red connections",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "The network never trains on Las Meninas or on my drawing.",
        ha="center",
        fontsize=11,
        color="#4a4844",
    )
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.93))
    figure.savefig(outputPath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def saveTrainingLoss(history: dict[str, list[float]], outputPath: Path) -> None:
    epochs = np.arange(1, len(history["train"]) + 1)
    figure, axis = plt.subplots(figsize=(8.2, 4.8), facecolor="#f6f3ed")
    axis.set_facecolor("#fbfaf7")
    axis.plot(epochs, history["train"], label="Training", color="#9f2725", linewidth=2.4)
    axis.plot(epochs, history["validation"], label="Validation", color="#2f5964", linewidth=2.4)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Training Loss", fontsize=16, fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outputPath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def saveReconstructionFigure(
    inputTensor: Tensor,
    reconstruction: Tensor,
    outputPath: Path,
) -> None:
    inputImage = tensorToImage(inputTensor)
    reconstructionImage = tensorToImage(reconstruction)
    differenceMap = np.mean(np.abs(inputImage - reconstructionImage), axis=2)
    redInput = np.clip(
        inputImage[:, :, 0] - (inputImage[:, :, 1] + inputImage[:, :, 2]) / 2.0,
        0.0,
        1.0,
    )
    redOutput = np.clip(
        reconstructionImage[:, :, 0]
        - (reconstructionImage[:, :, 1] + reconstructionImage[:, :, 2]) / 2.0,
        0.0,
        1.0,
    )

    figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.4), facecolor="#f6f3ed")
    panels = [
        (inputImage, "Input at 128 × 128", None),
        (reconstructionImage, "After encoder + decoder", None),
        (differenceMap, "What the network loses", "magma"),
        (redInput - redOutput, "Red tether: input − output", "coolwarm"),
    ]
    for axis, (panel, title, colorMap) in zip(axes, panels, strict=True):
        if colorMap is None:
            axis.imshow(panel)
        elif title.startswith("Red"):
            maximum = max(0.05, float(np.abs(panel).max()))
            axis.imshow(panel, cmap=colorMap, vmin=-maximum, vmax=maximum)
        else:
            axis.imshow(panel, cmap=colorMap, vmin=0.0, vmax=max(0.25, float(panel.max())))
        axis.set_title(title, fontsize=11.5, fontweight="bold")
        axis.axis("off")
    figure.suptitle(
        "My Work Through a Geometric Bottleneck",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "The model preserves recurring geometry more readily than handwriting, faces, "
        "or autobiographical detail.",
        ha="center",
        fontsize=10.5,
        color="#4a4844",
    )
    figure.tight_layout(rect=(0.01, 0.07, 0.99, 0.92))
    figure.savefig(outputPath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def saveFeatureMaps(
    inputTensor: Tensor,
    earlyFeatures: Tensor,
    bottleneck: Tensor,
    outputPath: Path,
) -> None:
    featureVariances = earlyFeatures.squeeze(0).flatten(1).var(dim=1)
    selectedChannels = torch.argsort(featureVariances, descending=True)[:8].tolist()

    figure = plt.figure(figsize=(14, 7.7), facecolor="#f6f3ed")
    grid = figure.add_gridspec(2, 6, height_ratios=(1, 1), hspace=0.3, wspace=0.08)

    inputAxis = figure.add_subplot(grid[0, :2])
    inputAxis.imshow(tensorToImage(inputTensor))
    inputAxis.set_title("Input to the CNN\n128 × 128 RGB", fontsize=13, fontweight="bold")
    inputAxis.axis("off")

    bottleneckMean = normalizeMap(bottleneck[0].mean(dim=0).numpy())
    bottleneckAxis = figure.add_subplot(grid[1, :2])
    bottleneckAxis.imshow(bottleneckMean, cmap="inferno", vmin=0.0, vmax=1.0)
    bottleneckAxis.set_title("Bottleneck mean\n16 × 16 × 32", fontsize=13, fontweight="bold")
    bottleneckAxis.axis("off")

    for mapIndex, channelIndex in enumerate(selectedChannels):
        rowIndex = mapIndex // 4
        columnIndex = 2 + mapIndex % 4
        axis = figure.add_subplot(grid[rowIndex, columnIndex])
        featureMap = normalizeMap(earlyFeatures[0, channelIndex].numpy())
        axis.imshow(featureMap, cmap="bone", vmin=0.0, vmax=1.0)
        axis.set_title(f"Early map {channelIndex + 1}", fontsize=9.5)
        axis.axis("off")

    figure.suptitle(
        "Inside the Encoder — selected high-variance feature maps",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.02,
        "Individual channels respond to edges, directions, frames, and color contrast; "
        "they are not named objects.",
        ha="center",
        fontsize=10.5,
        color="#4a4844",
    )
    figure.savefig(outputPath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def saveComparisonFigure(
    model: GeometricAutoencoder,
    ownInput: Tensor,
    referenceInput: Tensor,
    device: torch.device,
    outputPath: Path,
) -> None:
    ownEarly, ownBottleneck, ownReconstruction = getFeatureMaps(model, ownInput, device)
    referenceEarly, referenceBottleneck, referenceReconstruction = getFeatureMaps(
        model,
        referenceInput,
        device,
    )
    rows = [
        ("My work", ownInput, ownReconstruction, ownEarly, ownBottleneck),
        (
            "Las Meninas",
            referenceInput,
            referenceReconstruction,
            referenceEarly,
            referenceBottleneck,
        ),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(12.5, 7.1), facecolor="#f6f3ed")
    columnTitles = ["Input", "Reconstruction", "Early response", "Bottleneck mean"]

    for rowIndex, (
        rowLabel,
        inputTensor,
        reconstruction,
        earlyFeatures,
        bottleneck,
    ) in enumerate(rows):
        panels = [
            tensorToImage(inputTensor),
            tensorToImage(reconstruction),
            normalizeMap(earlyFeatures[0].mean(dim=0).numpy()),
            normalizeMap(bottleneck[0].mean(dim=0).numpy()),
        ]
        for columnIndex, panel in enumerate(panels):
            axis = axes[rowIndex, columnIndex]
            if columnIndex < 2:
                axis.imshow(panel)
            else:
                axis.imshow(panel, cmap="inferno", vmin=0.0, vmax=1.0)
            if rowIndex == 0:
                axis.set_title(columnTitles[columnIndex], fontsize=11.5, fontweight="bold")
            if columnIndex == 0:
                axis.set_ylabel(rowLabel, fontsize=12, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        "One Geometric Filter, Two Related Images",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "This comparison reveals shared frames, openings, diagonals, and depth cues—"
        "not semantic understanding or authorship.",
        ha="center",
        fontsize=10.5,
        color="#4a4844",
    )
    figure.tight_layout(rect=(0.02, 0.065, 0.98, 0.93))
    figure.savefig(outputPath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def calculateMetrics(inputTensor: Tensor, reconstruction: Tensor) -> dict[str, float]:
    inputArray = tensorToImage(inputTensor)
    reconstructionArray = tensorToImage(reconstruction)
    difference = inputArray - reconstructionArray
    meanSquaredError = float(np.mean(np.square(difference)))
    meanAbsoluteError = float(np.mean(np.abs(difference)))

    inputRed = np.clip(inputArray[:, :, 0] - inputArray[:, :, 1:].mean(axis=2), 0.0, 1.0)
    outputRed = np.clip(
        reconstructionArray[:, :, 0] - reconstructionArray[:, :, 1:].mean(axis=2),
        0.0,
        1.0,
    )
    inputRedEnergy = float(inputRed.sum())
    outputRedEnergy = float(outputRed.sum())
    redEnergyRatio = outputRedEnergy / max(1e-8, inputRedEnergy)
    return {
        "meanSquaredError": meanSquaredError,
        "meanAbsoluteError": meanAbsoluteError,
        "redDominanceEnergyInput": inputRedEnergy,
        "redDominanceEnergyOutput": outputRedEnergy,
        "redEnergyOutputToInputRatio": redEnergyRatio,
    }


def saveModel(
    model: GeometricAutoencoder,
    config: ExperimentConfig,
    history: dict[str, list[float]],
    outputPath: Path,
) -> None:
    checkpoint = {
        "modelState": model.state_dict(),
        "config": asdict(config),
        "history": history,
        "description": "Small convolutional autoencoder trained only on synthetic geometry.",
    }
    torch.save(checkpoint, outputPath)


def validateArtifacts(model: GeometricAutoencoder, config: ExperimentConfig) -> None:
    """在交付前验证网络尺寸、数值与输出文件，避免只生成半套结果。"""

    model.eval()
    testInput = torch.zeros((1, 3, config.imageSize, config.imageSize))
    with torch.inference_mode():
        bottleneck = model.encoder(testInput)
        reconstruction = model(testInput)
    if tuple(bottleneck.shape) != (1, 32, 16, 16):
        raise RuntimeError(f"Unexpected bottleneck shape: {tuple(bottleneck.shape)}")
    if tuple(reconstruction.shape) != tuple(testInput.shape):
        raise RuntimeError(f"Unexpected reconstruction shape: {tuple(reconstruction.shape)}")
    if not torch.isfinite(reconstruction).all():
        raise RuntimeError("Model output contains non-finite values.")

    expectedFiles = [
        OUTPUT_DIR / "01-resolution-ladder.png",
        OUTPUT_DIR / "02-geometric-training-patterns.png",
        OUTPUT_DIR / "03-autoencoder-reconstruction.png",
        OUTPUT_DIR / "04-feature-maps.png",
        OUTPUT_DIR / "05-las-meninas-comparison.png",
        OUTPUT_DIR / "training-loss.png",
        OUTPUT_DIR / "metrics.json",
        MODEL_DIR / "geometric-autoencoder.pt",
    ]
    missingFiles = [
        str(path)
        for path in expectedFiles
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missingFiles:
        raise RuntimeError(f"Missing or empty artifacts: {missingFiles}")


def runExperiment(config: ExperimentConfig, requestedDevice: str) -> None:
    validateConfig(config)
    setRandomSeed(config.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not OWN_WORK_PATH.is_file():
        raise FileNotFoundError(f"Missing artwork: {OWN_WORK_PATH}")
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"Missing reference: {REFERENCE_PATH}")

    device = chooseDevice(requestedDevice)
    print(f"Using device: {device}")
    print("Generating synthetic geometry...")
    dataset = buildDataset(config.trainSamples, config.seed)
    trainLoader, validationLoader = createDataLoaders(dataset, config.batchSize, config.seed)

    saveResolutionLadder(OWN_WORK_PATH, OUTPUT_DIR / "01-resolution-ladder.png")
    savePatternSamples(dataset, OUTPUT_DIR / "02-geometric-training-patterns.png")

    model = GeometricAutoencoder()
    parameterCount = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameterCount:,}")
    history = trainModel(model, trainLoader, validationLoader, config, device)
    saveTrainingLoss(history, OUTPUT_DIR / "training-loss.png")
    saveModel(model, config, history, MODEL_DIR / "geometric-autoencoder.pt")

    _, ownInput = loadImageForModel(OWN_WORK_PATH)
    earlyFeatures, bottleneck, ownReconstruction = getFeatureMaps(model, ownInput, device)
    saveReconstructionFigure(
        ownInput,
        ownReconstruction,
        OUTPUT_DIR / "03-autoencoder-reconstruction.png",
    )
    saveFeatureMaps(
        ownInput,
        earlyFeatures,
        bottleneck,
        OUTPUT_DIR / "04-feature-maps.png",
    )

    # 原作使用 112 × 128，保持其接近原始纵横比；全卷积模型可以接受这一尺寸。
    _, referenceInput = loadImageForModel(REFERENCE_PATH, targetSize=(112, 128))
    _, _, referenceReconstruction = getFeatureMaps(model, referenceInput, device)
    saveComparisonFigure(
        model,
        ownInput,
        referenceInput,
        device,
        OUTPUT_DIR / "05-las-meninas-comparison.png",
    )

    metrics = {
        "experiment": "What Survives the Bottleneck?",
        "device": str(device),
        "trainableParameters": parameterCount,
        "bottleneckShapeForOwnWork": list(bottleneck.shape),
        "config": asdict(config),
        "finalTrainingLoss": history["train"][-1],
        "finalValidationLoss": history["validation"][-1],
        "ownWork": calculateMetrics(ownInput, ownReconstruction),
        "lasMeninasReference": calculateMetrics(referenceInput, referenceReconstruction),
        "interpretationLimit": (
            "Reconstruction errors describe this model's visual compression, not artistic quality, "
            "meaning, or authorship."
        ),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    model.cpu()
    validateArtifacts(model, config)
    print(f"Experiment complete. Results saved to: {OUTPUT_DIR}")


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small geometric autoencoder and analyze My_Own_Work.png.",
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    arguments = parseArguments()
    config = ExperimentConfig(
        trainSamples=arguments.samples,
        epochs=arguments.epochs,
        batchSize=arguments.batch_size,
        learningRate=arguments.learning_rate,
    )
    runExperiment(config, arguments.device)


if __name__ == "__main__":
    main()
