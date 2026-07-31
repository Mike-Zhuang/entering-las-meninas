"""为小型几何自编码器生成自动结构图、论文式示意图和逐层前向传播图。"""

from __future__ import annotations

import colorsys
import json
import runpy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as matplotlibColors
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from PIL import Image, ImageOps
from torch import Tensor, nn
from torchview import draw_graph


ROOT_DIR = Path(__file__).resolve().parent
MODEL_CODE_PATH = ROOT_DIR / "autoencoder-experiment.py"
MODEL_PATH = ROOT_DIR / "model" / "geometric-autoencoder.pt"
OWN_WORK_PATH = ROOT_DIR / "My_Own_Work.png"
OUTPUT_DIR = ROOT_DIR / "outputs"
IMAGE_SIZE = 128
BACKGROUND_COLOR = "#f6f3ed"
TEXT_COLOR = "#202326"
MUTED_TEXT_COLOR = "#625f59"
ENCODER_COLOR = "#315d67"
BOTTLENECK_COLOR = "#a63b36"
DECODER_COLOR = "#b08a42"


def loadModelDefinition() -> type[nn.Module]:
    """从主实验脚本读取模型类，避免在可视化脚本中复制网络结构。"""

    namespace = runpy.run_path(str(MODEL_CODE_PATH))
    modelClass = namespace.get("GeometricAutoencoder")
    if modelClass is None:
        raise RuntimeError("GeometricAutoencoder was not found in the experiment script.")
    return modelClass


def loadTrainedModel() -> tuple[nn.Module, dict[str, Any]]:
    """加载已经训练并验证过的模型权重。"""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")
    modelClass = loadModelDefinition()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = modelClass()
    model.load_state_dict(checkpoint["modelState"])
    model.eval()
    return model, checkpoint


def loadOwnWork() -> Tensor:
    """按主实验的输入尺寸读取作品。"""

    with Image.open(OWN_WORK_PATH) as sourceImage:
        image = ImageOps.exif_transpose(sourceImage).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    imageArray = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(imageArray).permute(2, 0, 1).unsqueeze(0)


def tensorToImage(imageTensor: Tensor) -> np.ndarray:
    imageArray = imageTensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(imageArray, 0.0, 1.0)


def normalizeMap(featureMap: np.ndarray) -> np.ndarray:
    minimum = float(featureMap.min())
    maximum = float(featureMap.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(featureMap)
    return (featureMap - minimum) / (maximum - minimum)


def moduleParameterCount(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def collectArchitecture(
    model: nn.Module,
    inputTensor: Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Tensor], Tensor]:
    """用一次真实前向传播记录每层输入、输出、参数和激活。"""

    layerDefinitions = [
        ("encoder.conv1", model.encoder[0], model.encoder[1], "ReLU"),
        ("encoder.conv2", model.encoder[2], model.encoder[3], "ReLU"),
        ("encoder.conv3", model.encoder[4], model.encoder[5], "ReLU"),
        ("decoder.tconv1", model.decoder[0], model.decoder[1], "ReLU"),
        ("decoder.tconv2", model.decoder[2], model.decoder[3], "ReLU"),
        ("decoder.tconv3", model.decoder[4], model.decoder[5], "Sigmoid"),
    ]
    records: dict[str, dict[str, Any]] = {}
    activations: dict[str, Tensor] = {}
    handles = []

    for layerName, operationModule, activationModule, activationName in layerDefinitions:
        def operationHook(
            module: nn.Module,
            moduleInputs: tuple[Tensor, ...],
            moduleOutput: Tensor,
            currentName: str = layerName,
            currentActivation: str = activationName,
        ) -> None:
            kernelSize = tuple(int(value) for value in module.kernel_size)
            stride = tuple(int(value) for value in module.stride)
            records[currentName] = {
                "name": currentName,
                "type": type(module).__name__,
                "kernelSize": list(kernelSize),
                "stride": list(stride),
                "inputShape": list(moduleInputs[0].shape),
                "outputShape": list(moduleOutput.shape),
                "activation": currentActivation,
                "parameters": moduleParameterCount(module),
            }

        def activationHook(
            _module: nn.Module,
            _moduleInputs: tuple[Tensor, ...],
            moduleOutput: Tensor,
            currentName: str = layerName,
        ) -> None:
            activations[currentName] = moduleOutput.detach().cpu()

        handles.append(operationModule.register_forward_hook(operationHook))
        handles.append(activationModule.register_forward_hook(activationHook))

    with torch.inference_mode():
        reconstruction = model(inputTensor)
    for handle in handles:
        handle.remove()

    orderedRecords = [records[layerName] for layerName, *_ in layerDefinitions]
    return orderedRecords, activations, reconstruction.detach().cpu()


def saveArchitectureMetadata(
    records: list[dict[str, Any]],
    model: nn.Module,
    checkpoint: dict[str, Any],
) -> None:
    metadata = {
        "model": type(model).__name__,
        "inputShapeConvention": "NCHW",
        "layers": records,
        "totalTrainableParameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainingConfig": checkpoint["config"],
        "note": (
            "Shapes were recorded from a real forward pass. Activations are ReLU except for "
            "the final Sigmoid output."
        ),
    }
    outputPath = OUTPUT_DIR / "network-architecture.json"
    outputPath.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def saveTorchviewGraph(model: nn.Module) -> None:
    """用 torchview 自动追踪真实模型，生成可核查的 SVG 与 PNG。"""

    modelGraph = draw_graph(
        model,
        input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        device="cpu",
        depth=3,
        expand_nested=True,
        graph_dir="LR",
        show_shapes=True,
        hide_inner_tensors=True,
        hide_module_functions=True,
    )
    graph = modelGraph.visual_graph
    graph.graph_attr.update(
        {
            "bgcolor": BACKGROUND_COLOR,
            "fontname": "Helvetica",
            "fontsize": "18",
            "fontcolor": TEXT_COLOR,
            "label": (
                "GeometricAutoencoder — exact forward graph\n"
                "Input: 1 × 3 × 128 × 128  |  Parameters: 27,983"
            ),
            "labelloc": "t",
            "labeljust": "l",
            "pad": "0.35",
            "ranksep": "0.48",
            "nodesep": "0.22",
            "dpi": "300",
        }
    )
    graph.node_attr.update({"fontname": "Helvetica", "fontsize": "10"})
    graph.edge_attr.update({"color": "#68645d", "penwidth": "1.2"})

    outputStem = str(OUTPUT_DIR / "06-torchview-network")
    graph.format = "svg"
    graph.render(filename=outputStem, cleanup=True)
    graph.format = "png"
    graph.render(filename=outputStem, cleanup=True)


def adjustColor(color: str, lightnessMultiplier: float) -> tuple[float, float, float]:
    red, green, blue = matplotlibColors.to_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    adjustedLightness = max(0.0, min(1.0, lightness * lightnessMultiplier))
    return colorsys.hls_to_rgb(hue, adjustedLightness, saturation)


def drawTensorBlock(
    axis: plt.Axes,
    centerX: float,
    centerY: float,
    width: float,
    height: float,
    depthX: float,
    depthY: float,
    color: str,
) -> tuple[float, float]:
    """用三个面画出特征张量，深度只表示通道数量的相对变化。"""

    left = centerX - width / 2
    bottom = centerY - height / 2
    right = left + width
    top = bottom + height
    topFace = Polygon(
        [(left, top), (right, top), (right + depthX, top + depthY), (left + depthX, top + depthY)],
        closed=True,
        facecolor=adjustColor(color, 1.25),
        edgecolor=TEXT_COLOR,
        linewidth=0.8,
        zorder=3,
    )
    sideFace = Polygon(
        [
            (right, bottom),
            (right, top),
            (right + depthX, top + depthY),
            (right + depthX, bottom + depthY),
        ],
        closed=True,
        facecolor=adjustColor(color, 0.78),
        edgecolor=TEXT_COLOR,
        linewidth=0.8,
        zorder=3,
    )
    frontFace = Rectangle(
        (left, bottom),
        width,
        height,
        facecolor=color,
        edgecolor=TEXT_COLOR,
        linewidth=1.1,
        zorder=4,
    )
    axis.add_patch(topFace)
    axis.add_patch(sideFace)
    axis.add_patch(frontFace)
    return left, right + depthX


def drawImageTensor(
    axis: plt.Axes,
    imageArray: np.ndarray,
    centerX: float,
    centerY: float,
    width: float,
    height: float,
    borderColor: str,
) -> tuple[float, float]:
    left = centerX - width / 2
    right = centerX + width / 2
    bottom = centerY - height / 2
    top = centerY + height / 2
    axis.imshow(imageArray, extent=(left, right, bottom, top), zorder=4)
    axis.add_patch(
        Rectangle(
            (left, bottom),
            width,
            height,
            facecolor="none",
            edgecolor=borderColor,
            linewidth=1.8,
            zorder=5,
        )
    )
    return left, right


def drawFlowArrow(
    axis: plt.Axes,
    startX: float,
    endX: float,
    yPosition: float,
    operation: str,
    parameterCount: int,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (startX, yPosition),
            (endX, yPosition),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=TEXT_COLOR,
            zorder=2,
        )
    )
    axis.text(
        (startX + endX) / 2,
        yPosition + 0.42,
        operation,
        ha="center",
        va="bottom",
        fontsize=6.9,
        fontweight="bold",
        color=TEXT_COLOR,
        linespacing=1.2,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": BACKGROUND_COLOR,
            "edgecolor": "none",
            "alpha": 0.94,
        },
        zorder=6,
    )
    axis.text(
        (startX + endX) / 2,
        yPosition - 0.38,
        f"{parameterCount:,} P",
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTED_TEXT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": BACKGROUND_COLOR,
            "edgecolor": "none",
            "alpha": 0.94,
        },
        zorder=6,
    )


def savePaperArchitecture(
    records: list[dict[str, Any]],
    inputTensor: Tensor,
    reconstruction: Tensor,
) -> None:
    """生成用于课堂或 PPT 的论文式编码器—解码器示意图。"""

    inputImage = tensorToImage(inputTensor)
    outputImage = tensorToImage(reconstruction)
    figure, axis = plt.subplots(figsize=(16, 9), facecolor=BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 9.0)
    axis.axis("off")

    axis.text(
        0.45,
        8.55,
        "WHAT SURVIVES THE BOTTLENECK?",
        fontsize=24,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    axis.text(
        0.48,
        8.12,
        "A 27,983-parameter geometric convolutional autoencoder",
        fontsize=11.5,
        color=MUTED_TEXT_COLOR,
        ha="left",
        va="top",
    )
    axis.plot([0.48, 15.52], [7.76, 7.76], color="#cfc9be", linewidth=1.0)
    axis.text(4.7, 7.42, "ENCODER  /  COMPRESSION", ha="center", fontsize=10, fontweight="bold")
    axis.text(11.1, 7.42, "DECODER  /  RECONSTRUCTION", ha="center", fontsize=10, fontweight="bold")

    centerY = 4.55
    stageCenters = [1.25, 3.35, 5.35, 7.65, 9.95, 11.95, 14.75]
    boundaries: list[tuple[float, float]] = []
    boundaries.append(
        drawImageTensor(
            axis,
            inputImage,
            centerX=stageCenters[0],
            centerY=centerY,
            width=1.75,
            height=3.15,
            borderColor=ENCODER_COLOR,
        )
    )
    boundaries.append(
        drawTensorBlock(axis, stageCenters[1], centerY, 0.7, 2.65, 0.25, 0.2, ENCODER_COLOR)
    )
    boundaries.append(
        drawTensorBlock(axis, stageCenters[2], centerY, 0.7, 2.05, 0.42, 0.28, ENCODER_COLOR)
    )
    boundaries.append(
        drawTensorBlock(axis, stageCenters[3], centerY, 0.72, 1.45, 0.58, 0.36, BOTTLENECK_COLOR)
    )
    boundaries.append(
        drawTensorBlock(axis, stageCenters[4], centerY, 0.7, 2.05, 0.42, 0.28, DECODER_COLOR)
    )
    boundaries.append(
        drawTensorBlock(axis, stageCenters[5], centerY, 0.7, 2.65, 0.25, 0.2, DECODER_COLOR)
    )
    boundaries.append(
        drawImageTensor(
            axis,
            outputImage,
            centerX=stageCenters[6],
            centerY=centerY,
            width=1.75,
            height=3.15,
            borderColor=DECODER_COLOR,
        )
    )

    operations = [
        "CONV 5×5\nS2 • RELU",
        "CONV 3×3\nS2 • RELU",
        "CONV 3×3\nS2 • RELU",
        "TCONV 4×4\nS2 • RELU",
        "TCONV 4×4\nS2 • RELU",
        "TCONV 4×4\nS2 • SIGMOID",
    ]
    for operationIndex, operation in enumerate(operations):
        drawFlowArrow(
            axis,
            boundaries[operationIndex][1] + 0.12,
            boundaries[operationIndex + 1][0] - 0.12,
            centerY,
            operation,
            records[operationIndex]["parameters"],
        )

    tensorLabels = [
        ("MY WORK", "3 × 128 × 128"),
        ("FEATURES 1", "12 × 64 × 64"),
        ("FEATURES 2", "24 × 32 × 32"),
        ("BOTTLENECK", "32 × 16 × 16"),
        ("FEATURES 4", "24 × 32 × 32"),
        ("FEATURES 5", "12 × 64 × 64"),
        ("RECONSTRUCTION", "3 × 128 × 128"),
    ]
    for stageIndex, (stageName, stageShape) in enumerate(tensorLabels):
        labelColor = BOTTLENECK_COLOR if stageIndex == 3 else TEXT_COLOR
        axis.text(
            stageCenters[stageIndex],
            2.45,
            stageName,
            fontsize=8.5,
            fontweight="bold",
            color=labelColor,
            ha="center",
        )
        axis.text(
            stageCenters[stageIndex],
            2.16,
            stageShape,
            fontsize=8.3,
            color=MUTED_TEXT_COLOR,
            ha="center",
        )

    axis.text(
        stageCenters[3],
        6.68,
        "VISUAL BOTTLENECK",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=BOTTLENECK_COLOR,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3dfdb", "edgecolor": "none"},
    )
    axis.annotate(
        "compresses recurring visual structure\nnot autobiographical meaning",
        xy=(stageCenters[3] + 0.25, 5.38),
        xytext=(stageCenters[3] + 0.05, 6.2),
        fontsize=8.3,
        color=MUTED_TEXT_COLOR,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": BOTTLENECK_COLOR, "linewidth": 1.0},
    )

    axis.plot([0.48, 15.52], [1.55, 1.55], color="#cfc9be", linewidth=1.0)
    axis.text(
        0.5,
        1.16,
        "ONE FORWARD PASS",
        fontsize=8.5,
        fontweight="bold",
        color=BOTTLENECK_COLOR,
        ha="left",
    )
    axis.text(
        2.45,
        1.16,
        "49,152 input values → 8,192 bottleneck activations → 49,152 reconstructed values",
        fontsize=9.4,
        color=TEXT_COLOR,
        ha="left",
    )
    axis.text(
        2.45,
        0.72,
        "No classifier • no semantic labels • no skip connections • "
        "trained only on synthetic geometry",
        fontsize=9.4,
        color=MUTED_TEXT_COLOR,
        ha="left",
    )

    for extension in ("png", "svg"):
        figure.savefig(
            OUTPUT_DIR / f"07-paper-style-architecture.{extension}",
            dpi=190,
            bbox_inches="tight",
            facecolor=BACKGROUND_COLOR,
        )
    plt.close(figure)


def saveForwardPass(
    inputTensor: Tensor,
    activations: dict[str, Tensor],
    reconstruction: Tensor,
) -> None:
    """把作品在每个空间尺度上的真实激活按前向顺序排成一行。"""

    panels = [
        ("Input", "3 × 128 × 128", tensorToImage(inputTensor), "rgb"),
        (
            "Encoder 1",
            "12 × 64 × 64",
            normalizeMap(activations["encoder.conv1"][0].mean(dim=0).numpy()),
            "map",
        ),
        (
            "Encoder 2",
            "24 × 32 × 32",
            normalizeMap(activations["encoder.conv2"][0].mean(dim=0).numpy()),
            "map",
        ),
        (
            "Bottleneck",
            "32 × 16 × 16",
            normalizeMap(activations["encoder.conv3"][0].mean(dim=0).numpy()),
            "bottleneck",
        ),
        (
            "Decoder 1",
            "24 × 32 × 32",
            normalizeMap(activations["decoder.tconv1"][0].mean(dim=0).numpy()),
            "map",
        ),
        (
            "Decoder 2",
            "12 × 64 × 64",
            normalizeMap(activations["decoder.tconv2"][0].mean(dim=0).numpy()),
            "map",
        ),
        ("Output", "3 × 128 × 128", tensorToImage(reconstruction), "rgb"),
    ]
    figure, axes = plt.subplots(1, 7, figsize=(18, 4.35), facecolor=BACKGROUND_COLOR)
    for axisIndex, (axis, (stageName, shapeText, panel, panelType)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        if panelType == "rgb":
            axis.imshow(panel)
        elif panelType == "bottleneck":
            axis.imshow(panel, cmap="magma", vmin=0.0, vmax=1.0)
            for spine in axis.spines.values():
                spine.set_edgecolor(BOTTLENECK_COLOR)
                spine.set_linewidth(3.0)
        else:
            axis.imshow(panel, cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(stageName, fontsize=11.5, fontweight="bold", pad=10)
        axis.text(
            0.5,
            -0.08,
            shapeText,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=MUTED_TEXT_COLOR,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        if axisIndex < len(axes) - 1:
            axis.text(
                1.08,
                0.5,
                "→",
                transform=axis.transAxes,
                fontsize=21,
                color=TEXT_COLOR,
                ha="center",
                va="center",
                clip_on=False,
            )

    figure.suptitle(
        "The Tethered Viewer — one real forward pass through every scale",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.98,
    )
    figure.text(
        0.5,
        0.035,
        "Heatmaps show the mean activation across channels and are normalized independently. "
        "They are not attention maps or semantic explanations.",
        ha="center",
        fontsize=10,
        color=MUTED_TEXT_COLOR,
    )
    figure.tight_layout(rect=(0.01, 0.09, 0.99, 0.9), w_pad=1.35)
    figure.savefig(
        OUTPUT_DIR / "08-actual-forward-pass.png",
        dpi=190,
        bbox_inches="tight",
        facecolor=BACKGROUND_COLOR,
    )
    plt.close(figure)


def validateOutputs(records: list[dict[str, Any]], model: nn.Module) -> None:
    totalParameters = sum(parameter.numel() for parameter in model.parameters())
    recordedParameters = sum(int(record["parameters"]) for record in records)
    if totalParameters != 27_983 or recordedParameters != totalParameters:
        raise RuntimeError(
            f"Parameter mismatch: model={totalParameters}, recorded={recordedParameters}"
        )
    expectedShapes = [
        [1, 12, 64, 64],
        [1, 24, 32, 32],
        [1, 32, 16, 16],
        [1, 24, 32, 32],
        [1, 12, 64, 64],
        [1, 3, 128, 128],
    ]
    actualShapes = [record["outputShape"] for record in records]
    if actualShapes != expectedShapes:
        raise RuntimeError(f"Unexpected layer shapes: {actualShapes}")

    expectedFiles = [
        OUTPUT_DIR / "06-torchview-network.svg",
        OUTPUT_DIR / "06-torchview-network.png",
        OUTPUT_DIR / "07-paper-style-architecture.svg",
        OUTPUT_DIR / "07-paper-style-architecture.png",
        OUTPUT_DIR / "08-actual-forward-pass.png",
        OUTPUT_DIR / "network-architecture.json",
    ]
    invalidFiles = [
        str(path)
        for path in expectedFiles
        if not path.is_file() or path.stat().st_size == 0
    ]
    if invalidFiles:
        raise RuntimeError(f"Missing or empty visualization files: {invalidFiles}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, checkpoint = loadTrainedModel()
    inputTensor = loadOwnWork()
    records, activations, reconstruction = collectArchitecture(model, inputTensor)
    saveArchitectureMetadata(records, model, checkpoint)
    saveTorchviewGraph(model)
    savePaperArchitecture(records, inputTensor, reconstruction)
    saveForwardPass(inputTensor, activations, reconstruction)
    validateOutputs(records, model)
    print(f"Network visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
