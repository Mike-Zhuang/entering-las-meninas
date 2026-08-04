"""生成并执行 Second Delivery 所需的两份 Jupyter Notebook。"""

from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT_DIR / "submission"
ASSET_DIR = SUBMISSION_DIR / "assets"
FIGURE_DIR = SUBMISSION_DIR / "figures"
KERNEL_NAME = "cogsci-oh-submission"


def markdownCell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip())


def codeCell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


def notebookMetadata() -> dict[str, object]:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }


def sharedSetupCells() -> list[nbformat.NotebookNode]:
    return [
        markdownCell(
            """
## Setup

The notebook uses the same compact model and trained checkpoint as the project experiment. It does not require `torchvision`; only PyTorch, Pillow, NumPy, and Matplotlib are used.
"""
        ),
        codeCell(
            """
%matplotlib inline

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

SEED = 139
IMAGE_SIZE = 128
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Python/PyTorch environment ready: torch {torch.__version__}")
print(f"Fixed random seed: {SEED}")
"""
        ),
        markdownCell(
            """
### Load the two submission assets

The notebook reads a small 128 × 128 copy of my artwork and the trained checkpoint from the neighboring `assets/` folder. Keeping binary data outside the notebook makes the Python code readable.
"""
        ),
        codeCell(
            """
def findAsset(fileName: str) -> Path:
    searchRoots = [
        Path.cwd(),
        Path.cwd() / "submission",
        Path.cwd() / "OH_Programming" / "submission",
    ]
    for searchRoot in searchRoots:
        candidatePath = searchRoot / "assets" / fileName
        if candidatePath.is_file():
            return candidatePath.resolve()
    raise FileNotFoundError(
        f"Could not locate assets/{fileName}. Keep the assets folder beside the notebooks."
    )


artworkPath = findAsset("my-own-work-128.png")
checkpointPath = findAsset("geometric-autoencoder.pt")
print(f"Artwork:   {artworkPath.name}")
print(f"Checkpoint: {checkpointPath.name}")
"""
        ),
        markdownCell(
            """
### Model definition

The encoder lowers spatial resolution three times: `128 → 64 → 32 → 16`. The decoder then returns to `128 × 128`. There is no classifier, semantic label, attention mechanism, or skip connection.
"""
        ),
        codeCell(
            """
class GeometricAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputTensor: torch.Tensor) -> torch.Tensor:
        encodedTensor = self.encoder(inputTensor)
        return self.decoder(encodedTensor)


try:
    checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpointPath, map_location="cpu")

model = GeometricAutoencoder()
model.load_state_dict(checkpoint["modelState"])
model.eval()

parameterCount = sum(parameter.numel() for parameter in model.parameters())
print(model)
print(f"Trainable parameters: {parameterCount:,}")
print(f"Training configuration: {checkpoint['config']}")
"""
        ),
        codeCell(
            """
with Image.open(artworkPath) as sourceImage:
    artworkImage = ImageOps.exif_transpose(sourceImage).convert("RGB")

artworkArray = np.asarray(artworkImage, dtype=np.float32) / 255.0
inputTensor = torch.from_numpy(artworkArray).permute(2, 0, 1).unsqueeze(0)

print(f"Input image size: {artworkImage.size}")
print(f"Input tensor shape (NCHW): {tuple(inputTensor.shape)}")
print(f"Input range: {inputTensor.min().item():.4f} to {inputTensor.max().item():.4f}")
"""
        ),
    ]


def buildLayerNotebook() -> nbformat.NotebookNode:
    cells = [
        markdownCell(
            """
# Following *The Tethered Viewer* Through Every CNN Layer

**COGSCI 139 — Art, Geometry, and Cognition · Second Delivery**

## TL;DR

This notebook passes my selected work, *The Tethered Viewer*, through the trained convolutional autoencoder once and records the output of **all 12 leaf layers**: three encoder convolutions, three encoder activations, three decoder transposed convolutions, two decoder ReLUs, and the final Sigmoid. The early layers retain edges and room divisions; the `32 × 16 × 16` bottleneck compresses them into coarse spatial regions; the decoder expands those regions but cannot restore handwriting or personal meaning.
"""
        ),
        markdownCell(
            """
## Context and method

The instructor asked us to examine the output of all layers as a figure is processed by a neural network. Forward hooks are attached to every leaf module. Each hook saves that module's real numerical output during one inference pass. The visualizations below show channel means normalized within each panel, so color indicates relative response inside a layer—not semantic attention and not directly comparable absolute strength across layers.

### Key assumptions

- The checkpoint was trained for 18 epochs on 800 programmatically generated geometric images.
- The network never trained on my artwork or on *Las Meninas*.
- The image is reduced to `128 × 128` before entering the network.
"""
        ),
        *sharedSetupCells(),
        markdownCell(
            """
## Results: capture every layer

Hooks are attached to the 12 modules that actually perform operations. The order below is the true forward-pass order.
"""
        ),
        codeCell(
            """
layerOutputs = {}
hookHandles = []


def makeHook(layerName: str):
    def captureOutput(_module, _inputs, moduleOutput):
        layerOutputs[layerName] = moduleOutput.detach().cpu().clone()
    return captureOutput


leafLayers = [
    (layerName, layerModule)
    for layerName, layerModule in model.named_modules()
    if layerName and not list(layerModule.children())
]

for layerName, layerModule in leafLayers:
    hookHandles.append(layerModule.register_forward_hook(makeHook(layerName)))

with torch.inference_mode():
    reconstructionTensor = model(inputTensor)

for hookHandle in hookHandles:
    hookHandle.remove()

print(f"Forward pass completed: captured {len(layerOutputs)} of {len(leafLayers)} leaf layers.")
print(f"Final output shape: {tuple(reconstructionTensor.shape)}")
"""
        ),
        markdownCell(
            """
### Numerical summary

The table makes the compression path explicit. Negative minima occur before ReLU; the following ReLU clips those values to zero. The final Sigmoid restricts reconstructed RGB values to `[0, 1]`.
"""
        ),
        codeCell(
            """
header = f"{'#':>2}  {'layer':<12} {'type':<16} {'output shape':<22} {'min':>9} {'max':>9} {'mean':>9} {'std':>9}"
print(header)
print("-" * len(header))

for layerIndex, (layerName, layerModule) in enumerate(leafLayers, start=1):
    outputTensor = layerOutputs[layerName]
    print(
        f"{layerIndex:>2}  {layerName:<12} {type(layerModule).__name__:<16} "
        f"{str(tuple(outputTensor.shape)):<22} "
        f"{outputTensor.min().item():>9.4f} {outputTensor.max().item():>9.4f} "
        f"{outputTensor.mean().item():>9.4f} {outputTensor.std().item():>9.4f}"
    )
"""
        ),
        markdownCell(
            """
### Visual summary of all 12 layer outputs

Every panel below comes from the same forward pass. For multichannel tensors, the panel is the mean across channels; the final three-channel Sigmoid output is displayed as RGB.
"""
        ),
        codeCell(
            """
def normalizeMap(featureMap: np.ndarray) -> np.ndarray:
    minimumValue = float(featureMap.min())
    maximumValue = float(featureMap.max())
    if maximumValue - minimumValue < 1e-8:
        return np.zeros_like(featureMap)
    return (featureMap - minimumValue) / (maximumValue - minimumValue)


figure, axes = plt.subplots(3, 4, figsize=(15, 11), facecolor="#f6f3ed")
for layerIndex, (axis, (layerName, layerModule)) in enumerate(
    zip(axes.flat, leafLayers, strict=True), start=1
):
    outputTensor = layerOutputs[layerName][0]
    if layerName == "decoder.5":
        panelImage = outputTensor.permute(1, 2, 0).numpy()
        axis.imshow(np.clip(panelImage, 0.0, 1.0))
    else:
        channelMean = outputTensor.mean(dim=0).numpy()
        axis.imshow(normalizeMap(channelMean), cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title(
        f"{layerIndex}. {layerName} · {type(layerModule).__name__}\\n{tuple(outputTensor.shape)}",
        fontsize=10,
        fontweight="bold",
    )
    axis.axis("off")

figure.suptitle(
    "The Tethered Viewer: output of every CNN layer",
    fontsize=18,
    fontweight="bold",
)
figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
plt.show()
"""
        ),
        markdownCell(
            """
### Individual channels in the first activated feature tensor

The mean maps above compress each tensor into one image. This second view shows that channels within a single layer respond differently to the same artwork. They separate local contrasts and directions without assigning natural-language meanings such as “mirror” or “person.”
"""
        ),
        codeCell(
            """
earlyTensor = layerOutputs["encoder.1"][0]
figure, axes = plt.subplots(3, 4, figsize=(11, 8.5), facecolor="#f6f3ed")
for channelIndex, axis in enumerate(axes.flat):
    channelMap = earlyTensor[channelIndex].numpy()
    axis.imshow(normalizeMap(channelMap), cmap="bone", vmin=0.0, vmax=1.0)
    axis.set_title(f"Channel {channelIndex + 1}", fontsize=10)
    axis.axis("off")

figure.suptitle(
    "All 12 channels after the first Conv2d + ReLU",
    fontsize=17,
    fontweight="bold",
)
figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
plt.show()
"""
        ),
        markdownCell(
            """
## Checks

The following assertions make the notebook fail visibly if a layer is missing, a tensor has an unexpected shape, or any output contains `NaN`/infinite values.
"""
        ),
        codeCell(
            """
expectedLayerNames = [
    "encoder.0", "encoder.1", "encoder.2", "encoder.3", "encoder.4", "encoder.5",
    "decoder.0", "decoder.1", "decoder.2", "decoder.3", "decoder.4", "decoder.5",
]
expectedShapes = [
    (1, 12, 64, 64), (1, 12, 64, 64),
    (1, 24, 32, 32), (1, 24, 32, 32),
    (1, 32, 16, 16), (1, 32, 16, 16),
    (1, 24, 32, 32), (1, 24, 32, 32),
    (1, 12, 64, 64), (1, 12, 64, 64),
    (1, 3, 128, 128), (1, 3, 128, 128),
]

assert list(layerOutputs) == expectedLayerNames
assert [tuple(layerOutputs[name].shape) for name in expectedLayerNames] == expectedShapes
assert all(torch.isfinite(layerOutputs[name]).all() for name in expectedLayerNames)
assert tuple(reconstructionTensor.shape) == (1, 3, 128, 128)

print("PASS: all 12 layer outputs were captured in the correct order.")
print("PASS: every tensor has the expected shape and contains only finite values.")
print("PASS: final reconstruction returns to the input shape, 1 × 3 × 128 × 128.")
"""
        ),
        markdownCell(
            """
## Takeaways

1. The network progressively changes the image from RGB pixels into multiple feature channels while reducing spatial resolution.
2. ReLU is visible as an operation: negative convolution responses are removed before the next convolution.
3. At the bottleneck, the drawing is represented by `32 × 16 × 16` activations. The arrangement remains spatial, but fine writing and contours are no longer recoverable.
4. The decoder enlarges the compressed representation into a plausible geometric reconstruction. It does not retrieve meaning from a hidden label.
5. A CNN can separate visual regularities across channels, but the interpretation of those patterns remains a human act.
"""
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=notebookMetadata())


def buildReconstructionNotebook() -> nbformat.NotebookNode:
    cells = [
        markdownCell(
            """
# Reconstructing *The Tethered Viewer* with a Convolutional Autoencoder

**COGSCI 139 — Art, Geometry, and Cognition · Second Delivery**

## TL;DR

This notebook uses a small trained convolutional autoencoder to encode and reconstruct my selected work, *The Tethered Viewer*. The reconstruction preserves the broad room frame, central rectangle, foreground grouping, and some line directions, but it blurs handwriting and weakens or diffuses the red tether. The measured reconstruction error is reported below; it describes pixel loss in this model, not artistic quality or meaning.
"""
        ),
        markdownCell(
            """
## Context and method

The model was trained on 800 simple geometric images generated from perspective lines, nested frames, grids, arcs, radial connections, and occasional red connectors. It never trained on my artwork. The purpose is therefore not to imitate a style, but to ask which visual structures survive an encoder–decoder bottleneck.

### Key assumptions

- Input resolution is fixed at `128 × 128`.
- The checkpoint contains the weights learned during 18 training epochs.
- Mean squared error compares pixels; it cannot measure memory, authorship, or artistic value.
"""
        ),
        *sharedSetupCells(),
        markdownCell(
            """
## Results: encode and reconstruct

The encoder converts the input into a `32 × 16 × 16` bottleneck. The decoder expands that representation back into an RGB image of the original model resolution.
"""
        ),
        codeCell(
            """
with torch.inference_mode():
    bottleneckTensor = model.encoder(inputTensor)
    reconstructionTensor = model.decoder(bottleneckTensor)

inputArray = inputTensor[0].permute(1, 2, 0).numpy()
reconstructionArray = reconstructionTensor[0].permute(1, 2, 0).numpy()
absoluteDifference = np.abs(inputArray - reconstructionArray)
differenceMap = absoluteDifference.mean(axis=2)

meanSquaredError = float(np.mean(np.square(inputArray - reconstructionArray)))
meanAbsoluteError = float(np.mean(absoluteDifference))
peakSignalToNoiseRatio = float(10.0 * np.log10(1.0 / max(meanSquaredError, 1e-12)))

print(f"Bottleneck shape:      {tuple(bottleneckTensor.shape)}")
print(f"Reconstruction shape:  {tuple(reconstructionTensor.shape)}")
print(f"Mean squared error:    {meanSquaredError:.6f}")
print(f"Mean absolute error:   {meanAbsoluteError:.6f}")
print(f"PSNR:                  {peakSignalToNoiseRatio:.2f} dB")
"""
        ),
        markdownCell(
            """
### Input, reconstruction, and loss

The absolute-difference panel is brightest where the model changes the image most. Fine writing, irregular figure contours, and local edges produce stronger differences than broad pale regions.
"""
        ),
        codeCell(
            """
redInput = np.clip(
    inputArray[:, :, 0] - inputArray[:, :, 1:].mean(axis=2),
    0.0,
    1.0,
)
redOutput = np.clip(
    reconstructionArray[:, :, 0] - reconstructionArray[:, :, 1:].mean(axis=2),
    0.0,
    1.0,
)
redDifference = redInput - redOutput
redLimit = max(0.05, float(np.abs(redDifference).max()))

figure, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor="#f6f3ed")
axes[0].imshow(inputArray)
axes[0].set_title("Input at 128 × 128", fontweight="bold")
axes[1].imshow(np.clip(reconstructionArray, 0.0, 1.0))
axes[1].set_title("Autoencoder reconstruction", fontweight="bold")
axes[2].imshow(differenceMap, cmap="magma", vmin=0.0, vmax=max(0.25, float(differenceMap.max())))
axes[2].set_title("Absolute pixel difference", fontweight="bold")
axes[3].imshow(redDifference, cmap="coolwarm", vmin=-redLimit, vmax=redLimit)
axes[3].set_title("Red dominance: input − output", fontweight="bold")

for axis in axes:
    axis.axis("off")

figure.suptitle(
    "What survives the geometric bottleneck?",
    fontsize=18,
    fontweight="bold",
)
figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.92))
plt.show()
"""
        ),
        markdownCell(
            """
### Looking into the bottleneck

The first panel averages the 32 bottleneck channels into one map. The remaining panels show the eight channels with the highest spatial variance. Different channels preserve different local responses, but none should be treated as a named-object or meaning detector.
"""
        ),
        codeCell(
            """
def normalizeMap(featureMap: np.ndarray) -> np.ndarray:
    minimumValue = float(featureMap.min())
    maximumValue = float(featureMap.max())
    if maximumValue - minimumValue < 1e-8:
        return np.zeros_like(featureMap)
    return (featureMap - minimumValue) / (maximumValue - minimumValue)


bottleneckChannels = bottleneckTensor[0].cpu()
channelVariances = bottleneckChannels.flatten(1).var(dim=1)
selectedChannels = torch.argsort(channelVariances, descending=True)[:8].tolist()

figure, axes = plt.subplots(3, 3, figsize=(10, 9.5), facecolor="#f6f3ed")
meanMap = bottleneckChannels.mean(dim=0).numpy()
axes[0, 0].imshow(normalizeMap(meanMap), cmap="inferno", vmin=0.0, vmax=1.0)
axes[0, 0].set_title("Mean of all 32 channels", fontweight="bold")
axes[0, 0].axis("off")

for plotIndex, channelIndex in enumerate(selectedChannels, start=1):
    axis = axes.flat[plotIndex]
    channelMap = bottleneckChannels[channelIndex].numpy()
    axis.imshow(normalizeMap(channelMap), cmap="inferno", vmin=0.0, vmax=1.0)
    axis.set_title(f"Channel {channelIndex + 1}")
    axis.axis("off")

figure.suptitle(
    "The 32 × 16 × 16 visual bottleneck",
    fontsize=17,
    fontweight="bold",
)
figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
plt.show()
"""
        ),
        markdownCell(
            """
### Evidence that the checkpoint was trained

The loss curves stored in the checkpoint fall over 18 epochs. This shows that the model became better at reconstructing held-out synthetic patterns; it does not show that the model understands the artwork.
"""
        ),
        codeCell(
            """
trainingHistory = checkpoint["history"]
epochs = np.arange(1, len(trainingHistory["train"]) + 1)

figure, axis = plt.subplots(figsize=(8.5, 4.8), facecolor="#f6f3ed")
axis.plot(epochs, trainingHistory["train"], label="Training", color="#9f2725", linewidth=2.4)
axis.plot(epochs, trainingHistory["validation"], label="Validation", color="#2f5964", linewidth=2.4)
axis.set_xlabel("Epoch")
axis.set_ylabel("Mean squared error")
axis.set_title("Stored training history", fontsize=16, fontweight="bold")
axis.grid(alpha=0.2)
axis.legend(frameon=False)
plt.show()

print(f"First training loss:  {trainingHistory['train'][0]:.6f}")
print(f"Final training loss:  {trainingHistory['train'][-1]:.6f}")
print(f"Final validation loss: {trainingHistory['validation'][-1]:.6f}")
"""
        ),
        markdownCell(
            """
## Checks

These assertions verify the encoder and decoder dimensions, finite values, and metric calculations.
"""
        ),
        codeCell(
            """
assert tuple(inputTensor.shape) == (1, 3, 128, 128)
assert tuple(bottleneckTensor.shape) == (1, 32, 16, 16)
assert tuple(reconstructionTensor.shape) == tuple(inputTensor.shape)
assert torch.isfinite(bottleneckTensor).all()
assert torch.isfinite(reconstructionTensor).all()
assert 0.0 <= reconstructionTensor.min().item() <= reconstructionTensor.max().item() <= 1.0
assert math.isclose(meanSquaredError, 0.003884, rel_tol=0.02, abs_tol=1e-5)

print("PASS: encoder output is the expected 1 × 32 × 16 × 16 bottleneck.")
print("PASS: decoder reconstruction matches the input shape and contains only finite values.")
print("PASS: reconstruction values remain in the Sigmoid range [0, 1].")
print("PASS: measured MSE agrees with the recorded project result.")
"""
        ),
        markdownCell(
            """
## Takeaways

1. The autoencoder performs an actual encode–decode process: the image becomes a smaller multichannel representation and is then reconstructed.
2. The reconstruction retains repeated geometric structures more successfully than handwriting and irregular local detail.
3. The red tether is not preserved as a meaningful autobiographical symbol. Parts weaken, while a faint learned red tendency spreads elsewhere.
4. The experiment separates **formal survival** from **human interpretation**: the network compresses recurring visual patterns, not the personal significance attached to them.
"""
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=notebookMetadata())


def prepareAssets() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(ROOT_DIR / "My_Own_Work.png") as sourceImage:
        artworkImage = ImageOps.exif_transpose(sourceImage).convert("RGB")
        resizedImage = artworkImage.resize((128, 128), Image.Resampling.LANCZOS)
        resizedImage.save(ASSET_DIR / "my-own-work-128.png", optimize=True)

    shutil.copy2(
        ROOT_DIR / "model" / "geometric-autoencoder.pt",
        ASSET_DIR / "geometric-autoencoder.pt",
    )

    figureNames = [
        "01-resolution-ladder.png",
        "02-geometric-training-patterns.png",
        "03-autoencoder-reconstruction.png",
        "04-feature-maps.png",
        "05-las-meninas-comparison.png",
        "06-torchview-network.png",
        "06-torchview-network.svg",
        "07-paper-style-architecture.png",
        "07-paper-style-architecture.svg",
        "08-actual-forward-pass.png",
        "training-loss.png",
    ]
    for figureName in figureNames:
        shutil.copy2(ROOT_DIR / "outputs" / figureName, FIGURE_DIR / figureName)


def executeNotebook(notebookPath: Path) -> None:
    notebook = nbformat.read(notebookPath, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(SUBMISSION_DIR)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, notebookPath)


def validateNotebook(notebookPath: Path) -> None:
    notebook = nbformat.read(notebookPath, as_version=4)
    codeCells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not codeCells:
        raise RuntimeError(f"Notebook has no code cells: {notebookPath}")

    missingExecutionCounts = [
        cellIndex
        for cellIndex, cell in enumerate(codeCells, start=1)
        if cell.execution_count is None
    ]
    if missingExecutionCounts:
        raise RuntimeError(
            f"Unexecuted code cells in {notebookPath.name}: {missingExecutionCounts}"
        )

    errorOutputs = [
        output
        for cell in codeCells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errorOutputs:
        raise RuntimeError(f"Notebook contains error outputs: {notebookPath}")

    imageOutputCount = sum(
        1
        for cell in codeCells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    if imageOutputCount < 2:
        raise RuntimeError(
            f"Expected at least two embedded image outputs in {notebookPath.name}, "
            f"found {imageOutputCount}."
        )

    print(
        f"Validated {notebookPath.name}: "
        f"{len(codeCells)} executed code cells, {imageOutputCount} embedded figures, 0 errors."
    )


def main() -> None:
    prepareAssets()
    notebookDefinitions = {
        "01-all-layer-outputs.ipynb": buildLayerNotebook(),
        "02-autoencoder-reconstruction.ipynb": buildReconstructionNotebook(),
    }

    for notebookName, notebook in notebookDefinitions.items():
        notebookPath = SUBMISSION_DIR / notebookName
        nbformat.write(notebook, notebookPath)
        executeNotebook(notebookPath)
        validateNotebook(notebookPath)

    print(f"Submission notebooks are ready in: {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()
