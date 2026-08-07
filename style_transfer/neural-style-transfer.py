#!/usr/bin/env python3
"""Generate a reproducible *Las Meninas* neural-style-transfer series with Gatys/VGG19.

This module deliberately defines "style" as Gram correlations among VGG feature
channels rather than as a complete art-historical style. Both the content and style
images are resized with their original aspect ratios intact and are never center-cropped.
Each style strength is optimized independently from the same deterministic initial image,
which makes strength a controlled experimental variable.

The default content image is the public-domain *Las Meninas* included in this folder.
The default style image is the project’s original
``inputs/cognitive-map-style-reference.png``. Each run saves a PNG and per-step loss
records in CSV and JSON, plus a ``manifest.json`` containing input hashes, the model,
device, random seed, and all explicit parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    import torch.nn.functional as torch_functional
    from torch import Tensor, nn
except ImportError as exc:
    raise RuntimeError(
        "neural-style-transfer.py requires PyTorch. Run it in an environment with "
        "torch and torchvision installed."
    ) from exc


FORMAT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONTENT_IMAGE = PROJECT_ROOT / "inputs" / "las-meninas-content.jpg"
DEFAULT_STYLE_IMAGE = PROJECT_ROOT / "inputs" / "cognitive-map-style-reference.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_STYLE_LAYERS = (
    "relu1_1",
    "relu2_1",
    "relu3_1",
    "relu4_1",
    "relu5_1",
)
DEFAULT_CONTENT_LAYER = "relu4_2"
DEFAULT_STYLE_STRENGTHS = (0.25, 0.5, 1.0)


def project_relative_path(path: str | Path) -> str:
    """Prefer project-relative paths so manifests do not expose local user directories."""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)

# These indices match torchvision.models.vgg19().features. Post-ReLU activations avoid
# cancellation between positive and negative responses and follow common Gatys conventions.
VGG19_LAYER_INDICES: dict[str, int] = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu4_3": 24,
    "relu4_4": 26,
    "relu5_1": 29,
    "relu5_2": 31,
    "relu5_3": 33,
    "relu5_4": 35,
}

DeviceChoice = Literal["auto", "cpu", "mps"]
WeightChoice = Literal["default", "none"]


@dataclass(frozen=True)
class StyleTransferConfig:
    """Reproducible configuration for one complete style-transfer run."""

    content_image: Path
    style_image: Path
    output_dir: Path
    style_strengths: tuple[float, ...] = DEFAULT_STYLE_STRENGTHS
    long_side: int = 512
    steps: int = 500
    learning_rate: float = 0.02
    content_weight: float = 1.0
    style_weight: float = 1_000_000.0
    tv_weight: float = 0.0001
    initial_noise: float = 0.02
    seed: int = 139
    device: DeviceChoice = "auto"
    weights: WeightChoice = "default"
    content_layer: str = DEFAULT_CONTENT_LAYER
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS
    max_source_pixels: int = 1_000_000_000
    progress_every: int = 25
    quiet: bool = False
    overwrite: bool = False

    def validate(self) -> None:
        if not self.content_image.is_file():
            raise FileNotFoundError(f"Content image does not exist: {self.content_image}")
        if not self.style_image.is_file():
            raise FileNotFoundError(f"Style-reference image does not exist: {self.style_image}")
        if self.content_image.resolve() == self.style_image.resolve():
            raise ValueError("The content and style-reference images must be different files.")
        if not self.style_strengths:
            raise ValueError("style_strengths must not be empty.")
        for strength in self.style_strengths:
            if not math.isfinite(strength) or strength < 0.0:
                raise ValueError("Every style strength must be finite and nonnegative.")
        if len(set(self.style_strengths)) != len(self.style_strengths):
            raise ValueError("style_strengths must not contain duplicate values.")
        if self.long_side < 32:
            raise ValueError("long_side must be at least 32.")
        if self.steps < 1:
            raise ValueError("steps must be at least 1.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        for field_name, value in (
            ("content_weight", self.content_weight),
            ("style_weight", self.style_weight),
            ("tv_weight", self.tv_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")
        if self.content_weight == self.style_weight == self.tv_weight == 0.0:
            raise ValueError("Content, style, and TV weights cannot all be zero.")
        if not math.isfinite(self.initial_noise) or not 0.0 <= self.initial_noise <= 1.0:
            raise ValueError("initial_noise must be between 0 and 1.")
        if self.seed < 0 or self.seed >= 2**63:
            raise ValueError("seed must be between 0 and 2^63 - 1.")
        if self.max_source_pixels < 1:
            raise ValueError("max_source_pixels must be a positive integer.")
        if self.progress_every < 1:
            raise ValueError("progress_every must be a positive integer.")
        if self.content_layer not in VGG19_LAYER_INDICES:
            raise ValueError(f"Unknown content layer: {self.content_layer}")
        if not self.style_layers:
            raise ValueError("style_layers must not be empty.")
        unknown_layers = sorted(set(self.style_layers) - set(VGG19_LAYER_INDICES))
        if unknown_layers:
            raise ValueError(f"Unknown style layers: {', '.join(unknown_layers)}")


@dataclass(frozen=True)
class LoadedImage:
    """An aspect-ratio-preserving image plus auditable size records."""

    image: Image.Image
    original_width: int
    original_height: int
    decoded_width: int
    decoded_height: int
    working_width: int
    working_height: int

    def manifest_record(self) -> dict[str, object]:
        return {
            "original_size": [self.original_width, self.original_height],
            "decoded_size": [self.decoded_width, self.decoded_height],
            "working_size": [self.working_width, self.working_height],
            "preprocessing": "jpeg-draft-if-available_then-aspect-preserving-lanczos-resize",
            "center_crop": False,
        }


@dataclass(frozen=True)
class LossRecord:
    """Raw and weighted losses for one optimization state."""

    step: int
    style_strength: float
    content_loss: float
    style_loss: float
    tv_loss: float
    weighted_content_loss: float
    weighted_style_loss: float
    weighted_tv_loss: float
    total_loss: float


@dataclass(frozen=True)
class StrengthResult:
    """Final files and loss records for one style strength."""

    style_strength: float
    output_image: Path
    loss_csv: Path
    loss_json: Path
    output_sha256: str
    records: tuple[LossRecord, ...]


class VGG19FeatureExtractor(nn.Module):
    """VGG19 feature extractor that stops after the last requested layer."""

    def __init__(
        self,
        features: nn.Sequential,
        *,
        content_layer: str,
        style_layers: Sequence[str],
    ) -> None:
        super().__init__()
        if content_layer not in VGG19_LAYER_INDICES:
            raise ValueError(f"Unknown content layer: {content_layer}")
        unknown_layers = sorted(set(style_layers) - set(VGG19_LAYER_INDICES))
        if unknown_layers:
            raise ValueError(f"Unknown style layers: {', '.join(unknown_layers)}")
        requested_layers = set(style_layers)
        requested_layers.add(content_layer)
        self.features = features
        self.content_layer = content_layer
        self.style_layers = tuple(style_layers)
        self.index_to_name = {
            layer_index: layer_name
            for layer_name, layer_index in VGG19_LAYER_INDICES.items()
            if layer_name in requested_layers
        }
        self.last_index = max(self.index_to_name)

    def forward(self, input_tensor: Tensor) -> dict[str, Tensor]:
        if input_tensor.ndim != 4:
            raise ValueError("VGG input must have shape [N, C, H, W].")
        current = input_tensor
        selected: dict[str, Tensor] = {}
        for layer_index, layer in enumerate(self.features):
            current = layer(current)
            layer_name = self.index_to_name.get(layer_index)
            if layer_name is not None:
                selected[layer_name] = current
            if layer_index >= self.last_index:
                break
        expected = set(self.style_layers)
        expected.add(self.content_layer)
        missing = sorted(expected - set(selected))
        if missing:
            raise RuntimeError(
                "The feature network did not produce the requested layers. Confirm that "
                "the network matches the VGG19 indices: " + ", ".join(missing)
            )
        return selected

    @torch.no_grad()
    def content_target(self, normalized_content: Tensor) -> Tensor:
        return self(normalized_content)[self.content_layer].detach()

    @torch.no_grad()
    def style_targets(self, normalized_style: Tensor) -> dict[str, Tensor]:
        """Compute Gram targets layer by layer to release shallow activations promptly."""

        current = normalized_style
        targets: dict[str, Tensor] = {}
        style_layer_set = set(self.style_layers)
        for layer_index, layer in enumerate(self.features):
            current = layer(current)
            layer_name = self.index_to_name.get(layer_index)
            if layer_name in style_layer_set:
                targets[layer_name] = gram_matrix(current).detach()
            if layer_index >= self.last_index:
                break
        missing = sorted(style_layer_set - set(targets))
        if missing:
            raise RuntimeError("Unable to produce style targets: " + ", ".join(missing))
        return targets


def parse_style_strengths(value: str | Sequence[float]) -> tuple[float, ...]:
    """Parse unique style strengths and return them in ascending order."""

    raw_values: Sequence[object]
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
        if any(part == "" for part in raw_values):
            raise ValueError("Style strengths must be comma-separated numbers.")
    else:
        raw_values = value
    try:
        strengths = tuple(float(raw_value) for raw_value in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Style strengths must be comma-separated numbers.") from exc
    if not strengths:
        raise ValueError("At least one style strength is required.")
    if any(not math.isfinite(strength) or strength < 0.0 for strength in strengths):
        raise ValueError("Style strengths must be finite and nonnegative.")
    if len(set(strengths)) != len(strengths):
        raise ValueError("Style strengths must not contain duplicate values.")
    return tuple(sorted(strengths))


def choose_device(requested: DeviceChoice) -> torch.device:
    """Prefer Apple MPS automatically and otherwise fall back to the CPU."""

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return torch.device("mps" if mps_available else "cpu")
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch environment.")
        return torch.device("mps")
    return torch.device("cpu")


def configure_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, and Torch and request deterministic algorithms."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    # A few MPS operators may not have fully deterministic implementations. warn_only
    # avoids aborting the run; the manifest records the device for same-device replication.
    torch.use_deterministic_algorithms(True, warn_only=True)


def gram_matrix(features: Tensor) -> Tensor:
    """Compute a batched Gram matrix normalized by C × H × W."""

    if features.ndim != 4:
        raise ValueError("features must have shape [N, C, H, W].")
    batch_size, channels, height, width = features.shape
    flattened = features.reshape(batch_size, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / float(channels * height * width)


def total_variation_loss(image_tensor: Tensor) -> Tensor:
    """Compute normalized L1 total variation to discourage high-frequency noise."""

    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have shape [N, C, H, W].")
    if image_tensor.shape[-2] < 2 or image_tensor.shape[-1] < 2:
        raise ValueError("Total variation requires an image of at least 2 × 2 pixels.")
    vertical = torch.mean(torch.abs(image_tensor[:, :, 1:, :] - image_tensor[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :-1]))
    return vertical + horizontal


def normalize_for_vgg(image_tensor: Tensor) -> Tensor:
    """Normalize an RGB tensor for Torchvision's ImageNet VGG19 weights."""

    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError("image_tensor must have shape [N, 3, H, W].")
    mean = image_tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image_tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image_tensor - mean) / std


def image_to_tensor(image: Image.Image, device: torch.device) -> Tensor:
    """Convert an RGB PIL image to a [1, 3, H, W] float32 tensor in [0, 1]."""

    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def tensor_to_image(image_tensor: Tensor) -> Image.Image:
    """Convert a [0, 1] image tensor to a PIL image suitable for lossless RGB PNG."""

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
        raise ValueError("image_tensor must have shape [1, 3, H, W].")
    array = image_tensor.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    uint8_array = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_array, mode="RGB")


def load_resized_rgb(
    image_path: str | Path,
    *,
    long_side: int,
    max_source_pixels: int,
) -> LoadedImage:
    """Load an image within explicit limits while preserving its aspect ratio.

    No center crop or letterboxing is applied. The original JPEG contains roughly
    780 million pixels, so decoding it at full resolution would consume several
    gigabytes of memory. Pillow's JPEG ``draft`` mode requests a lower-resolution
    decode before an exact Lanczos resize to the working dimensions. Non-JPEG inputs
    are still checked against the source-pixel limit before further processing.
    """

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if long_side < 32:
        raise ValueError("long_side must be at least 32.")
    if max_source_pixels < 1:
        raise ValueError("max_source_pixels must be a positive integer.")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max(max_source_pixels // 2 + 1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                original_width, original_height = source.size
                source_pixels = original_width * original_height
                if source_pixels > max_source_pixels:
                    raise ValueError(
                        f"The source image contains {source_pixels:,} pixels, exceeding "
                        f"the limit of {max_source_pixels:,}."
                    )
                scale = min(1.0, long_side / max(original_width, original_height))
                draft_size = (
                    max(1, round(original_width * scale)),
                    max(1, round(original_height * scale)),
                )
                source.draft("RGB", draft_size)
                oriented = ImageOps.exif_transpose(source)
                decoded_width, decoded_height = oriented.size
                # ``thumbnail`` does not enlarge small images, which would make the same
                # --long-side value produce inconsistent working scales. Compute the target
                # dimensions explicitly; JPEG draft only reduces the cost of decoding very
                # large images and does not change the final aspect ratio.
                resize_scale = long_side / max(decoded_width, decoded_height)
                target_size = (
                    max(1, round(decoded_width * resize_scale)),
                    max(1, round(decoded_height * resize_scale)),
                )
                working = oriented.convert("RGB").resize(
                    target_size,
                    Image.Resampling.LANCZOS,
                    reducing_gap=3.0,
                )
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    working_width, working_height = working.size
    # VGG19 has five 2 × 2 pooling stages. A short side below 32 would collapse later layers.
    if min(working_width, working_height) < 32:
        raise ValueError(
            "The short side is below 32 pixels after aspect-ratio-preserving resizing. "
            "Increase --long-side or use an image with a less extreme aspect ratio."
        )
    return LoadedImage(
        image=working,
        original_width=original_width,
        original_height=original_height,
        decoded_width=decoded_width,
        decoded_height=decoded_height,
        working_width=working_width,
        working_height=working_height,
    )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading a high-resolution source all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while True:
            block = file_handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def style_strength_slug(strength: float) -> str:
    """Convert a strength to a stable, filename-safe decimal identifier."""

    text = f"{strength:.8f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def create_vgg19_extractor(
    config: StyleTransferConfig,
    device: torch.device,
) -> tuple[VGG19FeatureExtractor, str]:
    """Create a frozen VGG19 feature network through Torchvision's official API."""

    try:
        import torchvision
        from torchvision.models import VGG19_Weights, vgg19
    except ImportError as exc:
        raise RuntimeError("torchvision is not installed, so VGG19 cannot be created.") from exc

    weights = VGG19_Weights.IMAGENET1K_V1 if config.weights == "default" else None
    model = vgg19(weights=weights, progress=not config.quiet)
    features = model.features.eval()
    for parameter in features.parameters():
        parameter.requires_grad_(False)
    extractor = VGG19FeatureExtractor(
        features,
        content_layer=config.content_layer,
        style_layers=config.style_layers,
    ).to(device)
    signature = (
        f"torchvision={torchvision.__version__};architecture=vgg19;"
        f"weights={'VGG19_Weights.IMAGENET1K_V1' if weights is not None else 'none-random'}"
    )
    del model
    return extractor, signature


def make_initial_image(content_tensor: Tensor, *, noise_amount: float, seed: int) -> Tensor:
    """Generate seeded CPU noise so every strength shares the same initial state."""

    if not 0.0 <= noise_amount <= 1.0:
        raise ValueError("noise_amount must be between 0 and 1.")
    if noise_amount == 0.0:
        return content_tensor.detach().clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cpu_noise = torch.randn(
        content_tensor.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    noise = cpu_noise.to(content_tensor.device)
    return (content_tensor.detach() + noise_amount * noise).clamp(0.0, 1.0)


def compute_perceptual_losses(
    generated_tensor: Tensor,
    *,
    extractor: VGG19FeatureExtractor,
    content_target: Tensor,
    style_targets: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the unweighted content, style, and total-variation losses."""

    features = extractor(normalize_for_vgg(generated_tensor))
    content_loss = torch_functional.mse_loss(
        features[extractor.content_layer],
        content_target,
    )
    per_layer_style_losses = [
        torch_functional.mse_loss(gram_matrix(features[layer_name]), style_targets[layer_name])
        for layer_name in extractor.style_layers
    ]
    style_loss = torch.stack(per_layer_style_losses).mean()
    tv_loss = total_variation_loss(generated_tensor)
    return content_loss, style_loss, tv_loss


def optimize_style_strength(
    *,
    initial_tensor: Tensor,
    extractor: VGG19FeatureExtractor,
    content_target: Tensor,
    style_targets: dict[str, Tensor],
    config: StyleTransferConfig,
    style_strength: float,
) -> tuple[Tensor, tuple[LossRecord, ...]]:
    """Optimize one style strength with Adam and record every state from start to finish."""

    if not math.isfinite(style_strength) or style_strength < 0.0:
        raise ValueError("style_strength must be finite and nonnegative.")
    generated = nn.Parameter(initial_tensor.detach().clone())
    # foreach=False fixes the single-tensor update path and reduces small differences caused
    # when different device backends automatically select foreach implementations.
    optimizer = torch.optim.Adam(
        [generated],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    records: list[LossRecord] = []

    # step=0 is the shared initial state. Each updated state is recorded on the next loop,
    # so step=steps corresponds exactly to the saved image, not the pre-update state.
    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        content_loss, style_loss, tv_loss = compute_perceptual_losses(
            generated,
            extractor=extractor,
            content_target=content_target,
            style_targets=style_targets,
        )
        weighted_content = config.content_weight * content_loss
        weighted_style = config.style_weight * style_strength * style_loss
        weighted_tv = config.tv_weight * tv_loss
        total_loss = weighted_content + weighted_style + weighted_tv

        record = LossRecord(
            step=step,
            style_strength=style_strength,
            content_loss=float(content_loss.detach().cpu()),
            style_loss=float(style_loss.detach().cpu()),
            tv_loss=float(tv_loss.detach().cpu()),
            weighted_content_loss=float(weighted_content.detach().cpu()),
            weighted_style_loss=float(weighted_style.detach().cpu()),
            weighted_tv_loss=float(weighted_tv.detach().cpu()),
            total_loss=float(total_loss.detach().cpu()),
        )
        records.append(record)

        if not config.quiet and (
            step == 0 or step == config.steps or step % config.progress_every == 0
        ):
            print(
                f"strength={style_strength:g} step={step}/{config.steps} "
                f"total={record.total_loss:.6g} content={record.content_loss:.6g} "
                f"style={record.style_loss:.6g} tv={record.tv_loss:.6g}",
                flush=True,
            )

        if step == config.steps:
            break
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"A non-finite loss occurred for strength={style_strength:g} at step={step}. "
                "Lower the learning rate or inspect the input images."
            )
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            generated.clamp_(0.0, 1.0)

    return generated.detach(), tuple(records)


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    """Create the output directory without overwriting earlier experiments by default."""

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"The output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"The output directory is not empty: {output_dir}. Choose another directory "
            "or explicitly pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    """Write UTF-8 JSON with stable key ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
        file_handle.write("\n")


def write_loss_logs(
    *,
    output_dir: Path,
    style_strength: float,
    records: Sequence[LossRecord],
) -> tuple[Path, Path]:
    """Save per-step losses as CSV and as JSON with an explicit semantic description."""

    if not records:
        raise ValueError("records must not be empty.")
    slug = style_strength_slug(style_strength)
    csv_path = output_dir / f"loss-strength-{slug}.csv"
    json_path = output_dir / f"loss-strength-{slug}.json"
    rows = [asdict(record) for record in records]
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        json_path,
        {
            "format_version": FORMAT_VERSION,
            "style_strength": style_strength,
            "step_semantics": (
                "step 0 is the shared initial state; step N is the state after N Adam updates"
            ),
            "records": rows,
        },
    )
    return csv_path, json_path


def save_strength_result(
    *,
    output_dir: Path,
    style_strength: float,
    image_tensor: Tensor,
    records: Sequence[LossRecord],
) -> StrengthResult:
    """Save the lossless image and loss logs for one strength."""

    slug = style_strength_slug(style_strength)
    image_path = output_dir / f"neural-style-strength-{slug}.png"
    tensor_to_image(image_tensor).save(image_path, format="PNG", optimize=True)
    csv_path, json_path = write_loss_logs(
        output_dir=output_dir,
        style_strength=style_strength,
        records=records,
    )
    return StrengthResult(
        style_strength=style_strength,
        output_image=image_path,
        loss_csv=csv_path,
        loss_json=json_path,
        output_sha256=sha256_file(image_path),
        records=tuple(records),
    )


def build_manifest(
    *,
    config: StyleTransferConfig,
    device: torch.device,
    model_signature: str,
    content_image: LoadedImage,
    style_image: LoadedImage,
    results: Sequence[StrengthResult],
) -> dict[str, object]:
    """Build a manifest without relying on unrecorded implicit defaults."""

    return {
        "format_version": FORMAT_VERSION,
        "project": "Entering Las Meninas: How a Painting Changed the Way I Look at Art",
        "method": "Gatys neural style transfer with frozen VGG19 features",
        "implementation": {
            "script": project_relative_path(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "torch": torch.__version__,
            "model_signature": model_signature,
        },
        "reproducibility": {
            "seed": config.seed,
            "device": str(device),
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": True,
            "shared_initial_state_across_strengths": True,
            "optimizer": "Adam",
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "adam_foreach": False,
        },
        "inputs": {
            "content": {
                "path": project_relative_path(config.content_image),
                "sha256": sha256_file(config.content_image),
                **content_image.manifest_record(),
            },
            "style": {
                "path": project_relative_path(config.style_image),
                "sha256": sha256_file(config.style_image),
                **style_image.manifest_record(),
            },
        },
        "parameters": {
            "style_strengths": list(config.style_strengths),
            "long_side": config.long_side,
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "content_weight": config.content_weight,
            "style_weight": config.style_weight,
            "tv_weight": config.tv_weight,
            "initial_noise": config.initial_noise,
            "weights": config.weights,
            "content_layer": config.content_layer,
            "style_layers": list(config.style_layers),
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "max_source_pixels": config.max_source_pixels,
            "center_crop": False,
            "aspect_ratio_preserved": True,
        },
        "outputs": [
            {
                "style_strength": result.style_strength,
                "image": project_relative_path(result.output_image),
                "image_sha256": result.output_sha256,
                "loss_csv": project_relative_path(result.loss_csv),
                "loss_json": project_relative_path(result.loss_json),
                "loss_records": len(result.records),
                "final_loss": asdict(result.records[-1]),
            }
            for result in results
        ],
    }


def run_style_transfer(
    config: StyleTransferConfig,
    *,
    extractor_override: VGG19FeatureExtractor | None = None,
    model_signature_override: str | None = None,
) -> dict[str, object]:
    """Run every strength and return the same dictionary written to ``manifest.json``."""

    config.validate()
    prepare_output_directory(config.output_dir, overwrite=config.overwrite)
    configure_reproducibility(config.seed)
    device = choose_device(config.device)

    content_loaded = load_resized_rgb(
        config.content_image,
        long_side=config.long_side,
        max_source_pixels=config.max_source_pixels,
    )
    style_loaded = load_resized_rgb(
        config.style_image,
        long_side=config.long_side,
        max_source_pixels=config.max_source_pixels,
    )
    content_tensor = image_to_tensor(content_loaded.image, device)
    style_tensor = image_to_tensor(style_loaded.image, device)

    if extractor_override is None:
        extractor, model_signature = create_vgg19_extractor(config, device)
    else:
        extractor = extractor_override.to(device).eval()
        model_signature = model_signature_override or "externally-supplied-feature-extractor"
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    content_target = extractor.content_target(normalize_for_vgg(content_tensor))
    style_targets = extractor.style_targets(normalize_for_vgg(style_tensor))
    initial_tensor = make_initial_image(
        content_tensor,
        noise_amount=config.initial_noise,
        seed=config.seed,
    )

    results: list[StrengthResult] = []
    for style_strength in config.style_strengths:
        final_tensor, records = optimize_style_strength(
            initial_tensor=initial_tensor,
            extractor=extractor,
            content_target=content_target,
            style_targets=style_targets,
            config=config,
            style_strength=style_strength,
        )
        results.append(
            save_strength_result(
                output_dir=config.output_dir,
                style_strength=style_strength,
                image_tensor=final_tensor,
                records=records,
            )
        )

    manifest = build_manifest(
        config=config,
        device=device,
        model_signature=model_signature,
        content_image=content_loaded,
        style_image=style_loaded,
        results=results,
    )
    write_json(config.output_dir / "manifest.json", manifest)
    if not config.quiet:
        print(f"Completed {len(results)} style strengths: {config.output_dir}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Apply Gatys neural style transfer to Las Meninas using frozen VGG19 features, "
            "Gram-matrix style loss, content loss, and total-variation loss. Inputs retain "
            "their aspect ratios and are not center-cropped."
        )
    )
    parser.add_argument(
        "content_image",
        nargs="?",
        type=Path,
        default=DEFAULT_CONTENT_IMAGE,
        help=f"Content-image path (default: {DEFAULT_CONTENT_IMAGE})",
    )
    parser.add_argument(
        "--style-image",
        type=Path,
        default=DEFAULT_STYLE_IMAGE,
        help=f"Style-reference image (default: original project image at {DEFAULT_STYLE_IMAGE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--style-strengths",
        default=",".join(f"{value:g}" for value in DEFAULT_STYLE_STRENGTHS),
        help="Comma-separated nonnegative strength multipliers, for example 0.25,0.5,1.0.",
    )
    parser.add_argument(
        "--long-side", type=int, default=512, help="Longest side of each working image; default 512."
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of Adam updates for each strength."
    )
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Adam learning rate.")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Content-loss weight.")
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1_000_000.0,
        help="Base style-loss weight, subsequently multiplied by the style strength.",
    )
    parser.add_argument("--tv-weight", type=float, default=0.0001, help="TV-loss weight.")
    parser.add_argument(
        "--initial-noise",
        type=float,
        default=0.02,
        help="Amount of deterministic Gaussian noise added to the content initialization, 0–1.",
    )
    parser.add_argument("--seed", type=int, default=139, help="Python/NumPy/Torch random seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="auto prefers Apple MPS and falls back to the CPU when MPS is unavailable.",
    )
    parser.add_argument(
        "--weights",
        choices=("default", "none"),
        default="default",
        help=(
            "default uses ImageNet-pretrained VGG19; none is for offline diagnostics only "
            "and should not be used for formal analysis."
        ),
    )
    parser.add_argument(
        "--content-layer", default=DEFAULT_CONTENT_LAYER, help="VGG content layer."
    )
    parser.add_argument(
        "--style-layers",
        default=",".join(DEFAULT_STYLE_LAYERS),
        help="Comma-separated VGG style layers.",
    )
    parser.add_argument(
        "--max-source-pixels",
        type=int,
        default=1_000_000_000,
        help="Maximum source-image pixel count that may be read.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print losses every N steps. CSV and JSON always record every step.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress optimization progress.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow same-name artifacts to be overwritten; unrelated files are not deleted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        style_strengths = parse_style_strengths(arguments.style_strengths)
    except ValueError as exc:
        parser.error(str(exc))
    style_layers = tuple(
        layer_name.strip() for layer_name in arguments.style_layers.split(",") if layer_name.strip()
    )
    config = StyleTransferConfig(
        content_image=arguments.content_image,
        style_image=arguments.style_image,
        output_dir=arguments.output_dir,
        style_strengths=style_strengths,
        long_side=arguments.long_side,
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        content_weight=arguments.content_weight,
        style_weight=arguments.style_weight,
        tv_weight=arguments.tv_weight,
        initial_noise=arguments.initial_noise,
        seed=arguments.seed,
        device=arguments.device,
        weights=arguments.weights,
        content_layer=arguments.content_layer,
        style_layers=style_layers,
        max_source_pixels=arguments.max_source_pixels,
        progress_every=arguments.progress_every,
        quiet=arguments.quiet,
        overwrite=arguments.overwrite,
    )
    run_style_transfer(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
