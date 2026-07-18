#!/usr/bin/env python3
"""使用预训练 VGG19 分离绘画的风格表征与空间内容表征。

该模块把 CNN 当作测量工具，而不是分类器。它导出三类结果：

1. 多层通道能量图，用于观察不同深度的激活位置；
2. Gram matrix，用作明确受限的 CNN 风格描述符；
3. 保留二维位置的池化特征，用于空间内容距离。

默认权重来自 TorchVision 官方 ``VGG19_Weights.IMAGENET1K_V1``。输入图像会
等比例缩放并居中填充为正方形；超大 JPEG 会先调用解码器原生降采样，避免把
《宫娥》高清原图完整展开为数 GB 的 RGB 数组。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    import torch.nn.functional as torch_functional
    from torch import Tensor, nn
except ImportError as exc:
    raise RuntimeError(
        "cnn-analysis.py 需要 PyTorch。请在已安装 torch 与 torchvision 的环境中运行。"
    ) from exc


FORMAT_VERSION = 1
MODEL_NAME = "torchvision-vgg19"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 索引对应 torchvision.models.vgg19().features。使用 ReLU 后的激活与经典
# neural-style 表征一致，同时避免卷积输出中的正负抵消。
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

DEFAULT_STYLE_LAYERS = (
    "relu1_1",
    "relu2_1",
    "relu3_1",
    "relu4_1",
    "relu5_1",
)
DEFAULT_SPATIAL_LAYERS = ("relu3_1", "relu4_2", "relu5_1")
DEFAULT_ACTIVATION_LAYERS = (
    "relu1_1",
    "relu2_1",
    "relu3_1",
    "relu4_1",
    "relu4_2",
    "relu5_1",
)

DeviceChoice = Literal["auto", "cpu", "mps", "cuda"]
WeightChoice = Literal["default", "none"]


def project_relative_path(path: str | Path) -> str:
    """优先输出项目相对路径，保持公开结果可移植且不暴露本机目录。"""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class AnalysisConfig:
    """影响特征结果与缓存键的全部配置。"""

    image_size: int = 512
    spatial_size: int = 32
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS
    spatial_layers: tuple[str, ...] = DEFAULT_SPATIAL_LAYERS
    activation_layers: tuple[str, ...] = DEFAULT_ACTIVATION_LAYERS
    weights: WeightChoice = "default"
    max_non_jpeg_pixels: int = 100_000_000

    def validate(self) -> None:
        if self.image_size < 32:
            raise ValueError("image_size 必须至少为 32。")
        if self.spatial_size < 1 or self.spatial_size > self.image_size:
            raise ValueError("spatial_size 必须位于 1 与 image_size 之间。")
        if self.max_non_jpeg_pixels < 1:
            raise ValueError("max_non_jpeg_pixels 必须为正整数。")
        for group_name, layers in (
            ("style_layers", self.style_layers),
            ("spatial_layers", self.spatial_layers),
            ("activation_layers", self.activation_layers),
        ):
            if not layers:
                raise ValueError(f"{group_name} 不能为空。")
            unknown = sorted(set(layers) - set(VGG19_LAYER_INDICES))
            if unknown:
                raise ValueError(f"{group_name} 包含未知层：{', '.join(unknown)}")

    def cache_payload(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "model_name": MODEL_NAME,
            "weights": self.weights,
            "image_size": self.image_size,
            "spatial_size": self.spatial_size,
            "style_layers": list(self.style_layers),
            "spatial_layers": list(self.spatial_layers),
            "activation_layers": list(self.activation_layers),
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
            "preprocessing": "bounded-jpeg-draft_then_aspect-letterbox",
        }


@dataclass(frozen=True)
class LoadedImage:
    """经过有界解码和正方形 letterbox 后的图像及尺寸记录。"""

    square_rgb: Image.Image
    original_width: int
    original_height: int
    decoded_width: int
    decoded_height: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int


@dataclass
class TensorDescriptors:
    """一次 CNN 前向传播直接得到的设备张量描述符。"""

    grams: dict[str, Tensor]
    spatial: dict[str, Tensor]
    activations: dict[str, Tensor]


@dataclass
class FeatureBundle:
    """可移植、可缓存的 NumPy 特征及其元数据。"""

    metadata: dict[str, object]
    grams: dict[str, np.ndarray]
    spatial: dict[str, np.ndarray]
    activations: dict[str, np.ndarray]


@dataclass(frozen=True)
class ExtractionResult:
    bundle: FeatureBundle
    cache_path: Path
    cache_hit: bool
    image_id: str


class VGGDescriptorExtractor:
    """在 VGG19 前向传播过程中即时压缩所需激活。

    不保留所有原始 feature maps 是刻意的工程选择：512 像素输入的第一层激活
    单层即可超过 64 MB。这里在对应层立刻计算 Gram、固定网格空间特征和通道
    RMS 能量图，既满足逐层分析，又控制 GPU 与磁盘占用。
    """

    def __init__(
        self,
        features: nn.Sequential,
        config: AnalysisConfig,
        device: torch.device,
    ) -> None:
        config.validate()
        self.features = features.eval().to(device)
        self.config = config
        self.device = device
        requested_layers = set(config.style_layers)
        requested_layers.update(config.spatial_layers)
        requested_layers.update(config.activation_layers)
        self.requested_indices = {
            VGG19_LAYER_INDICES[layer_name]: layer_name
            for layer_name in requested_layers
        }
        self.last_index = max(self.requested_indices)

    @torch.inference_mode()
    def extract(self, input_tensor: Tensor) -> TensorDescriptors:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("input_tensor 必须具有 [1, C, H, W] 形状。")
        current = input_tensor.to(self.device, dtype=torch.float32)
        grams: dict[str, Tensor] = {}
        spatial: dict[str, Tensor] = {}
        activations: dict[str, Tensor] = {}

        for layer_index, layer in enumerate(self.features):
            current = layer(current)
            layer_name = self.requested_indices.get(layer_index)
            if layer_name is not None:
                if layer_name in self.config.style_layers:
                    grams[layer_name] = gram_matrix(current).cpu()
                if layer_name in self.config.spatial_layers:
                    pooled = torch_functional.adaptive_avg_pool2d(
                        current,
                        output_size=(
                            self.config.spatial_size,
                            self.config.spatial_size,
                        ),
                    )
                    spatial[layer_name] = pooled.squeeze(0).cpu()
                if layer_name in self.config.activation_layers:
                    channel_energy = torch.sqrt(
                        torch.mean(torch.square(current), dim=1) + 1e-12
                    )
                    activations[layer_name] = channel_energy.squeeze(0).cpu()
            if layer_index >= self.last_index:
                break

        return TensorDescriptors(
            grams=grams,
            spatial=spatial,
            activations=activations,
        )


def gram_matrix(features: Tensor) -> Tensor:
    """计算按 C×H×W 归一化的批量 Gram matrix。"""

    if features.ndim != 4:
        raise ValueError("features 必须具有 [N, C, H, W] 形状。")
    batch_size, channels, height, width = features.shape
    flattened = features.reshape(batch_size, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    normalizer = float(channels * height * width)
    return gram / normalizer


def choose_device(requested: DeviceChoice) -> torch.device:
    """按 MPS、CUDA、CPU 顺序选择可用设备，也允许显式指定。"""

    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("请求了 MPS，但当前 PyTorch 环境不可用。")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前 PyTorch 环境不可用。")
        return torch.device("cuda")
    return torch.device("cpu")


def create_vgg19_extractor(
    config: AnalysisConfig,
    device: torch.device,
) -> tuple[VGGDescriptorExtractor, str]:
    """从 TorchVision 官方 API 创建冻结的 VGG19 特征主干。"""

    try:
        import torchvision
        from torchvision.models import VGG19_Weights, vgg19
    except ImportError as exc:
        raise RuntimeError(
            "未安装 torchvision。请使用包含 torch 与 torchvision 的同一 Python 环境。"
        ) from exc

    weights = VGG19_Weights.IMAGENET1K_V1 if config.weights == "default" else None
    model = vgg19(weights=weights, progress=True)
    features = model.features
    for parameter in features.parameters():
        parameter.requires_grad_(False)
    weight_signature = (
        "VGG19_Weights.IMAGENET1K_V1"
        if config.weights == "default"
        else "random-untrained-weights"
    )
    signature = (
        f"torch={torch.__version__};torchvision={torchvision.__version__};"
        f"{weight_signature}"
    )
    del model
    return VGGDescriptorExtractor(features, config, device), signature


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算输入哈希，确保缓存不会因同名文件而误命中。"""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            block = file_handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def slugify(value: str) -> str:
    """把文件名转换为稳定且适合输出目录的 ASCII 标识。"""

    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()
    return slug or "image"


def _bounded_open_rgb(
    image_path: Path,
    max_side: int,
    max_non_jpeg_pixels: int,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    """安全解码图像；仅对 JPEG 使用解码器原生 draft 降采样。

    PIL 的默认像素上限会拒绝本项目 26065×30000 的可信高清 JPEG。函数只在读取
    头信息和执行 JPEG draft 的局部临界区临时关闭该检查；对无法原生 draft 的
    超大 PNG/TIFF 仍主动拒绝，避免压缩炸弹或数 GB 内存峰值。
    """

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(image_path) as opened:
            original_size = opened.size
            image_format = (opened.format or "").upper()
            pixel_count = original_size[0] * original_size[1]
            if (
                image_format not in {"JPEG", "JPG"}
                and pixel_count > max_non_jpeg_pixels
            ):
                raise ValueError(
                    "非 JPEG 输入像素过大，无法进行解码器原生降采样："
                    f"{original_size[0]}×{original_size[1]}。"
                )
            if image_format in {"JPEG", "JPG"} and max(original_size) > max_side:
                opened.draft("RGB", (max_side, max_side))
            oriented = ImageOps.exif_transpose(opened)
            decoded_size = oriented.size
            oriented.thumbnail(
                (max_side, max_side),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
            oriented.load()
            rgb = oriented.convert("RGB").copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    return rgb, original_size, decoded_size


def load_and_letterbox(
    image_path: Path,
    image_size: int,
    max_non_jpeg_pixels: int = 100_000_000,
) -> LoadedImage:
    """等比例缩放输入，并以 ImageNet 均值色居中填充为正方形。"""

    if not image_path.is_file():
        raise FileNotFoundError(f"找不到输入图像：{image_path}")
    rgb, original_size, decoded_size = _bounded_open_rgb(
        image_path,
        max_side=image_size,
        max_non_jpeg_pixels=max_non_jpeg_pixels,
    )
    resized_size = rgb.size
    fill = tuple(int(round(channel * 255.0)) for channel in IMAGENET_MEAN)
    square = Image.new("RGB", (image_size, image_size), color=fill)
    offset_x = (image_size - resized_size[0]) // 2
    offset_y = (image_size - resized_size[1]) // 2
    square.paste(rgb, (offset_x, offset_y))
    return LoadedImage(
        square_rgb=square,
        original_width=original_size[0],
        original_height=original_size[1],
        decoded_width=decoded_size[0],
        decoded_height=decoded_size[1],
        resized_width=resized_size[0],
        resized_height=resized_size[1],
        offset_x=offset_x,
        offset_y=offset_y,
    )


def pil_to_normalized_tensor(image: Image.Image) -> Tensor:
    """把 RGB PIL 图像转换为 ImageNet 归一化的 [1,3,H,W] 张量。"""

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1)
    standard_deviation = torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(
        1, 3, 1, 1
    )
    return (tensor - mean) / standard_deviation


def _tensor_descriptors_to_numpy(descriptors: TensorDescriptors) -> FeatureBundle:
    return FeatureBundle(
        metadata={},
        grams={
            layer: tensor.detach().numpy().astype(np.float32, copy=False)
            for layer, tensor in descriptors.grams.items()
        },
        spatial={
            layer: tensor.detach().numpy().astype(np.float32, copy=False)
            for layer, tensor in descriptors.spatial.items()
        },
        activations={
            layer: tensor.detach().numpy().astype(np.float32, copy=False)
            for layer, tensor in descriptors.activations.items()
        },
    )


def make_cache_key(
    image_sha256: str,
    config: AnalysisConfig,
    model_signature: str,
) -> str:
    payload = config.cache_payload()
    payload["image_sha256"] = image_sha256
    payload["model_signature"] = model_signature
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_feature_bundle(path: Path, bundle: FeatureBundle) -> None:
    """原子写入不允许 pickle 的 NPZ 缓存。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(bundle.metadata, sort_keys=True, ensure_ascii=False)
        )
    }
    arrays.update({f"gram__{name}": value for name, value in bundle.grams.items()})
    arrays.update({f"spatial__{name}": value for name, value in bundle.spatial.items()})
    arrays.update(
        {f"activation__{name}": value for name, value in bundle.activations.items()}
    )
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as file_handle:
            np.savez_compressed(file_handle, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_feature_bundle(path: Path) -> FeatureBundle:
    """读取并验证本工具生成的 NPZ 缓存。"""

    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError(f"缓存缺少 metadata_json：{path}")
        decoded = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(decoded, dict):
            raise ValueError(f"缓存元数据不是对象：{path}")
        format_version = decoded.get("format_version")
        if format_version != FORMAT_VERSION:
            raise ValueError(
                f"缓存格式版本不兼容：期望 {FORMAT_VERSION}，实际 {format_version}。"
            )
        grams = {
            key.removeprefix("gram__"): archive[key].astype(np.float32, copy=True)
            for key in archive.files
            if key.startswith("gram__")
        }
        spatial = {
            key.removeprefix("spatial__"): archive[key].astype(np.float32, copy=True)
            for key in archive.files
            if key.startswith("spatial__")
        }
        activations = {
            key.removeprefix("activation__"): archive[key].astype(np.float32, copy=True)
            for key in archive.files
            if key.startswith("activation__")
        }
    return FeatureBundle(
        metadata={str(key): value for key, value in decoded.items()},
        grams=grams,
        spatial=spatial,
        activations=activations,
    )


def extract_with_cache(
    image_path: Path,
    extractor: VGGDescriptorExtractor,
    config: AnalysisConfig,
    model_signature: str,
    cache_dir: Path,
    force: bool = False,
) -> ExtractionResult:
    """提取单张图像；相同内容与配置直接复用缓存。"""

    resolved_path = image_path.expanduser().resolve()
    image_sha256 = sha256_file(resolved_path)
    cache_key = make_cache_key(image_sha256, config, model_signature)
    cache_path = cache_dir / f"{cache_key}.npz"
    image_id = f"{slugify(resolved_path.stem)}-{image_sha256[:10]}"
    if cache_path.is_file() and not force:
        cached_bundle = load_feature_bundle(cache_path)
        # 同一像素内容可能来自不同实验条件（例如多个 0% 基线）。缓存只复用张量，
        # 对外清单必须保留本次输入的条件名与路径，不能继承最先写入缓存的文件名。
        cached_bundle.metadata = dict(cached_bundle.metadata)
        cached_bundle.metadata["image_path"] = project_relative_path(resolved_path)
        cached_bundle.metadata["image_id"] = image_id
        return ExtractionResult(
            bundle=cached_bundle,
            cache_path=cache_path,
            cache_hit=True,
            image_id=image_id,
        )

    loaded = load_and_letterbox(
        resolved_path,
        image_size=config.image_size,
        max_non_jpeg_pixels=config.max_non_jpeg_pixels,
    )
    input_tensor = pil_to_normalized_tensor(loaded.square_rgb)
    started = time.perf_counter()
    tensor_descriptors = extractor.extract(input_tensor)
    elapsed_seconds = time.perf_counter() - started
    bundle = _tensor_descriptors_to_numpy(tensor_descriptors)
    bundle.metadata = {
        "format_version": FORMAT_VERSION,
        "image_path": project_relative_path(resolved_path),
        "image_sha256": image_sha256,
        "image_id": image_id,
        "model_name": MODEL_NAME,
        "model_signature": model_signature,
        "weights": config.weights,
        "device_used": str(extractor.device),
        "image_size": config.image_size,
        "spatial_size": config.spatial_size,
        "style_layers": list(config.style_layers),
        "spatial_layers": list(config.spatial_layers),
        "activation_layers": list(config.activation_layers),
        "original_width": loaded.original_width,
        "original_height": loaded.original_height,
        "decoded_width": loaded.decoded_width,
        "decoded_height": loaded.decoded_height,
        "resized_width": loaded.resized_width,
        "resized_height": loaded.resized_height,
        "offset_x": loaded.offset_x,
        "offset_y": loaded.offset_y,
        "inference_seconds": elapsed_seconds,
    }
    save_feature_bundle(cache_path, bundle)
    return ExtractionResult(
        bundle=bundle,
        cache_path=cache_path,
        cache_hit=False,
        image_id=image_id,
    )


def _validate_same_layers(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    descriptor_name: str,
) -> tuple[str, ...]:
    reference_layers = set(reference)
    candidate_layers = set(candidate)
    if reference_layers != candidate_layers:
        raise ValueError(
            f"{descriptor_name} 层不一致：reference={sorted(reference_layers)}, "
            f"candidate={sorted(candidate_layers)}"
        )
    return tuple(sorted(reference_layers, key=lambda name: VGG19_LAYER_INDICES[name]))


def compute_style_distances(
    reference: FeatureBundle,
    candidate: FeatureBundle,
) -> dict[str, dict[str, float]]:
    """逐层计算 Gram 的绝对及相对 Frobenius 距离。"""

    layers = _validate_same_layers(reference.grams, candidate.grams, "Gram")
    distances: dict[str, dict[str, float]] = {}
    absolute_values: list[float] = []
    relative_values: list[float] = []
    for layer in layers:
        reference_gram = reference.grams[layer].astype(np.float64, copy=False)
        candidate_gram = candidate.grams[layer].astype(np.float64, copy=False)
        if reference_gram.shape != candidate_gram.shape:
            raise ValueError(f"{layer} Gram 形状不一致。")
        absolute = float(np.linalg.norm(candidate_gram - reference_gram))
        denominator = max(float(np.linalg.norm(reference_gram)), 1e-12)
        relative = absolute / denominator
        distances[layer] = {
            "absolute_frobenius": absolute,
            "relative_frobenius": relative,
        }
        absolute_values.append(absolute)
        relative_values.append(relative)
    distances["aggregate"] = {
        "absolute_frobenius": float(np.mean(absolute_values)),
        "relative_frobenius": float(np.mean(relative_values)),
    }
    return distances


def compute_spatial_distances(
    reference: FeatureBundle,
    candidate: FeatureBundle,
) -> dict[str, dict[str, float]]:
    """比较保留网格位置的激活，报告相对 RMS 与平均余弦距离。"""

    layers = _validate_same_layers(reference.spatial, candidate.spatial, "spatial")
    distances: dict[str, dict[str, float]] = {}
    rms_values: list[float] = []
    cosine_values: list[float] = []
    for layer in layers:
        reference_feature = reference.spatial[layer].astype(np.float64, copy=False)
        candidate_feature = candidate.spatial[layer].astype(np.float64, copy=False)
        if reference_feature.shape != candidate_feature.shape:
            raise ValueError(f"{layer} spatial feature 形状不一致。")
        difference_rms = float(
            np.sqrt(np.mean(np.square(candidate_feature - reference_feature)))
        )
        reference_rms = max(
            float(np.sqrt(np.mean(np.square(reference_feature)))), 1e-12
        )
        relative_rms = difference_rms / reference_rms

        reference_vectors = np.moveaxis(reference_feature, 0, -1).reshape(
            -1, reference_feature.shape[0]
        )
        candidate_vectors = np.moveaxis(candidate_feature, 0, -1).reshape(
            -1, candidate_feature.shape[0]
        )
        numerator = np.sum(reference_vectors * candidate_vectors, axis=1)
        denominator = np.linalg.norm(reference_vectors, axis=1) * np.linalg.norm(
            candidate_vectors, axis=1
        )
        valid = denominator > 1e-12
        if np.any(valid):
            cosine_distance = float(
                np.mean(1.0 - numerator[valid] / denominator[valid])
            )
        else:
            cosine_distance = 0.0
        distances[layer] = {
            "relative_rms": relative_rms,
            "mean_cosine_distance": cosine_distance,
        }
        rms_values.append(relative_rms)
        cosine_values.append(cosine_distance)
    distances["aggregate"] = {
        "relative_rms": float(np.mean(rms_values)),
        "mean_cosine_distance": float(np.mean(cosine_values)),
    }
    return distances


def normalize_heatmap(activation: np.ndarray) -> np.ndarray:
    """使用稳健分位数把激活映射到 uint8，减少极端值支配可视化。"""

    if activation.ndim != 2:
        raise ValueError("activation 必须是二维数组。")
    finite = activation[np.isfinite(activation)]
    if finite.size == 0:
        return np.zeros(activation.shape, dtype=np.uint8)
    lower, upper = np.percentile(finite, (1.0, 99.0))
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
        return np.zeros(activation.shape, dtype=np.uint8)
    if upper <= lower + 1e-12:
        return np.zeros(activation.shape, dtype=np.uint8)
    normalized = np.clip((activation - lower) / (upper - lower), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def save_activation_visualizations(
    image_path: Path,
    result: ExtractionResult,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """保存逐层热图和叠加图，叠加底图使用与 CNN 完全相同的 letterbox。"""

    loaded = load_and_letterbox(
        image_path.expanduser().resolve(),
        image_size=config.image_size,
        max_non_jpeg_pixels=config.max_non_jpeg_pixels,
    )
    base_rgb = np.asarray(loaded.square_rgb, dtype=np.uint8)
    image_output_dir = output_dir / "activations" / result.image_id
    image_output_dir.mkdir(parents=True, exist_ok=True)
    loaded.square_rgb.save(image_output_dir / "cnn-input.png")
    for layer_name in config.activation_layers:
        activation = result.bundle.activations[layer_name]
        gray = normalize_heatmap(activation)
        resized_gray = cv2.resize(
            gray,
            (config.image_size, config.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        color_bgr = cv2.applyColorMap(resized_gray, cv2.COLORMAP_INFERNO)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        overlay = np.round(0.55 * base_rgb + 0.45 * color_rgb).astype(np.uint8)
        Image.fromarray(color_rgb, mode="RGB").save(
            image_output_dir / f"{layer_name}-heatmap.png"
        )
        Image.fromarray(overlay, mode="RGB").save(
            image_output_dir / f"{layer_name}-overlay.png"
        )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                payload, file_handle, indent=2, ensure_ascii=False, sort_keys=True
            )
            file_handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_distance_csvs(
    output_dir: Path,
    comparisons: Sequence[dict[str, object]],
) -> None:
    style_path = output_dir / "style-distances.csv"
    spatial_path = output_dir / "spatial-distances.csv"
    with style_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                "candidate_image_id",
                "candidate_path",
                "layer",
                "absolute_frobenius",
                "relative_frobenius",
            ),
        )
        writer.writeheader()
        for comparison in comparisons:
            style = comparison["style"]
            if not isinstance(style, dict):
                raise TypeError("comparison.style 必须是对象。")
            for layer_name, values in style.items():
                if not isinstance(values, dict):
                    raise TypeError("style layer 必须是对象。")
                writer.writerow(
                    {
                        "candidate_image_id": comparison["candidate_image_id"],
                        "candidate_path": comparison["candidate_path"],
                        "layer": layer_name,
                        "absolute_frobenius": values["absolute_frobenius"],
                        "relative_frobenius": values["relative_frobenius"],
                    }
                )
    with spatial_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                "candidate_image_id",
                "candidate_path",
                "layer",
                "relative_rms",
                "mean_cosine_distance",
            ),
        )
        writer.writeheader()
        for comparison in comparisons:
            spatial = comparison["spatial"]
            if not isinstance(spatial, dict):
                raise TypeError("comparison.spatial 必须是对象。")
            for layer_name, values in spatial.items():
                if not isinstance(values, dict):
                    raise TypeError("spatial layer 必须是对象。")
                writer.writerow(
                    {
                        "candidate_image_id": comparison["candidate_image_id"],
                        "candidate_path": comparison["candidate_path"],
                        "layer": layer_name,
                        "relative_rms": values["relative_rms"],
                        "mean_cosine_distance": values["mean_cosine_distance"],
                    }
                )


def analyze(
    reference_path: Path,
    candidate_paths: Sequence[Path],
    output_dir: Path,
    cache_dir: Path,
    config: AnalysisConfig,
    requested_device: DeviceChoice,
    force: bool,
    seed: int,
) -> dict[str, object]:
    """执行完整分析并将描述符、激活图、JSON 与 CSV 写入输出目录。"""

    config.validate()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device(requested_device)
    extractor, model_signature = create_vgg19_extractor(config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ordered_paths: list[Path] = [reference_path]
    seen = {reference_path.expanduser().resolve()}
    for candidate_path in candidate_paths:
        resolved = candidate_path.expanduser().resolve()
        if resolved not in seen:
            ordered_paths.append(candidate_path)
            seen.add(resolved)

    results: list[ExtractionResult] = []
    for image_path in ordered_paths:
        result = extract_with_cache(
            image_path=image_path,
            extractor=extractor,
            config=config,
            model_signature=model_signature,
            cache_dir=cache_dir,
            force=force,
        )
        results.append(result)
        save_activation_visualizations(
            image_path=image_path,
            result=result,
            output_dir=output_dir,
            config=config,
        )
        descriptor_output = output_dir / "descriptors" / f"{result.image_id}.npz"
        descriptor_output.parent.mkdir(parents=True, exist_ok=True)
        save_feature_bundle(descriptor_output, result.bundle)

    reference_result = results[0]
    comparisons: list[dict[str, object]] = []
    for candidate_result in results[1:]:
        comparisons.append(
            {
                "candidate_image_id": candidate_result.image_id,
                "candidate_path": candidate_result.bundle.metadata["image_path"],
                "style": compute_style_distances(
                    reference_result.bundle, candidate_result.bundle
                ),
                "spatial": compute_spatial_distances(
                    reference_result.bundle, candidate_result.bundle
                ),
            }
        )

    summary: dict[str, object] = {
        "analysis": "VGG19 multi-layer style and spatial-content comparison",
        "format_version": FORMAT_VERSION,
        "reference_image_id": reference_result.image_id,
        "reference_path": reference_result.bundle.metadata["image_path"],
        "device": str(device),
        "model_signature": model_signature,
        "config": config.cache_payload(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "images": [
            {
                "image_id": result.image_id,
                "image_path": result.bundle.metadata["image_path"],
                "image_sha256": result.bundle.metadata["image_sha256"],
                "cache_hit": result.cache_hit,
                "cache_path": project_relative_path(result.cache_path),
                "original_size": [
                    result.bundle.metadata["original_width"],
                    result.bundle.metadata["original_height"],
                ],
                "decoded_size": [
                    result.bundle.metadata["decoded_width"],
                    result.bundle.metadata["decoded_height"],
                ],
                "letterboxed_content_size": [
                    result.bundle.metadata["resized_width"],
                    result.bundle.metadata["resized_height"],
                ],
            }
            for result in results
        ],
        "comparisons": comparisons,
        "interpretation_limits": [
            "Gram distance measures CNN texture/channel co-activation, not complete art-historical style.",
            "Spatial-content distance can still respond to color and texture; it is not a pure geometry metric.",
            "ImageNet-pretrained VGG19 was trained primarily on photographs, so painting-domain bias is expected.",
        ],
        "seed": seed,
    }
    write_json(output_dir / "cnn-analysis.json", summary)
    write_distance_csvs(output_dir, comparisons)
    return summary


def parse_layer_list(value: str) -> tuple[str, ...]:
    layers = tuple(item.strip() for item in value.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("层列表不能为空。")
    unknown = sorted(set(layers) - set(VGG19_LAYER_INDICES))
    if unknown:
        raise argparse.ArgumentTypeError(f"未知 VGG19 层：{', '.join(unknown)}")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("层列表不能包含重复项。")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提取 VGG19 多层 activation、Gram 风格描述符与空间内容距离。"
    )
    parser.add_argument("--reference", type=Path, required=True, help="参考原作路径。")
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=(),
        help="需要与参考图比较的一个或多个图像。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cnn-analysis"),
        help="可提交结果目录。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/cnn-analysis"),
        help="内容寻址的 NPZ 缓存目录。",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument(
        "--style-layers",
        type=parse_layer_list,
        default=DEFAULT_STYLE_LAYERS,
        help="逗号分隔的 Gram 层。",
    )
    parser.add_argument(
        "--spatial-layers",
        type=parse_layer_list,
        default=DEFAULT_SPATIAL_LAYERS,
        help="逗号分隔的二维空间特征层。",
    )
    parser.add_argument(
        "--activation-layers",
        type=parse_layer_list,
        default=DEFAULT_ACTIVATION_LAYERS,
        help="逗号分隔的激活可视化层。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--weights",
        choices=("default", "none"),
        default="default",
        help="正式分析必须使用 default；none 只用于离线工程测试。",
    )
    parser.add_argument(
        "--max-non-jpeg-pixels",
        type=int,
        default=100_000_000,
        help="非 JPEG 输入的像素安全上限。",
    )
    parser.add_argument("--seed", type=int, default=139)
    parser.add_argument("--force", action="store_true", help="忽略已有缓存重新提取。")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = AnalysisConfig(
        image_size=arguments.image_size,
        spatial_size=arguments.spatial_size,
        style_layers=tuple(arguments.style_layers),
        spatial_layers=tuple(arguments.spatial_layers),
        activation_layers=tuple(arguments.activation_layers),
        weights=arguments.weights,
        max_non_jpeg_pixels=arguments.max_non_jpeg_pixels,
    )
    try:
        summary = analyze(
            reference_path=arguments.reference,
            candidate_paths=arguments.images,
            output_dir=arguments.output_dir,
            cache_dir=arguments.cache_dir,
            config=config,
            requested_device=arguments.device,
            force=arguments.force,
            seed=arguments.seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str((arguments.output_dir / "cnn-analysis.json").resolve()),
                "device": summary["device"],
                "images": len(summary["images"]),
                "comparisons": len(summary["comparisons"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
