#!/usr/bin/env python3
"""从 CNN 边缘响应中提取直线、方向分布与主消失点。

首选后端是 Xie 与 Tu 发布的原始 HED Caffe 模型；模型与 deploy 文件分别从
UCSD 原作者站点和原作者 GitHub 仓库下载，并以 SHA-256 校验。若 ``auto`` 模式
无法取得 HED，脚本会明确记录原因并改用 ImageNet VGG19 多层激活不连续性图。
后者是严谨、可重复的 CNN 替代测量，但不会被误称为经过边缘监督训练的 HED。

无论选用哪种 CNN 后端，脚本都会额外运行 Canny + Hough 作为传统视觉对照。
消失点通过带长度权重的 RANSAC 求得，再对内点直线执行加权最小二乘精化。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import unicodedata
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

FORMAT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HED_MODEL_URL = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
HED_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt"
)
HED_MODEL_SHA256 = "4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7"
HED_PROTOTXT_SHA256 = "378a9246383da889cf8e0290c47554d75dcf9c5b6bbabd8ab6c481c34aa12b8a"
HED_BGR_MEAN = (104.00698793, 116.66876762, 122.67891434)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BackendChoice = Literal["auto", "hed", "vgg", "canny"]
DeviceChoice = Literal["auto", "cpu", "mps", "cuda"]


def project_relative_path(path: str | Path) -> str:
    """优先输出项目相对路径，避免公开清单包含本机绝对目录。"""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class GeometryConfig:
    max_side: int = 1280
    hough_threshold: int = 45
    min_line_length_ratio: float = 0.055
    max_line_gap_ratio: float = 0.012
    max_lines: int = 400
    ransac_iterations: int = 5000
    inlier_distance_ratio: float = 0.012
    minimum_pair_angle_degrees: float = 5.0
    intersection_bound_ratio: float = 2.0
    vanishing_roi: tuple[float, float, float, float] | None = None
    seed: int = 139
    max_non_jpeg_pixels: int = 100_000_000

    def validate(self) -> None:
        if self.max_side < 128:
            raise ValueError("max_side 必须至少为 128。")
        if self.hough_threshold < 1:
            raise ValueError("hough_threshold 必须为正整数。")
        if not 0.0 < self.min_line_length_ratio <= 1.0:
            raise ValueError("min_line_length_ratio 必须位于 (0, 1]。")
        if not 0.0 <= self.max_line_gap_ratio <= 1.0:
            raise ValueError("max_line_gap_ratio 必须位于 [0, 1]。")
        if self.max_lines < 2:
            raise ValueError("max_lines 必须至少为 2。")
        if self.ransac_iterations < 1:
            raise ValueError("ransac_iterations 必须为正整数。")
        if not 0.0 < self.inlier_distance_ratio <= 0.25:
            raise ValueError("inlier_distance_ratio 必须位于 (0, 0.25]。")
        if not 0.0 < self.minimum_pair_angle_degrees < 90.0:
            raise ValueError("minimum_pair_angle_degrees 必须位于 (0, 90)。")
        if self.intersection_bound_ratio < 1.0:
            raise ValueError("intersection_bound_ratio 必须至少为 1。")
        if self.vanishing_roi is not None:
            minimum_x, minimum_y, maximum_x, maximum_y = self.vanishing_roi
            if not (
                0.0 <= minimum_x < maximum_x <= 1.0
                and 0.0 <= minimum_y < maximum_y <= 1.0
            ):
                raise ValueError(
                    "vanishing_roi 必须是位于 [0,1] 内且顺序为 x_min,y_min,x_max,y_max 的矩形。"
                )
        if self.max_non_jpeg_pixels < 1:
            raise ValueError("max_non_jpeg_pixels 必须为正整数。")


@dataclass(frozen=True)
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_degrees: float

    @classmethod
    def from_endpoints(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> LineSegment:
        delta_x = x2 - x1
        delta_y = y2 - y1
        length = math.hypot(delta_x, delta_y)
        if length <= 0.0:
            raise ValueError("线段端点不能重合。")
        angle = math.degrees(math.atan2(delta_y, delta_x)) % 180.0
        return cls(x1, y1, x2, y2, length, angle)

    def normalized_equation(self) -> tuple[float, float, float]:
        """返回单位法向量形式 a*x + b*y + c = 0。"""

        a = self.y1 - self.y2
        b = self.x2 - self.x1
        norm = math.hypot(a, b)
        if norm <= 1e-12:
            raise ValueError("无法为零长度线段建立直线方程。")
        c = self.x1 * self.y2 - self.x2 * self.y1
        return a / norm, b / norm, c / norm


@dataclass(frozen=True)
class VanishingPointEstimate:
    x: float
    y: float
    normalized_x: float
    normalized_y: float
    inlier_count: int
    total_line_count: int
    weighted_inlier_ratio: float
    median_inlier_distance_pixels: float
    ransac_iterations: int


@dataclass(frozen=True)
class HedAssets:
    prototxt_path: Path
    model_path: Path


@dataclass(frozen=True)
class EdgeDetectionResult:
    probability: np.ndarray
    backend_used: str
    backend_detail: str
    fallback_reason: str | None


class CropLayer:
    """OpenCV DNN 执行原版 HED deploy.prototxt 所需的 Caffe Crop 层。"""

    def __init__(self, params: object, blobs: object) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    def getMemoryShapes(
        self,
        inputs: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        source_shape = inputs[0]
        target_shape = inputs[1]
        batch_size, channels = int(source_shape[0]), int(source_shape[1])
        target_height, target_width = int(target_shape[2]), int(target_shape[3])
        self.start_y = (int(source_shape[2]) - target_height) // 2
        self.start_x = (int(source_shape[3]) - target_width) // 2
        self.end_y = self.start_y + target_height
        self.end_x = self.start_x + target_width
        return [[batch_size, channels, target_height, target_width]]

    def forward(self, inputs: Sequence[np.ndarray]) -> list[np.ndarray]:
        cropped = inputs[0][
            :,
            :,
            self.start_y : self.end_y,
            self.start_x : self.end_x,
        ]
        return [cropped]


def default_hed_cache_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "las-meninas" / "hed"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "las-meninas" / "hed"
    return Path.home() / ".cache" / "las-meninas" / "hed"


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            block = file_handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _download_verified(
    url: str,
    destination: Path,
    expected_sha256: str,
    timeout_seconds: int = 60,
) -> None:
    """流式下载并校验；只有完整文件通过 SHA-256 后才原子替换目标。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Las-Meninas-Geometry-Analysis/1.0"},
    )
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout_seconds) as response,
            temporary_path.open("wb") as file_handle,
        ):
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                file_handle.write(block)
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"下载文件 SHA-256 不匹配：{url}，实际 {actual_sha256}。"
            )
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _ensure_verified_asset(
    path: Path,
    url: str,
    expected_sha256: str,
    allow_download: bool,
) -> None:
    if path.is_file() and sha256_file(path) == expected_sha256:
        return
    if path.exists():
        quarantine = path.with_name(
            f"{path.name}.invalid-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        path.replace(quarantine)
    if not allow_download:
        raise FileNotFoundError(f"HED 资源不存在或校验失败，且下载已禁用：{path}")
    print(f"正在从原始来源下载并校验：{url}", file=sys.stderr)
    _download_verified(url, path, expected_sha256)


def ensure_hed_assets(
    cache_dir: Path,
    allow_download: bool = True,
    prototxt_override: Path | None = None,
    model_override: Path | None = None,
) -> HedAssets:
    """解析并校验 HED 资源；自定义路径同样必须匹配原始文件哈希。"""

    prototxt_path = (
        prototxt_override.expanduser().resolve()
        if prototxt_override is not None
        else cache_dir.expanduser().resolve() / "deploy.prototxt"
    )
    model_path = (
        model_override.expanduser().resolve()
        if model_override is not None
        else cache_dir.expanduser().resolve() / "hed_pretrained_bsds.caffemodel"
    )
    if prototxt_override is not None:
        if not prototxt_path.is_file():
            raise FileNotFoundError(f"找不到 HED prototxt：{prototxt_path}")
        actual = sha256_file(prototxt_path)
        if actual != HED_PROTOTXT_SHA256:
            raise ValueError(f"HED prototxt 校验失败：{actual}")
    else:
        _ensure_verified_asset(
            prototxt_path,
            HED_PROTOTXT_URL,
            HED_PROTOTXT_SHA256,
            allow_download,
        )
    if model_override is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到 HED caffemodel：{model_path}")
        actual = sha256_file(model_path)
        if actual != HED_MODEL_SHA256:
            raise ValueError(f"HED caffemodel 校验失败：{actual}")
    else:
        _ensure_verified_asset(
            model_path,
            HED_MODEL_URL,
            HED_MODEL_SHA256,
            allow_download,
        )
    return HedAssets(prototxt_path=prototxt_path, model_path=model_path)


def load_bounded_bgr(
    image_path: Path,
    max_side: int,
    max_non_jpeg_pixels: int = 100_000_000,
) -> tuple[np.ndarray, tuple[int, int]]:
    """有界解码图像并返回 BGR 数组及原始 (宽, 高)。"""

    if not image_path.is_file():
        raise FileNotFoundError(f"找不到输入图像：{image_path}")
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
            oriented.thumbnail(
                (max_side, max_side),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
            oriented.load()
            rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8).copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), original_size


def _register_hed_crop_layer() -> None:
    try:
        cv2.dnn_registerLayer("Crop", CropLayer)
    except cv2.error as exc:
        message = str(exc).lower()
        if "already" not in message and "registered" not in message:
            raise


def detect_edges_hed(image_bgr: np.ndarray, assets: HedAssets) -> np.ndarray:
    """运行原始 HED fusion output，返回 [0,1] float32 概率图。"""

    _register_hed_crop_layer()
    height, width = image_bgr.shape[:2]
    network = cv2.dnn.readNetFromCaffe(
        str(assets.prototxt_path),
        str(assets.model_path),
    )
    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=1.0,
        size=(width, height),
        mean=HED_BGR_MEAN,
        swapRB=False,
        crop=False,
    )
    network.setInput(blob)
    prediction = network.forward()
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise RuntimeError(f"HED 输出形状异常：{prediction.shape}")
    probability = prediction[0, 0]
    if probability.shape != (height, width):
        probability = cv2.resize(
            probability,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def _choose_torch_device(requested: DeviceChoice):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("VGG 几何后端需要 PyTorch。") from exc
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


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    lower, upper = np.percentile(finite, (2.0, 98.5))
    if upper <= lower + 1e-12:
        return np.zeros(values.shape, dtype=np.float32)
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def detect_edges_vgg(
    image_bgr: np.ndarray,
    requested_device: DeviceChoice,
) -> tuple[np.ndarray, str]:
    """融合 VGG19 多尺度 feature discontinuity，作为 HED 的透明替代。

    对 relu1_2、relu2_2、relu3_4 的每个通道计算相邻网格差分的 L2 幅值，再
    分位数归一化、上采样和加权融合。该结果确实来自预训练 CNN 的层级表征，
    但没有 BSDS 边缘监督，元数据会明确保留这一限制。
    """

    try:
        import torch
        import torch.nn.functional as torch_functional
        from torchvision.models import VGG19_Weights, vgg19
    except ImportError as exc:
        raise RuntimeError("VGG 几何后端需要 torch 与 torchvision。") from exc

    device = _choose_torch_device(requested_device)
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1, progress=True).features.eval()
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1)
    standard_deviation = torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(
        1, 3, 1, 1
    )
    current = ((tensor - mean) / standard_deviation).to(device)
    selected = {3: 0.50, 8: 0.30, 17: 0.20}
    height, width = image_bgr.shape[:2]
    fused = np.zeros((height, width), dtype=np.float32)
    with torch.inference_mode():
        for layer_index, layer in enumerate(model):
            current = layer(current)
            weight = selected.get(layer_index)
            if weight is not None:
                delta_x = torch_functional.pad(
                    current[:, :, :, 1:] - current[:, :, :, :-1],
                    (0, 1, 0, 0),
                )
                delta_y = torch_functional.pad(
                    current[:, :, 1:, :] - current[:, :, :-1, :],
                    (0, 0, 0, 1),
                )
                magnitude = torch.sqrt(
                    torch.mean(torch.square(delta_x) + torch.square(delta_y), dim=1)
                    + 1e-12
                )
                resized = torch_functional.interpolate(
                    magnitude.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                normalized = _robust_unit_interval(resized.cpu().numpy())
                fused += float(weight) * normalized
            if layer_index >= max(selected):
                break
    detail = f"VGG19_Weights.IMAGENET1K_V1 activation-discontinuity;device={device}"
    return _robust_unit_interval(fused), detail


def detect_edges_canny(image_bgr: np.ndarray) -> np.ndarray:
    """传统对照：基于图像中位亮度自动设置 Canny 双阈值。"""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)
    median = float(np.median(blurred))
    lower = int(max(0.0, (1.0 - 0.33) * median))
    upper = int(min(255.0, (1.0 + 0.33) * median))
    if upper <= lower:
        upper = min(255, lower + 32)
    edges = cv2.Canny(blurred, lower, upper, L2gradient=True)
    return edges.astype(np.float32) / 255.0


def detect_primary_edges(
    image_bgr: np.ndarray,
    backend: BackendChoice,
    hed_cache_dir: Path,
    allow_download: bool,
    prototxt_override: Path | None,
    model_override: Path | None,
    device: DeviceChoice,
) -> EdgeDetectionResult:
    """运行所选后端；auto 的回退原因会进入最终 JSON，不会静默掩盖失败。"""

    if backend == "canny":
        return EdgeDetectionResult(
            probability=detect_edges_canny(image_bgr),
            backend_used="canny",
            backend_detail="OpenCV Canny with median-derived thresholds",
            fallback_reason=None,
        )
    if backend in {"auto", "hed"}:
        try:
            assets = ensure_hed_assets(
                cache_dir=hed_cache_dir,
                allow_download=allow_download,
                prototxt_override=prototxt_override,
                model_override=model_override,
            )
            probability = detect_edges_hed(image_bgr, assets)
            return EdgeDetectionResult(
                probability=probability,
                backend_used="hed",
                backend_detail=(
                    "Xie-Tu HED fusion-output;"
                    f"model_sha256={HED_MODEL_SHA256};"
                    f"prototxt_sha256={HED_PROTOTXT_SHA256}"
                ),
                fallback_reason=None,
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError, cv2.error) as exc:
            if backend == "hed":
                raise RuntimeError(f"显式 HED 后端失败：{exc}") from exc
            fallback_reason = f"HED unavailable: {type(exc).__name__}: {exc}"
            probability, detail = detect_edges_vgg(image_bgr, device)
            return EdgeDetectionResult(
                probability=probability,
                backend_used="vgg",
                backend_detail=detail,
                fallback_reason=fallback_reason,
            )
    probability, detail = detect_edges_vgg(image_bgr, device)
    return EdgeDetectionResult(
        probability=probability,
        backend_used="vgg",
        backend_detail=detail,
        fallback_reason=None,
    )


def probability_to_thin_edges(probability: np.ndarray) -> np.ndarray:
    """在 CNN 概率/响应图上执行 Canny NMS 与滞后连接，供 Hough 使用。"""

    normalized = _robust_unit_interval(probability)
    response_u8 = np.round(normalized * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(response_u8, (3, 3), sigmaX=0.8)
    return cv2.Canny(blurred, 45, 130, L2gradient=True)


def extract_hough_segments(
    thin_edges: np.ndarray,
    config: GeometryConfig,
) -> list[LineSegment]:
    """用概率 Hough 提取并按长度截断线段集合。"""

    if thin_edges.ndim != 2 or thin_edges.dtype != np.uint8:
        raise ValueError("thin_edges 必须是二维 uint8 数组。")
    height, width = thin_edges.shape
    diagonal = math.hypot(width, height)
    minimum_length = max(8, int(round(config.min_line_length_ratio * diagonal)))
    maximum_gap = max(1, int(round(config.max_line_gap_ratio * diagonal)))
    raw_lines = cv2.HoughLinesP(
        thin_edges,
        rho=1.0,
        theta=np.pi / 720.0,
        threshold=config.hough_threshold,
        minLineLength=minimum_length,
        maxLineGap=maximum_gap,
    )
    if raw_lines is None:
        return []
    segments: list[LineSegment] = []
    for coordinates in raw_lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        try:
            segments.append(LineSegment.from_endpoints(x1, y1, x2, y2))
        except ValueError:
            continue
    segments.sort(key=lambda segment: segment.length, reverse=True)
    return segments[: config.max_lines]


def intersect_infinite_lines(
    first: LineSegment,
    second: LineSegment,
    minimum_angle_degrees: float,
) -> tuple[float, float] | None:
    """求两条无限延长直线交点；近平行线返回 None。"""

    angle_difference = abs(first.angle_degrees - second.angle_degrees) % 180.0
    acute_difference = min(angle_difference, 180.0 - angle_difference)
    if acute_difference < minimum_angle_degrees:
        return None
    a1, b1, c1 = first.normalized_equation()
    a2, b2, c2 = second.normalized_equation()
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) <= 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / determinant
    y = (c1 * a2 - c2 * a1) / determinant
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _line_distances(
    segments: Sequence[LineSegment],
    point_x: float,
    point_y: float,
) -> np.ndarray:
    equations = np.asarray(
        [segment.normalized_equation() for segment in segments], dtype=np.float64
    )
    return np.abs(
        equations[:, 0] * point_x + equations[:, 1] * point_y + equations[:, 2]
    )


def _refine_vanishing_point(
    segments: Sequence[LineSegment],
    inlier_mask: np.ndarray,
) -> tuple[float, float]:
    inlier_segments = [
        segment
        for segment, is_inlier in zip(segments, inlier_mask, strict=True)
        if is_inlier
    ]
    coefficients = np.asarray(
        [segment.normalized_equation() for segment in inlier_segments],
        dtype=np.float64,
    )
    weights = np.sqrt(
        np.asarray([segment.length for segment in inlier_segments], dtype=np.float64)
    )
    design = coefficients[:, :2] * weights[:, None]
    target = -coefficients[:, 2] * weights
    solution, _, rank, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    if rank < 2 or not np.all(np.isfinite(solution)):
        raise RuntimeError("消失点加权最小二乘矩阵退化。")
    return float(solution[0]), float(solution[1])


def estimate_vanishing_point_ransac(
    segments: Sequence[LineSegment],
    image_width: int,
    image_height: int,
    config: GeometryConfig,
) -> VanishingPointEstimate | None:
    """RANSAC 估计主消失点，并用所有内点进行加权最小二乘精化。"""

    if len(segments) < 3:
        return None
    diagonal = math.hypot(image_width, image_height)
    inlier_threshold = config.inlier_distance_ratio * diagonal
    total_weight = sum(segment.length for segment in segments)
    generator = random.Random(config.seed)
    best_point: tuple[float, float] | None = None
    best_mask: np.ndarray | None = None
    best_score = -math.inf
    bound_x = config.intersection_bound_ratio * image_width
    bound_y = config.intersection_bound_ratio * image_height
    if config.vanishing_roi is None:
        minimum_x, maximum_x = -bound_x, bound_x
        minimum_y, maximum_y = -bound_y, bound_y
    else:
        (
            normalized_minimum_x,
            normalized_minimum_y,
            normalized_maximum_x,
            normalized_maximum_y,
        ) = config.vanishing_roi
        minimum_x = normalized_minimum_x * image_width
        maximum_x = normalized_maximum_x * image_width
        minimum_y = normalized_minimum_y * image_height
        maximum_y = normalized_maximum_y * image_height

    for _ in range(config.ransac_iterations):
        first_index, second_index = generator.sample(range(len(segments)), 2)
        point = intersect_infinite_lines(
            segments[first_index],
            segments[second_index],
            config.minimum_pair_angle_degrees,
        )
        if point is None:
            continue
        point_x, point_y = point
        if not (
            minimum_x <= point_x <= maximum_x and minimum_y <= point_y <= maximum_y
        ):
            continue
        distances = _line_distances(segments, point_x, point_y)
        mask = distances <= inlier_threshold
        if int(np.count_nonzero(mask)) < 3:
            continue
        inlier_weight = sum(
            segment.length
            for segment, is_inlier in zip(segments, mask, strict=True)
            if is_inlier
        )
        robust_error = float(np.median(distances[mask]))
        score = inlier_weight / max(total_weight, 1e-12) - 0.05 * (
            robust_error / max(inlier_threshold, 1e-12)
        )
        if score > best_score:
            best_score = score
            best_point = point
            best_mask = mask

    if best_point is None or best_mask is None:
        return None
    try:
        refined_x, refined_y = _refine_vanishing_point(segments, best_mask)
    except RuntimeError:
        refined_x, refined_y = best_point
    if not (
        minimum_x <= refined_x <= maximum_x and minimum_y <= refined_y <= maximum_y
    ):
        refined_x, refined_y = best_point
    refined_distances = _line_distances(segments, refined_x, refined_y)
    refined_mask = refined_distances <= inlier_threshold
    if int(np.count_nonzero(refined_mask)) >= 3:
        try:
            second_x, second_y = _refine_vanishing_point(segments, refined_mask)
        except RuntimeError:
            second_x, second_y = refined_x, refined_y
        if minimum_x <= second_x <= maximum_x and minimum_y <= second_y <= maximum_y:
            best_mask = refined_mask
            refined_x, refined_y = second_x, second_y
        refined_distances = _line_distances(segments, refined_x, refined_y)

    inlier_weight = sum(
        segment.length
        for segment, is_inlier in zip(segments, best_mask, strict=True)
        if is_inlier
    )
    return VanishingPointEstimate(
        x=refined_x,
        y=refined_y,
        normalized_x=refined_x / float(image_width),
        normalized_y=refined_y / float(image_height),
        inlier_count=int(np.count_nonzero(best_mask)),
        total_line_count=len(segments),
        weighted_inlier_ratio=inlier_weight / max(total_weight, 1e-12),
        median_inlier_distance_pixels=float(np.median(refined_distances[best_mask])),
        ransac_iterations=config.ransac_iterations,
    )


def orientation_histogram(
    segments: Sequence[LineSegment],
    bin_count: int = 18,
) -> list[dict[str, float]]:
    """按线段长度加权统计 [0,180) 方向分布。"""

    if bin_count < 1:
        raise ValueError("bin_count 必须为正整数。")
    edges = np.linspace(0.0, 180.0, bin_count + 1)
    angles = np.asarray(
        [segment.angle_degrees for segment in segments], dtype=np.float64
    )
    weights = np.asarray([segment.length for segment in segments], dtype=np.float64)
    if len(segments) == 0:
        counts = np.zeros(bin_count, dtype=np.float64)
    else:
        counts, _ = np.histogram(angles, bins=edges, weights=weights)
    total = max(float(np.sum(counts)), 1e-12)
    return [
        {
            "start_degrees": float(edges[index]),
            "end_degrees": float(edges[index + 1]),
            "length_weight": float(counts[index]),
            "proportion": float(counts[index] / total),
        }
        for index in range(bin_count)
    ]


def estimate_evidence_grade(
    estimate: VanishingPointEstimate | None,
) -> Literal["unavailable", "low", "medium", "high"]:
    """按内点数量和长度覆盖率给候选证据分级，而不是伪造概率置信度。"""

    if estimate is None:
        return "unavailable"
    if estimate.inlier_count >= 16 and estimate.weighted_inlier_ratio >= 0.30:
        return "high"
    if estimate.inlier_count >= 9 and estimate.weighted_inlier_ratio >= 0.12:
        return "medium"
    return "low"


def draw_geometry_overlay(
    image_bgr: np.ndarray,
    segments: Sequence[LineSegment],
    vanishing_point: VanishingPointEstimate | None,
    max_draw_lines: int = 180,
) -> np.ndarray:
    overlay = image_bgr.copy()
    for segment in segments[:max_draw_lines]:
        cv2.line(
            overlay,
            (int(round(segment.x1)), int(round(segment.y1))),
            (int(round(segment.x2)), int(round(segment.y2))),
            color=(255, 210, 40),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    if vanishing_point is not None:
        height, width = overlay.shape[:2]
        point = (int(round(vanishing_point.x)), int(round(vanishing_point.y)))
        horizon_y = int(round(vanishing_point.y))
        if 0 <= horizon_y < height:
            cv2.line(
                overlay,
                (0, horizon_y),
                (width - 1, horizon_y),
                color=(60, 60, 240),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        if -40 <= point[0] <= width + 40 and -40 <= point[1] <= height + 40:
            cv2.drawMarker(
                overlay,
                point,
                color=(30, 30, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=3,
                line_type=cv2.LINE_AA,
            )
    return overlay


def _write_json(path: Path, payload: object) -> None:
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


def _write_segments_csv(
    path: Path,
    primary_segments: Sequence[LineSegment],
    canny_segments: Sequence[LineSegment],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        fieldnames = (
            "source",
            "rank",
            "x1",
            "y1",
            "x2",
            "y2",
            "length",
            "angle_degrees",
        )
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, segments in (
            ("cnn_or_selected_backend", primary_segments),
            ("canny_control", canny_segments),
        ):
            for rank, segment in enumerate(segments, start=1):
                writer.writerow(
                    {
                        "source": source,
                        "rank": rank,
                        "x1": segment.x1,
                        "y1": segment.y1,
                        "x2": segment.x2,
                        "y2": segment.y2,
                        "length": segment.length,
                        "angle_degrees": segment.angle_degrees,
                    }
                )


def _safe_image_stem(path: Path) -> str:
    normalized = unicodedata.normalize("NFKD", path.stem)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    filtered = "-".join(part for part in ascii_value.replace("_", " ").split() if part)
    return filtered.lower() or "image"


def analyze_geometry(
    image_path: Path,
    output_dir: Path,
    backend: BackendChoice,
    device: DeviceChoice,
    config: GeometryConfig,
    hed_cache_dir: Path,
    allow_download: bool,
    prototxt_override: Path | None = None,
    model_override: Path | None = None,
) -> dict[str, object]:
    """运行 CNN/传统边缘、Hough、RANSAC 并保存全部可审计结果。"""

    config.validate()
    resolved_path = image_path.expanduser().resolve()
    image_bgr, original_size = load_bounded_bgr(
        resolved_path,
        max_side=config.max_side,
        max_non_jpeg_pixels=config.max_non_jpeg_pixels,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    edge_result = detect_primary_edges(
        image_bgr=image_bgr,
        backend=backend,
        hed_cache_dir=hed_cache_dir,
        allow_download=allow_download,
        prototxt_override=prototxt_override,
        model_override=model_override,
        device=device,
    )
    primary_thin = probability_to_thin_edges(edge_result.probability)
    canny_probability = detect_edges_canny(image_bgr)
    canny_thin = np.round(canny_probability * 255.0).astype(np.uint8)

    primary_segments = extract_hough_segments(primary_thin, config)
    canny_segments = extract_hough_segments(canny_thin, config)
    height, width = image_bgr.shape[:2]
    primary_vanishing_point = estimate_vanishing_point_ransac(
        primary_segments,
        image_width=width,
        image_height=height,
        config=config,
    )
    canny_vanishing_point = estimate_vanishing_point_ransac(
        canny_segments,
        image_width=width,
        image_height=height,
        config=config,
    )

    edge_probability_u8 = np.round(
        _robust_unit_interval(edge_result.probability) * 255.0
    ).astype(np.uint8)
    cv2.imwrite(str(output_dir / "cnn-edge-probability.png"), edge_probability_u8)
    cv2.imwrite(str(output_dir / "cnn-edge-thinned.png"), primary_thin)
    cv2.imwrite(str(output_dir / "canny-control.png"), canny_thin)
    cv2.imwrite(str(output_dir / "working-image.png"), image_bgr)
    cv2.imwrite(
        str(output_dir / "cnn-lines-vanishing-point.png"),
        draw_geometry_overlay(image_bgr, primary_segments, primary_vanishing_point),
    )
    cv2.imwrite(
        str(output_dir / "canny-lines-vanishing-point.png"),
        draw_geometry_overlay(image_bgr, canny_segments, canny_vanishing_point),
    )
    _write_segments_csv(
        output_dir / "line-segments.csv",
        primary_segments,
        canny_segments,
    )

    comparison_distance: float | None = None
    if primary_vanishing_point is not None and canny_vanishing_point is not None:
        comparison_distance = math.hypot(
            primary_vanishing_point.x - canny_vanishing_point.x,
            primary_vanishing_point.y - canny_vanishing_point.y,
        ) / math.hypot(width, height)

    summary: dict[str, object] = {
        "analysis": "CNN edge to Hough lines and RANSAC vanishing point",
        "format_version": FORMAT_VERSION,
        "image_path": project_relative_path(resolved_path),
        "image_sha256": sha256_file(resolved_path),
        "image_stem": _safe_image_stem(resolved_path),
        "original_size": [original_size[0], original_size[1]],
        "working_size": [width, height],
        "backend_requested": backend,
        "backend_used": edge_result.backend_used,
        "backend_detail": edge_result.backend_detail,
        "fallback_reason": edge_result.fallback_reason,
        "hed_sources": {
            "model_url": HED_MODEL_URL,
            "model_sha256": HED_MODEL_SHA256,
            "prototxt_url": HED_PROTOTXT_URL,
            "prototxt_sha256": HED_PROTOTXT_SHA256,
        },
        "primary_geometry": {
            "estimate_label": "RANSAC dominant line-intersection candidate",
            "human_validation_required": True,
            "vanishing_roi_normalized": (
                list(config.vanishing_roi) if config.vanishing_roi is not None else None
            ),
            "line_count": len(primary_segments),
            "orientation_histogram": orientation_histogram(primary_segments),
            "vanishing_point": (
                asdict(primary_vanishing_point)
                if primary_vanishing_point is not None
                else None
            ),
            "evidence_grade": estimate_evidence_grade(primary_vanishing_point),
        },
        "canny_control": {
            "estimate_label": "RANSAC dominant line-intersection candidate",
            "human_validation_required": True,
            "vanishing_roi_normalized": (
                list(config.vanishing_roi) if config.vanishing_roi is not None else None
            ),
            "line_count": len(canny_segments),
            "orientation_histogram": orientation_histogram(canny_segments),
            "vanishing_point": (
                asdict(canny_vanishing_point)
                if canny_vanishing_point is not None
                else None
            ),
            "evidence_grade": estimate_evidence_grade(canny_vanishing_point),
        },
        "cnn_canny_vanishing_point_distance_normalized": comparison_distance,
        "config": asdict(config),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "interpretation_limits": [
            "HED was trained on natural-image boundaries, so painting-domain errors are expected.",
            "Hough detects strong straight evidence but does not understand doors, mirrors, or depicted space.",
            "The highest-consensus line intersection is an estimated pictorial vanishing point, not manually verified ground truth.",
            "An optional ROI may encode an independently justified art-historical prior; the ROI must be reported and cannot be presented as an automatic discovery.",
            "Agreement with Canny is a robustness check, not proof that either method is correct.",
        ],
    }
    _write_json(output_dir / "geometry-analysis.json", summary)
    return summary


def parse_vanishing_roi(value: str) -> tuple[float, float, float, float]:
    """解析归一化 ROI，并在进入分析前给出明确参数错误。"""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "vanishing-roi 必须包含四个逗号分隔数值：x_min,y_min,x_max,y_max。"
        )
    try:
        minimum_x, minimum_y, maximum_x, maximum_y = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("vanishing-roi 包含非数值内容。") from exc
    if not (
        0.0 <= minimum_x < maximum_x <= 1.0 and 0.0 <= minimum_y < maximum_y <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "vanishing-roi 必须位于 [0,1] 且满足最小值小于最大值。"
        )
    return minimum_x, minimum_y, maximum_x, maximum_y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 HED/VGG CNN 边缘、Hough 与 RANSAC 分析绘画几何。"
    )
    parser.add_argument("--image", type=Path, required=True, help="待分析图像。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/geometry-analysis"),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "hed", "vgg", "canny"),
        default="auto",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="仅 VGG 后端使用；HED 由 OpenCV DNN 在 CPU 执行。",
    )
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--hough-threshold", type=int, default=45)
    parser.add_argument("--min-line-length-ratio", type=float, default=0.055)
    parser.add_argument("--max-line-gap-ratio", type=float, default=0.012)
    parser.add_argument("--max-lines", type=int, default=400)
    parser.add_argument("--ransac-iterations", type=int, default=5000)
    parser.add_argument("--inlier-distance-ratio", type=float, default=0.012)
    parser.add_argument("--minimum-pair-angle-degrees", type=float, default=5.0)
    parser.add_argument("--intersection-bound-ratio", type=float, default=2.0)
    parser.add_argument(
        "--vanishing-roi",
        type=parse_vanishing_roi,
        help=(
            "可选的归一化候选区域 x_min,y_min,x_max,y_max；仅应依据独立人工透视图使用。"
        ),
    )
    parser.add_argument("--seed", type=int, default=139)
    parser.add_argument("--max-non-jpeg-pixels", type=int, default=100_000_000)
    parser.add_argument(
        "--hed-cache-dir",
        type=Path,
        default=default_hed_cache_dir(),
    )
    parser.add_argument("--hed-prototxt", type=Path)
    parser.add_argument("--hed-model", type=Path)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="禁止自动下载缺少的 HED 原始资源。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = GeometryConfig(
        max_side=arguments.max_side,
        hough_threshold=arguments.hough_threshold,
        min_line_length_ratio=arguments.min_line_length_ratio,
        max_line_gap_ratio=arguments.max_line_gap_ratio,
        max_lines=arguments.max_lines,
        ransac_iterations=arguments.ransac_iterations,
        inlier_distance_ratio=arguments.inlier_distance_ratio,
        minimum_pair_angle_degrees=arguments.minimum_pair_angle_degrees,
        intersection_bound_ratio=arguments.intersection_bound_ratio,
        vanishing_roi=arguments.vanishing_roi,
        seed=arguments.seed,
        max_non_jpeg_pixels=arguments.max_non_jpeg_pixels,
    )
    try:
        summary = analyze_geometry(
            image_path=arguments.image,
            output_dir=arguments.output_dir,
            backend=arguments.backend,
            device=arguments.device,
            config=config,
            hed_cache_dir=arguments.hed_cache_dir,
            allow_download=not arguments.no_download,
            prototxt_override=arguments.hed_prototxt,
            model_override=arguments.hed_model,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(
                    (arguments.output_dir / "geometry-analysis.json").resolve()
                ),
                "backend_used": summary["backend_used"],
                "primary_line_count": summary["primary_geometry"]["line_count"],
                "primary_vanishing_point": summary["primary_geometry"][
                    "vanishing_point"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
