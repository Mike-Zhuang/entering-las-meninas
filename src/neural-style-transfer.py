#!/usr/bin/env python3
"""使用 Gatys/VGG19 方法生成可复现的《宫娥》神经风格迁移序列。

本模块刻意把“style”限定为 VGG 特征通道的 Gram 相关性，而不是完整的艺术史
风格。内容图与风格图均保持原始纵横比缩放，不进行中心裁剪。每个 style strength
从同一个确定性初始图独立优化，便于把强度作为受控变量比较。

默认内容图是项目中的公版高清《宫娥》，默认风格图是本项目原创的
``assets/style/cognitive-map-style-reference.png``。运行后会为每个强度保存 PNG、
逐步 loss CSV/JSON，并生成记录输入哈希、模型、设备、随机种子和全部参数的
``manifest.json``。
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
        "neural-style-transfer.py 需要 PyTorch。请在已安装 torch 与 torchvision 的环境中运行。"
    ) from exc


FORMAT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTENT_IMAGE = (
    PROJECT_ROOT / "pics" / "Las_Meninas,_by_Diego_Velázquez,_from_Prado_in_Google_Earth.jpg"
)
DEFAULT_STYLE_IMAGE = PROJECT_ROOT / "assets" / "style" / "cognitive-map-style-reference.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "neural-style-transfer"

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
    """优先记录项目相对路径，避免清单泄露本机用户名与目录。"""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)

# 索引与 torchvision.models.vgg19().features 一致。选择 ReLU 后的激活可避免
# 正负响应互相抵消，也与常见 Gatys 实现的层级定义保持一致。
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
    """一次完整风格迁移运行的可复现配置。"""

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
            raise FileNotFoundError(f"内容图不存在：{self.content_image}")
        if not self.style_image.is_file():
            raise FileNotFoundError(f"风格参考图不存在：{self.style_image}")
        if self.content_image.resolve() == self.style_image.resolve():
            raise ValueError("内容图与风格参考图必须是不同文件。")
        if not self.style_strengths:
            raise ValueError("style_strengths 不能为空。")
        for strength in self.style_strengths:
            if not math.isfinite(strength) or strength < 0.0:
                raise ValueError("每个 style strength 必须是有限的非负数。")
        if len(set(self.style_strengths)) != len(self.style_strengths):
            raise ValueError("style_strengths 不能包含重复值。")
        if self.long_side < 32:
            raise ValueError("long_side 必须至少为 32。")
        if self.steps < 1:
            raise ValueError("steps 必须至少为 1。")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须是有限的正数。")
        for field_name, value in (
            ("content_weight", self.content_weight),
            ("style_weight", self.style_weight),
            ("tv_weight", self.tv_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} 必须是有限的非负数。")
        if self.content_weight == self.style_weight == self.tv_weight == 0.0:
            raise ValueError("content/style/TV 权重不能同时为零。")
        if not math.isfinite(self.initial_noise) or not 0.0 <= self.initial_noise <= 1.0:
            raise ValueError("initial_noise 必须位于 0 与 1 之间。")
        if self.seed < 0 or self.seed >= 2**63:
            raise ValueError("seed 必须位于 0 与 2^63-1 之间。")
        if self.max_source_pixels < 1:
            raise ValueError("max_source_pixels 必须为正整数。")
        if self.progress_every < 1:
            raise ValueError("progress_every 必须为正整数。")
        if self.content_layer not in VGG19_LAYER_INDICES:
            raise ValueError(f"未知 content layer：{self.content_layer}")
        if not self.style_layers:
            raise ValueError("style_layers 不能为空。")
        unknown_layers = sorted(set(self.style_layers) - set(VGG19_LAYER_INDICES))
        if unknown_layers:
            raise ValueError(f"未知 style layers：{', '.join(unknown_layers)}")


@dataclass(frozen=True)
class LoadedImage:
    """保持纵横比缩放后的图像及可审计尺寸记录。"""

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
    """某一优化状态的原始与加权 loss。"""

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
    """一个 style strength 对应的最终文件与 loss 记录。"""

    style_strength: float
    output_image: Path
    loss_csv: Path
    loss_json: Path
    output_sha256: str
    records: tuple[LossRecord, ...]


class VGG19FeatureExtractor(nn.Module):
    """只运行到最后一个所需层的 VGG19 特征提取器。"""

    def __init__(
        self,
        features: nn.Sequential,
        *,
        content_layer: str,
        style_layers: Sequence[str],
    ) -> None:
        super().__init__()
        if content_layer not in VGG19_LAYER_INDICES:
            raise ValueError(f"未知 content layer：{content_layer}")
        unknown_layers = sorted(set(style_layers) - set(VGG19_LAYER_INDICES))
        if unknown_layers:
            raise ValueError(f"未知 style layers：{', '.join(unknown_layers)}")
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
            raise ValueError("VGG 输入必须具有 [N, C, H, W] 形状。")
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
                "特征网络没有产生所需层；请确认网络与 VGG19 索引一致：" + ", ".join(missing)
            )
        return selected

    @torch.no_grad()
    def content_target(self, normalized_content: Tensor) -> Tensor:
        return self(normalized_content)[self.content_layer].detach()

    @torch.no_grad()
    def style_targets(self, normalized_style: Tensor) -> dict[str, Tensor]:
        """逐层计算 Gram target，避免长期保留高分辨率浅层激活。"""

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
            raise RuntimeError("无法生成 style targets：" + ", ".join(missing))
        return targets


def parse_style_strengths(value: str | Sequence[float]) -> tuple[float, ...]:
    """解析、去重并按升序返回 style strength。"""

    raw_values: Sequence[object]
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
        if any(part == "" for part in raw_values):
            raise ValueError("style strengths 必须是逗号分隔的数字。")
    else:
        raw_values = value
    try:
        strengths = tuple(float(raw_value) for raw_value in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("style strengths 必须是逗号分隔的数字。") from exc
    if not strengths:
        raise ValueError("style strengths 至少需要一个值。")
    if any(not math.isfinite(strength) or strength < 0.0 for strength in strengths):
        raise ValueError("style strengths 必须是有限的非负数。")
    if len(set(strengths)) != len(strengths):
        raise ValueError("style strengths 不能包含重复值。")
    return tuple(sorted(strengths))


def choose_device(requested: DeviceChoice) -> torch.device:
    """自动优先选择 Apple MPS，否则回退到 CPU。"""

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return torch.device("mps" if mps_available else "cpu")
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("请求了 MPS，但当前 PyTorch 环境不可用。")
        return torch.device("mps")
    return torch.device("cpu")


def configure_reproducibility(seed: int) -> None:
    """固定 Python、NumPy 与 Torch 随机源，并请求确定性算法。"""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    # MPS 的少数算子可能没有完全确定性的实现。warn_only 可避免运行中断，同时
    # manifest 会记录设备；同设备复现实验时仍使用相同随机源与优化顺序。
    torch.use_deterministic_algorithms(True, warn_only=True)


def gram_matrix(features: Tensor) -> Tensor:
    """计算按 C×H×W 归一化的批量 Gram matrix。"""

    if features.ndim != 4:
        raise ValueError("features 必须具有 [N, C, H, W] 形状。")
    batch_size, channels, height, width = features.shape
    flattened = features.reshape(batch_size, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / float(channels * height * width)


def total_variation_loss(image_tensor: Tensor) -> Tensor:
    """计算归一化 L1 total variation，抑制高频像素噪声。"""

    if image_tensor.ndim != 4:
        raise ValueError("image_tensor 必须具有 [N, C, H, W] 形状。")
    if image_tensor.shape[-2] < 2 or image_tensor.shape[-1] < 2:
        raise ValueError("total variation 至少需要 2×2 的图像。")
    vertical = torch.mean(torch.abs(image_tensor[:, :, 1:, :] - image_tensor[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :-1]))
    return vertical + horizontal


def normalize_for_vgg(image_tensor: Tensor) -> Tensor:
    """按 Torchvision VGG19 ImageNet 权重要求归一化 RGB 张量。"""

    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError("image_tensor 必须具有 [N, 3, H, W] 形状。")
    mean = image_tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image_tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image_tensor - mean) / std


def image_to_tensor(image: Image.Image, device: torch.device) -> Tensor:
    """把 RGB PIL 图像转换为 [1, 3, H, W]、0–1 float32 张量。"""

    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def tensor_to_image(image_tensor: Tensor) -> Image.Image:
    """把 0–1 图像张量转换为无损 RGB PNG 可用的 PIL 图像。"""

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
        raise ValueError("image_tensor 必须具有 [1, 3, H, W] 形状。")
    array = image_tensor.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    uint8_array = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_array, mode="RGB")


def load_resized_rgb(
    image_path: str | Path,
    *,
    long_side: int,
    max_source_pixels: int,
) -> LoadedImage:
    """有界读取图像并保持纵横比缩放，绝不进行中心裁剪或 letterbox。

    原作 JPEG 约 7.8 亿像素，直接完整解码会占用数 GB 内存。Pillow 的 JPEG
    ``draft`` 可在解码阶段请求低分辨率版本；之后再用 Lanczos 精确缩放到工作尺寸。
    对非 JPEG，仍先验证源像素总数，避免无界内存使用。
    """

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在：{path}")
    if long_side < 32:
        raise ValueError("long_side 必须至少为 32。")
    if max_source_pixels < 1:
        raise ValueError("max_source_pixels 必须为正整数。")

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
                        f"源图共有 {source_pixels:,} 像素，超过上限 {max_source_pixels:,}。"
                    )
                scale = min(1.0, long_side / max(original_width, original_height))
                draft_size = (
                    max(1, round(original_width * scale)),
                    max(1, round(original_height * scale)),
                )
                source.draft("RGB", draft_size)
                oriented = ImageOps.exif_transpose(source)
                decoded_width, decoded_height = oriented.size
                # ``thumbnail`` 不会放大小图，会导致同一个 --long-side 对不同输入
                # 产生不一致的工作尺度。这里显式按最长边计算目标尺寸；JPEG draft
                # 只负责降低超大图的解码成本，不改变最终纵横比。
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
    # VGG19 有五次 2×2 pooling；短边不足 32 时最后几层会退化为零尺寸。
    if min(working_width, working_height) < 32:
        raise ValueError(
            "保持纵横比缩放后短边小于 32 像素；请增大 --long-side 或使用较不极端的图像。"
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
    """流式计算文件 SHA-256，避免把高清原图一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while True:
            block = file_handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def style_strength_slug(strength: float) -> str:
    """把强度转换为稳定且适合文件名的十进制标识。"""

    text = f"{strength:.8f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def create_vgg19_extractor(
    config: StyleTransferConfig,
    device: torch.device,
) -> tuple[VGG19FeatureExtractor, str]:
    """通过 Torchvision 官方 API 创建冻结的 VGG19 特征网络。"""

    try:
        import torchvision
        from torchvision.models import VGG19_Weights, vgg19
    except ImportError as exc:
        raise RuntimeError("未安装 torchvision，无法创建 VGG19。") from exc

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
    """在 CPU 上生成固定噪声后移至目标设备，保证各强度共享同一起点。"""

    if not 0.0 <= noise_amount <= 1.0:
        raise ValueError("noise_amount 必须位于 0 与 1 之间。")
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
    """计算未加权的 content、style 与 TV loss。"""

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
    """用 Adam 独立优化一个 style strength，并记录初始到最终的每一步 loss。"""

    if not math.isfinite(style_strength) or style_strength < 0.0:
        raise ValueError("style_strength 必须是有限的非负数。")
    generated = nn.Parameter(initial_tensor.detach().clone())
    # foreach=False 固定单张量更新路径，减少不同设备后端自动选择 foreach 实现
    # 带来的细微差异。
    optimizer = torch.optim.Adam(
        [generated],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    records: list[LossRecord] = []

    # step=0 是共同初始状态；每次更新后在下一次循环记录，因此 step=steps
    # 精确对应保存的最终图像，而不是最后一次 Adam 更新前的旧状态。
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
                f"strength={style_strength:g} 在 step={step} 出现非有限 loss；"
                "请降低 learning rate 或检查输入图。"
            )
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            generated.clamp_(0.0, 1.0)

    return generated.detach(), tuple(records)


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    """创建输出目录，并避免默认覆盖先前实验。"""

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"输出路径不是目录：{output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空：{output_dir}。请更换目录或显式使用 --overwrite。")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    """以稳定键顺序写入 UTF-8 JSON。"""

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
    """保存逐步 loss CSV 与包含语义说明的 JSON。"""

    if not records:
        raise ValueError("records 不能为空。")
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
    """保存某一强度的无损图像与 loss 日志。"""

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
    """构建不依赖未记录隐式默认值的参数 manifest。"""

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
    """运行全部强度并返回与磁盘 ``manifest.json`` 相同的字典。"""

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
        print(f"已完成 {len(results)} 个 style strengths：{config.output_dir}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用冻结的 VGG19、Gram style loss、content loss 与 TV loss 对《宫娥》进行 "
            "Gatys neural style transfer。输入保持纵横比，不进行中心裁剪。"
        )
    )
    parser.add_argument(
        "content_image",
        nargs="?",
        type=Path,
        default=DEFAULT_CONTENT_IMAGE,
        help=f"内容图路径（默认：{DEFAULT_CONTENT_IMAGE}）",
    )
    parser.add_argument(
        "--style-image",
        type=Path,
        default=DEFAULT_STYLE_IMAGE,
        help=f"风格参考图（默认使用项目原创图：{DEFAULT_STYLE_IMAGE}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--style-strengths",
        default=",".join(f"{value:g}" for value in DEFAULT_STYLE_STRENGTHS),
        help="逗号分隔的非负强度乘数，例如 0.25,0.5,1.0。",
    )
    parser.add_argument("--long-side", type=int, default=512, help="工作图最长边，默认 512。")
    parser.add_argument("--steps", type=int, default=500, help="每个强度的 Adam 更新次数。")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Adam 学习率。")
    parser.add_argument("--content-weight", type=float, default=1.0, help="content loss 权重。")
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1_000_000.0,
        help="style loss 基础权重；最终再乘 style strength。",
    )
    parser.add_argument("--tv-weight", type=float, default=0.0001, help="TV loss 权重。")
    parser.add_argument(
        "--initial-noise",
        type=float,
        default=0.02,
        help="加入内容初始图的确定性高斯噪声强度，范围 0–1。",
    )
    parser.add_argument("--seed", type=int, default=139, help="Python/NumPy/Torch 随机种子。")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="auto 优先使用 Apple MPS，不可用时回退 CPU。",
    )
    parser.add_argument(
        "--weights",
        choices=("default", "none"),
        default="default",
        help="default 使用 ImageNet VGG19；none 仅供离线诊断，不应作为正式分析结果。",
    )
    parser.add_argument(
        "--content-layer", default=DEFAULT_CONTENT_LAYER, help="VGG content layer。"
    )
    parser.add_argument(
        "--style-layers",
        default=",".join(DEFAULT_STYLE_LAYERS),
        help="逗号分隔的 VGG style layers。",
    )
    parser.add_argument(
        "--max-source-pixels",
        type=int,
        default=1_000_000_000,
        help="允许读取的源图像最大像素总数。",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="每隔多少步打印一次 loss。CSV/JSON 始终记录每一步。",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印优化进度。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中的同名产物；不会删除无关文件。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

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
