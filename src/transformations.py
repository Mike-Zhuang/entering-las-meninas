"""《宫娥》项目的受控图像变换与代理图生成工具。

本模块只实现可复现的传统图像处理基线，不会把任何结果冒充为 CNN 或神经风格
迁移。核心设计目标是一次只改变一个声明过的变量，从而让后续 CNN 特征分析可以
区分 style、geometry 与 relational topology。
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

DEFAULT_LEVELS: tuple[int, ...] = (0, 25, 50, 75, 100)
DEFAULT_PROXY_LONG_SIDE = 2048
DEFAULT_SOURCE_PIXEL_LIMIT = 1_000_000_000
SUPPORTED_OUTPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class NormalizedBox:
    """用 0–1 坐标记录可跨分辨率复用的矩形区域。"""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("NormalizedBox coordinates must be finite")
        if not (0.0 <= self.left < self.right <= 1.0):
            raise ValueError("NormalizedBox horizontal coordinates must satisfy 0 <= left < right <= 1")
        if not (0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError("NormalizedBox vertical coordinates must satisfy 0 <= top < bottom <= 1")

    def to_pixels(self, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        """将归一化坐标转换为 PIL 使用的半开像素边界。"""

        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive dimensions")
        left = min(width - 1, max(0, round(self.left * width)))
        top = min(height - 1, max(0, round(self.top * height)))
        right = min(width, max(left + 1, round(self.right * width)))
        bottom = min(height, max(top + 1, round(self.bottom * height)))
        return left, top, right, bottom


@dataclass(frozen=True)
class TransformationRecord:
    """记录一个输出图像的变换条件，方便后续生成可审计的 manifest。"""

    name: str
    family: str
    intensity_percent: int
    output_path: str
    parameters: Mapping[str, object]


# 这些固定 crop 来自高清原作的归一化坐标，不依赖具体代理图分辨率。
# mirror-content 只覆盖镜面内部，故删除镜像时会保留原有镜框。
FIXED_CROPS: Mapping[str, NormalizedBox] = {
    "whole-room": NormalizedBox(0.08, 0.20, 0.94, 0.79),
    "mirror": NormalizedBox(0.365, 0.495, 0.480, 0.650),
    "mirror-content": NormalizedBox(0.389, 0.522, 0.455, 0.619),
    "rear-door": NormalizedBox(0.500, 0.495, 0.645, 0.735),
    "painter": NormalizedBox(0.145, 0.475, 0.365, 0.790),
    "central-figures": NormalizedBox(0.245, 0.570, 0.790, 0.950),
    "canvas": NormalizedBox(0.000, 0.125, 0.205, 0.985),
}

MIRROR_CONTENT_BOX = FIXED_CROPS["mirror-content"]
# matched control 使用同一镜面区域，只改变对比度而不删除人物，因而同时控制位置与面积。
MIRROR_CONTROL_BOX = MIRROR_CONTENT_BOX


def _validate_amount(amount: float) -> float:
    value = float(amount)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("amount must be a finite number between 0 and 1")
    return value


def parse_levels(value: str | Iterable[int]) -> tuple[int, ...]:
    """解析并验证 0–100 的有序强度列表。"""

    raw_levels = value.split(",") if isinstance(value, str) else list(value)
    try:
        levels = tuple(int(str(level).strip()) for level in raw_levels)
    except ValueError as error:
        raise ValueError("levels must be comma-separated integers") from error
    if not levels:
        raise ValueError("levels must contain at least one value")
    if any(level < 0 or level > 100 for level in levels):
        raise ValueError("every level must be between 0 and 100")
    if len(set(levels)) != len(levels):
        raise ValueError("levels must not contain duplicates")
    return tuple(sorted(levels))


def _safe_open_for_proxy(
    input_path: str | Path,
    *,
    max_long_side: int,
    source_pixel_limit: int,
) -> Image.Image:
    """低内存读取超大 JPEG，并在解码阶段尽早请求缩略分辨率。

    原作约 7.8 亿像素，直接解码为数组会占用数 GB 内存。Pillow 的 ``draft``
    可以让 JPEG 解码器先按 1/2、1/4 或 1/8 缩放，再进行高质量 thumbnail。
    这里先显式限制允许的源像素总数，再临时调整 Pillow 的炸弹图像阈值；这样既
    能读取已知原作，又不会对任意无限大输入静默放行。
    """

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"input image does not exist: {path}")
    if max_long_side < 64:
        raise ValueError("max_long_side must be at least 64 pixels")
    if source_pixel_limit <= 0:
        raise ValueError("source_pixel_limit must be positive")

    previous_limit = Image.MAX_IMAGE_PIXELS
    # 先仅解析文件头取得尺寸，再由本函数自己的上限判断是否允许真正解码。若直接把
    # Pillow 阈值设成项目上限的一半，超限输入会在我们产生可解释报错前被 Pillow 拦截。
    Image.MAX_IMAGE_PIXELS = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                width, height = source.size
                if width * height > source_pixel_limit:
                    raise ValueError(
                        f"source image has {width * height:,} pixels, exceeding the "
                        f"configured limit of {source_pixel_limit:,}"
                    )
                scale = min(1.0, max_long_side / max(width, height))
                draft_size = (max(1, round(width * scale)), max(1, round(height * scale)))
                source.draft("RGB", draft_size)
                oriented = ImageOps.exif_transpose(source)
                oriented.thumbnail(
                    (max_long_side, max_long_side),
                    Image.Resampling.LANCZOS,
                    reducing_gap=3.0,
                )
                return oriented.convert("RGB").copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def _save_image(image: Image.Image, output_path: str | Path, *, quality: int = 95) -> Path:
    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        raise ValueError(f"unsupported output suffix {suffix!r}; choose one of: {supported}")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, quality=quality, subsampling=0, optimize=True)
    elif suffix == ".png":
        image.save(path, optimize=True)
    else:
        image.save(path, quality=quality)
    return path


def create_working_proxy(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    max_long_side: int = DEFAULT_PROXY_LONG_SIDE,
    source_pixel_limit: int = DEFAULT_SOURCE_PIXEL_LIMIT,
    quality: int = 95,
) -> Image.Image:
    """创建适合本地 CNN/可视化处理的规范 RGB 代理图。"""

    proxy = _safe_open_for_proxy(
        input_path,
        max_long_side=max_long_side,
        source_pixel_limit=source_pixel_limit,
    )
    if output_path is not None:
        _save_image(proxy, output_path, quality=quality)
    return proxy


def extract_fixed_crops(
    image: Image.Image,
    output_dir: str | Path,
    *,
    boxes: Mapping[str, NormalizedBox] = FIXED_CROPS,
    suffix: str = ".png",
) -> dict[str, Path]:
    """从同一代理图导出固定局部区域，避免各实验阶段手工漂移 crop。"""

    normalized_suffix = suffix.lower()
    if normalized_suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        raise ValueError(f"unsupported crop suffix: {suffix}")
    rgb_image = image.convert("RGB")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, box in boxes.items():
        crop = rgb_image.crop(box.to_pixels(rgb_image.size))
        path = directory / f"{name}{normalized_suffix}"
        paths[name] = _save_image(crop, path)
    return paths


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _as_image(array: np.ndarray) -> Image.Image:
    clipped = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def perspective_homography(
    image_size: tuple[int, int],
    amount: float,
    *,
    vanishing_point: tuple[float, float] = (0.56, 0.53),
    convergence: float = 0.14,
    depth_expansion: float = 0.08,
) -> np.ndarray:
    """生成围绕指定消失点改变透视纵深的 3×3 单应矩阵。"""

    amount = _validate_amount(amount)
    width, height = image_size
    if width < 2 or height < 2:
        raise ValueError("image_size must be at least 2 x 2")
    vx, vy = vanishing_point
    if not (0.0 <= vx <= 1.0 and 0.0 <= vy <= 1.0):
        raise ValueError("vanishing_point coordinates must be between 0 and 1")
    if not 0.0 <= convergence <= 0.35:
        raise ValueError("convergence must be between 0 and 0.35")
    if not 0.0 <= depth_expansion <= 0.25:
        raise ValueError("depth_expansion must be between 0 and 0.25")

    max_x = float(width - 1)
    max_y = float(height - 1)
    vanishing_x = vx * max_x
    vanishing_y = vy * max_y
    source = np.float32([[0.0, 0.0], [max_x, 0.0], [max_x, max_y], [0.0, max_y]])

    # 顶边向消失点收拢、底边远离消失点，得到连续且可解释的纵深增强轨迹。
    top_fraction = convergence * amount
    bottom_fraction = -depth_expansion * amount
    destination = np.float32(
        [
            [top_fraction * vanishing_x, top_fraction * vanishing_y],
            [max_x + top_fraction * (vanishing_x - max_x), top_fraction * vanishing_y],
            [
                max_x + bottom_fraction * (vanishing_x - max_x),
                max_y + bottom_fraction * (vanishing_y - max_y),
            ],
            [
                bottom_fraction * vanishing_x,
                max_y + bottom_fraction * (vanishing_y - max_y),
            ],
        ]
    )
    return cv2.getPerspectiveTransform(source, destination)


def perspective_depth_transform(
    image: Image.Image,
    amount: float,
    *,
    vanishing_point: tuple[float, float] = (0.56, 0.53),
    convergence: float = 0.14,
    depth_expansion: float = 0.08,
) -> Image.Image:
    """在保持画布尺寸的前提下进行确定性的透视/纵深变换。"""

    amount = _validate_amount(amount)
    rgb_image = image.convert("RGB")
    if amount == 0.0:
        return rgb_image.copy()
    width, height = rgb_image.size
    matrix = perspective_homography(
        rgb_image.size,
        amount,
        vanishing_point=vanishing_point,
        convergence=convergence,
        depth_expansion=depth_expansion,
    )
    warped = cv2.warpPerspective(
        _as_rgb_array(rgb_image),
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    # 单应变换把顶角向内收时会在边缘留下需要外推的楔形区域。这里裁取四边形内部
    # 最大的安全轴对齐区域并恢复原尺寸，避免反射填充制造并不存在于原作的重复门框。
    source_corners = np.float32(
        [[[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]]
    )
    destination_corners = cv2.perspectiveTransform(source_corners, matrix)[0]
    crop_left = max(0, math.ceil(max(destination_corners[0, 0], destination_corners[3, 0])))
    crop_top = max(0, math.ceil(max(destination_corners[0, 1], destination_corners[1, 1])))
    crop_right = min(
        width,
        math.floor(min(destination_corners[1, 0], destination_corners[2, 0])) + 1,
    )
    crop_bottom = min(
        height,
        math.floor(min(destination_corners[2, 1], destination_corners[3, 1])) + 1,
    )
    if crop_right - crop_left >= 8 and crop_bottom - crop_top >= 8:
        warped = cv2.resize(
            warped[crop_top:crop_bottom, crop_left:crop_right],
            (width, height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return _as_image(warped)


def deterministic_style_transform(image: Image.Image, amount: float) -> Image.Image:
    """只改变色彩与局部纹理的确定性非神经基线。

    此函数不会搬动像素坐标，也不使用 CNN。它组合色调映射、颜色矩阵、细节增强
    与轻度量化，目的是为后续真正的 neural style 结果提供可复现对照。
    """

    amount = _validate_amount(amount)
    rgb_image = image.convert("RGB")
    if amount == 0.0:
        return rgb_image.copy()

    original = _as_rgb_array(rgb_image).astype(np.float32) / 255.0
    luminance = np.tensordot(original, np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0]))
    chroma = original - luminance[..., None]

    # 略抬暗部并压缩高光，让结构更平面；这改变光度统计但不改变几何坐标。
    remapped_luminance = np.power(np.clip(luminance, 0.0, 1.0), 0.82)
    remapped_luminance = 0.92 * remapped_luminance + 0.08 * np.sqrt(remapped_luminance)
    styled = remapped_luminance[..., None] + 1.28 * chroma

    color_matrix = np.array(
        [
            [1.08, 0.025, -0.035],
            [-0.015, 0.96, 0.025],
            [0.025, -0.025, 0.88],
        ],
        dtype=np.float32,
    )
    styled = styled @ color_matrix.T
    tonal_position = (remapped_luminance - 0.5)[..., None]
    styled += tonal_position * np.array([0.055, 0.012, -0.050], dtype=np.float32)

    blurred = cv2.GaussianBlur(original, (0, 0), sigmaX=1.35, sigmaY=1.35)
    detail = original - blurred
    styled += 0.72 * detail
    styled = np.clip(styled, 0.0, 1.0)
    quantized = np.rint(styled * 31.0) / 31.0
    styled = 0.82 * styled + 0.18 * quantized

    result = (1.0 - amount) * original + amount * styled
    return _as_image(result * 255.0)


def combined_transform(
    image: Image.Image,
    *,
    geometry_amount: float,
    style_amount: float,
    vanishing_point: tuple[float, float] = (0.56, 0.53),
) -> Image.Image:
    """按固定顺序组合 geometry 与 style，供 2×2 矩阵右下角使用。"""

    geometry_result = perspective_depth_transform(
        image,
        geometry_amount,
        vanishing_point=vanishing_point,
    )
    return deterministic_style_transform(geometry_result, style_amount)


def _cosine_feather_mask(height: int, width: int, feather_fraction: float) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("mask dimensions must be positive")
    if not 0.0 <= feather_fraction <= 0.5:
        raise ValueError("feather_fraction must be between 0 and 0.5")
    if feather_fraction == 0.0:
        return np.ones((height, width), dtype=np.float32)

    feather = max(1.0, feather_fraction * min(height, width))
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1)).astype(np.float32)
    x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1)).astype(np.float32)
    distance = np.minimum(y[:, None], x[None, :])
    normalized = np.clip(distance / feather, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * normalized)


def _donor_region_above(
    array: np.ndarray,
    target_pixels: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = target_pixels
    target_height = bottom - top
    target_width = right - left
    gap = max(2, round(target_height * 0.12))
    donor_bottom = max(1, top - gap)
    donor_top = donor_bottom - target_height
    if donor_top < 0:
        donor_top = min(array.shape[0] - target_height, bottom + gap)
        donor_bottom = donor_top + target_height
    donor_left = min(max(0, left), max(0, array.shape[1] - target_width))
    donor_right = donor_left + target_width
    donor = array[donor_top:donor_bottom, donor_left:donor_right]
    if donor.size == 0:
        raise ValueError("unable to derive a donor region for the selected box")
    return cv2.resize(donor, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def _replace_box_with_local_texture(
    image: Image.Image,
    box: NormalizedBox,
    *,
    amount: float,
    feather_fraction: float,
) -> Image.Image:
    amount = _validate_amount(amount)
    rgb_image = image.convert("RGB")
    if amount == 0.0:
        return rgb_image.copy()

    array = _as_rgb_array(rgb_image).astype(np.float32)
    left, top, right, bottom = box.to_pixels(rgb_image.size)
    target = array[top:bottom, left:right]
    donor = _donor_region_above(array, (left, top, right, bottom)).astype(np.float32)
    donor = cv2.GaussianBlur(donor, (0, 0), sigmaX=0.8, sigmaY=0.8)

    # 只匹配目标边缘带的低阶统计，避免把镜中人物的结构重新带回填充区域。
    border_width = max(1, round(min(target.shape[:2]) * 0.12))
    border_mask = np.zeros(target.shape[:2], dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    target_border = target[border_mask]
    donor_pixels = donor.reshape(-1, 3)
    donor_mean = donor_pixels.mean(axis=0)
    donor_std = np.maximum(donor_pixels.std(axis=0), 2.0)
    border_mean = target_border.mean(axis=0)
    border_std = np.maximum(target_border.std(axis=0), 2.0)
    donor = (donor - donor_mean) * np.clip(border_std / donor_std, 0.55, 1.8) + border_mean

    feather = _cosine_feather_mask(target.shape[0], target.shape[1], feather_fraction)
    alpha = (amount * feather)[..., None]
    array[top:bottom, left:right] = (1.0 - alpha) * target + alpha * donor
    return _as_image(array)


def mirror_deletion(
    image: Image.Image,
    amount: float = 1.0,
    *,
    mirror_content_box: NormalizedBox = MIRROR_CONTENT_BOX,
    feather_fraction: float = 0.10,
) -> Image.Image:
    """删除镜面内部人物，同时保留镜框与其余构图。"""

    return _replace_box_with_local_texture(
        image,
        mirror_content_box,
        amount=amount,
        feather_fraction=feather_fraction,
    )


def matched_control(
    image: Image.Image,
    amount: float = 1.0,
    *,
    mirror_content_box: NormalizedBox = MIRROR_CONTENT_BOX,
    control_box: NormalizedBox = MIRROR_CONTROL_BOX,
    feather_fraction: float = 0.10,
) -> Image.Image:
    """在同一镜面区域施加同 RMS 的对比度变化，保留镜中人物关系。

    镜像关系、位置和面积保持不变；函数把控制区域的变化幅度缩放到与完整镜像删除
    一致，以减少“只是改了更多像素”或“改动位置不同”的混淆。
    """

    amount = _validate_amount(amount)
    rgb_image = image.convert("RGB")
    if amount == 0.0:
        return rgb_image.copy()

    original = _as_rgb_array(rgb_image).astype(np.float32)
    deletion = _as_rgb_array(
        mirror_deletion(
            rgb_image,
            1.0,
            mirror_content_box=mirror_content_box,
            feather_fraction=feather_fraction,
        )
    ).astype(np.float32)
    mirror_pixels = mirror_content_box.to_pixels(rgb_image.size)
    ml, mt, mr, mb = mirror_pixels
    target_delta = deletion[mt:mb, ml:mr] - original[mt:mb, ml:mr]
    target_rms = float(np.sqrt(np.mean(np.square(target_delta))))

    cl, ct, cr, cb = control_box.to_pixels(rgb_image.size)
    control_patch = original[ct:cb, cl:cr]
    # 对整个镜面施加统一暖色偏移：人物形状、位置与反射关系不变，只有颜色统计改变。
    # 相比二值化式对比度增强，连续色偏能在较高 RMS 下仍保持人物清晰可认。
    tint_direction = np.array([1.0, -0.35, 0.55], dtype=np.float32)
    feather = _cosine_feather_mask(control_patch.shape[0], control_patch.shape[1], feather_fraction)
    control_delta = feather[..., None] * tint_direction
    control_rms = float(np.sqrt(np.mean(np.square(control_delta))))
    if control_rms < 1e-6:
        raise ValueError("control region is too uniform to construct a matched edit")

    def attainable_rms(delta: np.ndarray, scale: float) -> float:
        candidate_patch = np.clip(control_patch + scale * delta, 0.0, 255.0)
        return float(np.sqrt(np.mean(np.square(candidate_patch - control_patch))))

    # 极端输入中，水平重排可能有大面积像素完全相同，任意放大都达不到目标 RMS。
    # 此时改用朝相反亮度端点移动的平滑方向，确保 matched control 在数学上可实现。
    if target_rms > 0.0 and attainable_rms(control_delta, 512.0) < target_rms:
        endpoint_direction = np.where(control_patch < 127.5, 1.0, -1.0)
        control_delta = endpoint_direction * feather[..., None]
        control_rms = float(np.sqrt(np.mean(np.square(control_delta))))

    # 因为 uint8 截断会让“目标 RMS / 当前 RMS”的一次比例换算失真，这里用二分搜索
    # 找到经过 clipping 后仍与镜像删除 RMS 匹配的比例。它让 matched control 真正控制
    # 改动能量，而不是只在描述中声称二者相近。
    low_scale = 0.0
    high_scale = max(1.0, target_rms / control_rms) if target_rms > 0.0 else 0.0

    def clipped_rms(scale: float) -> float:
        return attainable_rms(control_delta, scale)

    while target_rms > 0.0 and clipped_rms(high_scale) < target_rms and high_scale < 512.0:
        high_scale *= 2.0
    high_scale = min(high_scale, 512.0)
    for _ in range(24):
        middle_scale = 0.5 * (low_scale + high_scale)
        if clipped_rms(middle_scale) < target_rms:
            low_scale = middle_scale
        else:
            high_scale = middle_scale
    scale = 0.5 * (low_scale + high_scale)

    result = original.copy()
    fully_matched = np.clip(control_patch + scale * control_delta, 0.0, 255.0)
    result[ct:cb, cl:cr] = control_patch + amount * (fully_matched - control_patch)
    return _as_image(result)


def sham_warp(
    image: Image.Image,
    amount: float = 1.0,
    *,
    shift_pixels: float = 0.45,
) -> Image.Image:
    """进行“平移后逆平移”的假变换，控制两次重采样造成的轻微模糊。"""

    amount = _validate_amount(amount)
    if shift_pixels < 0.0 or not math.isfinite(shift_pixels):
        raise ValueError("shift_pixels must be a finite non-negative value")
    rgb_image = image.convert("RGB")
    if amount == 0.0 or shift_pixels == 0.0:
        return rgb_image.copy()

    array = _as_rgb_array(rgb_image)
    height, width = array.shape[:2]
    shift = shift_pixels * amount
    forward = np.float32([[1.0, 0.0, shift], [0.0, 1.0, -0.5 * shift]])
    backward = np.float32([[1.0, 0.0, -shift], [0.0, 1.0, 0.5 * shift]])
    moved = cv2.warpAffine(
        array,
        forward,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    restored = cv2.warpAffine(
        moved,
        backward,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return _as_image(restored)


def generate_geometry_sequence(
    image: Image.Image,
    levels: Sequence[int] = DEFAULT_LEVELS,
    *,
    vanishing_point: tuple[float, float] = (0.56, 0.53),
) -> dict[int, Image.Image]:
    """生成 0/25/50/75/100% 等连续几何变换序列。"""

    validated_levels = parse_levels(levels)
    return {
        level: perspective_depth_transform(
            image,
            level / 100.0,
            vanishing_point=vanishing_point,
        )
        for level in validated_levels
    }


def generate_style_sequence(
    image: Image.Image,
    levels: Sequence[int] = DEFAULT_LEVELS,
) -> dict[int, Image.Image]:
    """生成确定性色彩/纹理基线序列。"""

    validated_levels = parse_levels(levels)
    return {
        level: deterministic_style_transform(image, level / 100.0)
        for level in validated_levels
    }


def generate_topology_variants(image: Image.Image) -> dict[str, Image.Image]:
    """生成镜像删除、matched control 与 sham warp 三个预注册条件。"""

    rgb_image = image.convert("RGB")
    return {
        "topology-original": rgb_image.copy(),
        "topology-mirror-deletion": mirror_deletion(rgb_image),
        "topology-matched-control": matched_control(rgb_image),
        "topology-sham-warp": sham_warp(rgb_image),
    }


def _save_sequence(
    sequence: Mapping[int, Image.Image],
    output_dir: Path,
    *,
    prefix: str,
    suffix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for level, image in sequence.items():
        path = output_dir / f"{prefix}-{level:03d}{suffix}"
        paths.append(_save_image(image, path))
    return paths


def run_controlled_transformations(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    levels: Sequence[int] = DEFAULT_LEVELS,
    proxy_long_side: int = DEFAULT_PROXY_LONG_SIDE,
    vanishing_point: tuple[float, float] = (0.56, 0.53),
    suffix: str = ".png",
) -> dict[str, object]:
    """端到端生成代理图、固定 crops、三类变换与可审计 manifest。"""

    validated_levels = parse_levels(levels)
    normalized_suffix = suffix.lower()
    if normalized_suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        raise ValueError(f"unsupported output suffix: {suffix}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    proxy_path = directory / f"working-proxy{normalized_suffix}"
    proxy = create_working_proxy(
        input_path,
        proxy_path,
        max_long_side=proxy_long_side,
    )
    crop_paths = extract_fixed_crops(
        proxy,
        directory / "crops",
        suffix=normalized_suffix,
    )

    geometry_sequence = generate_geometry_sequence(
        proxy,
        validated_levels,
        vanishing_point=vanishing_point,
    )
    geometry_paths = _save_sequence(
        geometry_sequence,
        directory / "geometry",
        prefix="geometry",
        suffix=normalized_suffix,
    )

    style_sequence = generate_style_sequence(proxy, validated_levels)
    style_paths = _save_sequence(
        style_sequence,
        directory / "style",
        prefix="style-baseline",
        suffix=normalized_suffix,
    )

    topology_paths: dict[str, Path] = {}
    for name, variant in generate_topology_variants(proxy).items():
        topology_paths[name] = _save_image(
            variant,
            directory / "topology" / f"{name}{normalized_suffix}",
        )

    combined_path = _save_image(
        combined_transform(proxy, geometry_amount=1.0, style_amount=1.0),
        directory / f"combined-geometry-style-100{normalized_suffix}",
    )

    records: list[TransformationRecord] = []
    for level, path in zip(validated_levels, geometry_paths, strict=True):
        records.append(
            TransformationRecord(
                name=f"geometry-{level:03d}",
                family="geometry-only",
                intensity_percent=level,
                output_path=str(path),
                parameters={
                    "vanishing_point": list(vanishing_point),
                    "convergence": 0.14,
                    "depth_expansion": 0.08,
                    "safe_trim_and_resize": True,
                },
            )
        )
    for level, path in zip(validated_levels, style_paths, strict=True):
        records.append(
            TransformationRecord(
                name=f"style-baseline-{level:03d}",
                family="style-only-deterministic-non-neural",
                intensity_percent=level,
                output_path=str(path),
                parameters={"method": "color-tone-detail-baseline", "uses_cnn": False},
            )
        )
    topology_families = {
        "topology-original": "topology-original",
        "topology-mirror-deletion": "topology-breaking",
        "topology-matched-control": "matched-control",
        "topology-sham-warp": "resampling-control",
    }
    for name, path in topology_paths.items():
        records.append(
            TransformationRecord(
                name=name,
                family=topology_families[name],
                intensity_percent=0 if name == "topology-original" else 100,
                output_path=str(path),
                parameters={
                    "mirror_content_box": asdict(MIRROR_CONTENT_BOX),
                    "control_box": asdict(MIRROR_CONTROL_BOX),
                },
            )
        )
    records.append(
        TransformationRecord(
            name="combined-geometry-style-100",
            family="combined-deterministic-non-neural",
            intensity_percent=100,
            output_path=str(combined_path),
            parameters={
                "geometry_amount": 1.0,
                "style_amount": 1.0,
                "vanishing_point": list(vanishing_point),
                "uses_cnn": False,
            },
        )
    )

    manifest: dict[str, object] = {
        "source": str(Path(input_path)),
        "proxy": {
            "path": str(proxy_path),
            "width": proxy.width,
            "height": proxy.height,
            "max_long_side": proxy_long_side,
        },
        "fixed_crops": {name: str(path) for name, path in crop_paths.items()},
        "levels": list(validated_levels),
        "records": [asdict(record) for record in records],
        "matrix_inputs": {
            "original": str(proxy_path),
            "style_only": str(style_paths[-1]),
            "geometry_only": str(geometry_paths[-1]),
            "combined": str(combined_path),
        },
        "method_note": (
            "The style-only sequence is a deterministic color/texture baseline and is not "
            "neural style transfer. The combined deterministic condition is likewise a "
            "non-neural control."
        ),
    }
    manifest_path = directory / "transformations-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _add_common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path, help="输入原作或代理图路径")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument(
        "--proxy-long-side",
        type=int,
        default=DEFAULT_PROXY_LONG_SIDE,
        help=f"工作代理图最长边，默认 {DEFAULT_PROXY_LONG_SIDE}",
    )
    parser.add_argument("--suffix", default=".png", choices=sorted(SUPPORTED_OUTPUT_SUFFIXES))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成《宫娥》项目的受控 style、geometry 与 topology 变换",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    proxy_parser = subparsers.add_parser("proxy", help="生成规范代理图与固定 crops")
    _add_common_input_arguments(proxy_parser)

    geometry_parser = subparsers.add_parser("geometry", help="生成连续透视/纵深序列")
    _add_common_input_arguments(geometry_parser)
    geometry_parser.add_argument("--levels", default="0,25,50,75,100")
    geometry_parser.add_argument("--vp-x", type=float, default=0.56)
    geometry_parser.add_argument("--vp-y", type=float, default=0.53)

    style_parser = subparsers.add_parser("style", help="生成确定性非神经色彩/纹理基线")
    _add_common_input_arguments(style_parser)
    style_parser.add_argument("--levels", default="0,25,50,75,100")

    topology_parser = subparsers.add_parser("topology", help="生成镜像关系与控制条件")
    _add_common_input_arguments(topology_parser)

    combined_parser = subparsers.add_parser("combined", help="生成 geometry+style 组合版本")
    _add_common_input_arguments(combined_parser)
    combined_parser.add_argument("--geometry-amount", type=float, default=1.0)
    combined_parser.add_argument("--style-amount", type=float, default=1.0)
    combined_parser.add_argument("--vp-x", type=float, default=0.56)
    combined_parser.add_argument("--vp-y", type=float, default=0.53)

    all_parser = subparsers.add_parser("all", help="一次生成全部受控变换和 manifest")
    _add_common_input_arguments(all_parser)
    all_parser.add_argument("--levels", default="0,25,50,75,100")
    all_parser.add_argument("--vp-x", type=float, default=0.56)
    all_parser.add_argument("--vp-y", type=float, default=0.53)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suffix = args.suffix.lower()
    output_dir: Path = args.output_dir

    if args.command == "all":
        manifest = run_controlled_transformations(
            args.input,
            output_dir,
            levels=parse_levels(args.levels),
            proxy_long_side=args.proxy_long_side,
            vanishing_point=(args.vp_x, args.vp_y),
            suffix=suffix,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = output_dir / f"working-proxy{suffix}"
    proxy = create_working_proxy(
        args.input,
        proxy_path,
        max_long_side=args.proxy_long_side,
    )

    output_paths: list[Path] = [proxy_path]
    if args.command == "proxy":
        output_paths.extend(extract_fixed_crops(proxy, output_dir / "crops", suffix=suffix).values())
    elif args.command == "geometry":
        sequence = generate_geometry_sequence(
            proxy,
            parse_levels(args.levels),
            vanishing_point=(args.vp_x, args.vp_y),
        )
        output_paths.extend(
            _save_sequence(sequence, output_dir / "geometry", prefix="geometry", suffix=suffix)
        )
    elif args.command == "style":
        sequence = generate_style_sequence(proxy, parse_levels(args.levels))
        output_paths.extend(
            _save_sequence(
                sequence,
                output_dir / "style",
                prefix="style-baseline",
                suffix=suffix,
            )
        )
    elif args.command == "topology":
        for name, variant in generate_topology_variants(proxy).items():
            output_paths.append(_save_image(variant, output_dir / "topology" / f"{name}{suffix}"))
    elif args.command == "combined":
        combined = combined_transform(
            proxy,
            geometry_amount=args.geometry_amount,
            style_amount=args.style_amount,
            vanishing_point=(args.vp_x, args.vp_y),
        )
        output_paths.append(_save_image(combined, output_dir / f"combined{suffix}"))
    else:
        raise AssertionError(f"unhandled command: {args.command}")

    print(json.dumps({"outputs": [str(path) for path in output_paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
