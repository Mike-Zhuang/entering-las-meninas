"""《宫娥》CNN 项目的统一制图与 2.5D 视差视频工具。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from .transformations import create_working_proxy
except ImportError:
    from transformations import create_working_proxy

ImageInput: TypeAlias = str | Path | Image.Image | np.ndarray
ArrayInput: TypeAlias = np.ndarray | Sequence[float] | Sequence[Sequence[float]]

FIGURE_BACKGROUND = "#f3efe6"
AXIS_TEXT = "#25231f"
GRID_COLOR = "#c9c1b4"
ACCENT_COLORS = ("#9d2d20", "#2f6173", "#a86f2a", "#5d4a78", "#477052")


def _ensure_output_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.suffix:
        raise ValueError("output_path must include a file suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resize_image(image: Image.Image, max_long_side: int) -> Image.Image:
    if max_long_side < 64:
        raise ValueError("max_long_side must be at least 64")
    result = image.convert("RGB").copy()
    result.thumbnail(
        (max_long_side, max_long_side),
        Image.Resampling.LANCZOS,
        reducing_gap=3.0,
    )
    return result


def load_visual_image(image: ImageInput, *, max_long_side: int = 2048) -> Image.Image:
    """把路径、PIL 图像或 NumPy 数组统一转换为安全尺寸的 RGB 图像。"""

    if isinstance(image, str | Path):
        return create_working_proxy(image, max_long_side=max_long_side)
    if isinstance(image, Image.Image):
        return _resize_image(image, max_long_side)

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in {1, 3, 4}:
        raise ValueError("image array must have shape HxW, HxWx1, HxWx3, or HxWx4")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("image array contains no finite values")
        if finite.min() >= 0.0 and finite.max() <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    pil_image = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return _resize_image(pil_image, max_long_side)


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(FIGURE_BACKGROUND)
    axis.tick_params(colors=AXIS_TEXT, labelsize=9)
    for spine in axis.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.8)


def _save_figure(figure: plt.Figure, output_path: str | Path, *, dpi: int = 180) -> Path:
    path = _ensure_output_parent(output_path)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


def plot_transformation_matrix(
    original: ImageInput,
    style_only: ImageInput,
    geometry_only: ImageInput,
    combined: ImageInput,
    output_path: str | Path,
    *,
    title: str = "Controlled Transformation Matrix",
    cell_titles: Sequence[str] = (
        "G0S0 · Original",
        "G0S1 · Style altered",
        "G1S0 · Geometry altered",
        "G1S1 · Geometry + style altered",
    ),
    max_long_side: int = 1600,
    dpi: int = 180,
) -> Path:
    """生成 geometry × style 的 2×2 实验矩阵。"""

    if len(cell_titles) != 4:
        raise ValueError("cell_titles must contain exactly four titles")
    images = [
        load_visual_image(original, max_long_side=max_long_side),
        load_visual_image(style_only, max_long_side=max_long_side),
        load_visual_image(geometry_only, max_long_side=max_long_side),
        load_visual_image(combined, max_long_side=max_long_side),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 13.5), facecolor=FIGURE_BACKGROUND)
    for index, (axis, image, cell_title) in enumerate(zip(axes.flat, images, cell_titles, strict=True)):
        axis.imshow(image)
        axis.set_title(cell_title, fontsize=12, color=AXIS_TEXT, pad=9, loc="left")
        axis.text(
            0.015,
            0.975,
            chr(ord("A") + index),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "square,pad=0.25", "facecolor": "#161512", "edgecolor": "none"},
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#b5ac9f")
            spine.set_linewidth(0.8)

    axes[0, 0].set_ylabel("Geometry retained", fontsize=12, color=AXIS_TEXT, labelpad=15)
    axes[1, 0].set_ylabel("Geometry altered", fontsize=12, color=AXIS_TEXT, labelpad=15)
    axes[0, 0].text(
        0.5,
        1.12,
        "Style retained",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=AXIS_TEXT,
    )
    axes[0, 1].text(
        0.5,
        1.12,
        "Style altered",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=AXIS_TEXT,
    )
    figure.suptitle(title, fontsize=20, fontweight="bold", color=AXIS_TEXT, y=0.995)
    figure.subplots_adjust(wspace=0.035, hspace=0.12, top=0.92)
    return _save_figure(figure, output_path, dpi=dpi)


def plot_image_comparison(
    images: Mapping[str, ImageInput],
    output_path: str | Path,
    *,
    columns: int = 3,
    title: str | None = None,
    captions: Mapping[str, str] | None = None,
    max_long_side: int = 1400,
    dpi: int = 180,
) -> Path:
    """按传入顺序生成任意数量的图像对比板。"""

    if not images:
        raise ValueError("images must not be empty")
    if columns <= 0:
        raise ValueError("columns must be positive")
    names = list(images)
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 5.7 * rows),
        squeeze=False,
        facecolor=FIGURE_BACKGROUND,
    )
    for index, name in enumerate(names):
        axis = axes.flat[index]
        axis.imshow(load_visual_image(images[name], max_long_side=max_long_side))
        axis.set_title(name, loc="left", fontsize=12, fontweight="bold", color=AXIS_TEXT, pad=8)
        if captions is not None and name in captions:
            axis.text(
                0.0,
                -0.035,
                captions[name],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#504c45",
                wrap=True,
            )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#b5ac9f")
            spine.set_linewidth(0.8)
    for index in range(len(names), rows * columns):
        axes.flat[index].axis("off")
    if title:
        figure.suptitle(title, fontsize=19, fontweight="bold", color=AXIS_TEXT, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97 if title else 1.0))
    return _save_figure(figure, output_path, dpi=dpi)


def _reduce_feature_map(feature_map: np.ndarray) -> np.ndarray:
    array = np.asarray(feature_map, dtype=np.float64)
    if array.ndim == 2:
        reduced = array
    elif array.ndim == 3:
        if array.shape[-1] in {1, 3, 4}:
            reduced = np.mean(array, axis=-1)
        else:
            reduced = np.mean(array, axis=0)
    elif array.ndim == 4 and array.shape[0] == 1:
        reduced = _reduce_feature_map(array[0])
    else:
        raise ValueError("feature maps must be 2D, 3D, or batched with a leading size of 1")
    return np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)


def _robust_limits(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [2.0, 98.0])
    if math.isclose(float(low), float(high), rel_tol=0.0, abs_tol=1e-12):
        center = float(low)
        return center - 0.5, center + 0.5
    return float(low), float(high)


def plot_feature_map_grid(
    feature_maps: Mapping[str, Mapping[str, np.ndarray]],
    output_path: str | Path,
    *,
    title: str = "CNN Feature Maps",
    cmap: str = "magma",
    normalization: str = "per-map",
    dpi: int = 180,
) -> Path:
    """绘制 image × layer 特征图网格。

    ``normalization`` 可取 ``per-map``、``global`` 或 ``none``。逐图归一化适合看结构，
    全局归一化适合比较响应强度；图注应明确实际选择，避免把两者混为一谈。
    """

    if not feature_maps:
        raise ValueError("feature_maps must not be empty")
    if normalization not in {"per-map", "global", "none"}:
        raise ValueError("normalization must be 'per-map', 'global', or 'none'")
    row_labels = list(feature_maps)
    column_labels = list(feature_maps[row_labels[0]])
    if not column_labels:
        raise ValueError("each feature map row must contain at least one layer")
    for row_label, row in feature_maps.items():
        if list(row) != column_labels:
            raise ValueError(f"row {row_label!r} does not use the same ordered layer names")

    reduced_maps = {
        row_label: {column: _reduce_feature_map(feature_maps[row_label][column]) for column in column_labels}
        for row_label in row_labels
    }
    global_limits: tuple[float, float] | None = None
    if normalization == "global":
        flattened = np.concatenate(
            [reduced_maps[row][column].reshape(-1) for row in row_labels for column in column_labels]
        )
        global_limits = _robust_limits(flattened)

    figure, axes = plt.subplots(
        len(row_labels),
        len(column_labels),
        figsize=(3.8 * len(column_labels), 3.6 * len(row_labels)),
        squeeze=False,
        facecolor=FIGURE_BACKGROUND,
    )
    last_image = None
    for row_index, row_label in enumerate(row_labels):
        for column_index, column_label in enumerate(column_labels):
            axis = axes[row_index, column_index]
            array = reduced_maps[row_label][column_label]
            if normalization == "per-map":
                limits = _robust_limits(array)
            elif normalization == "global":
                if global_limits is None:
                    raise AssertionError("global limits were not calculated")
                limits = global_limits
            else:
                limits = (float(np.min(array)), float(np.max(array)))
                if math.isclose(limits[0], limits[1], rel_tol=0.0, abs_tol=1e-12):
                    limits = (limits[0] - 0.5, limits[1] + 0.5)
            last_image = axis.imshow(array, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(column_label, fontsize=11, color=AXIS_TEXT, pad=7)
            if column_index == 0:
                axis.set_ylabel(row_label, fontsize=11, color=AXIS_TEXT, labelpad=8)
            for spine in axis.spines.values():
                spine.set_color("#b5ac9f")
                spine.set_linewidth(0.7)
    if normalization == "global" and last_image is not None:
        figure.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.78, pad=0.018, label="Activation")
    figure.suptitle(title, fontsize=18, fontweight="bold", color=AXIS_TEXT, y=0.995)
    figure.text(
        0.995,
        0.005,
        f"Normalization: {normalization}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#5c574f",
    )
    # colorbar 会创建不参与 tight_layout 的额外 Axes；显式边距可避免警告并稳定版式。
    figure.subplots_adjust(left=0.08, right=0.91, bottom=0.06, top=0.92, wspace=0.08, hspace=0.12)
    return _save_figure(figure, output_path, dpi=dpi)


def plot_distance_heatmap(
    values: ArrayInput,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "Representation Distance by Layer and Image",
    colorbar_label: str = "Distance",
    cmap: str = "magma",
    annotate: bool = True,
    dpi: int = 180,
) -> Path:
    """绘制 layer × image 或任意二维指标距离热图。"""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    if matrix.shape != (len(row_labels), len(column_labels)):
        raise ValueError("matrix shape must match row_labels and column_labels")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must contain only finite numbers")

    width = max(6.5, 1.15 * len(column_labels) + 2.2)
    height = max(4.8, 0.72 * len(row_labels) + 2.2)
    figure, axis = plt.subplots(figsize=(width, height), facecolor=FIGURE_BACKGROUND)
    image = axis.imshow(matrix, cmap=cmap, aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.tick_params(axis="x", rotation=35)
    _style_axis(axis)
    axis.set_title(title, fontsize=16, fontweight="bold", color=AXIS_TEXT, pad=14)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.87, pad=0.025)
    colorbar.set_label(colorbar_label, color=AXIS_TEXT)
    if annotate:
        threshold = float(np.min(matrix) + 0.55 * (np.max(matrix) - np.min(matrix)))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text_color = "white" if value >= threshold else "#181713"
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
    figure.tight_layout()
    return _save_figure(figure, output_path, dpi=dpi)


def plot_metric_curves(
    intensities: Sequence[float],
    metrics: Mapping[str, Sequence[float]],
    output_path: str | Path,
    *,
    title: str = "Metric Trajectories Across Transformation Strength",
    x_label: str = "Transformation strength (%)",
    y_label: str = "Normalized distance",
    dpi: int = 180,
) -> Path:
    """绘制多条 feature/metric 强度轨迹。"""

    x = np.asarray(intensities, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("intensities must be a non-empty finite one-dimensional sequence")
    if not metrics:
        raise ValueError("metrics must not be empty")

    figure, axis = plt.subplots(figsize=(9.0, 5.5), facecolor=FIGURE_BACKGROUND)
    for index, (name, values) in enumerate(metrics.items()):
        y = np.asarray(values, dtype=np.float64)
        if y.shape != x.shape or not np.all(np.isfinite(y)):
            raise ValueError(f"metric {name!r} must contain one finite value per intensity")
        axis.plot(
            x,
            y,
            marker="o",
            markersize=5.5,
            linewidth=2.1,
            label=name,
            color=ACCENT_COLORS[index % len(ACCENT_COLORS)],
        )
    _style_axis(axis)
    axis.grid(True, color=GRID_COLOR, alpha=0.55, linewidth=0.7)
    axis.set_xlabel(x_label, fontsize=11, color=AXIS_TEXT)
    axis.set_ylabel(y_label, fontsize=11, color=AXIS_TEXT)
    axis.set_title(title, fontsize=16, fontweight="bold", color=AXIS_TEXT, pad=13)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    return _save_figure(figure, output_path, dpi=dpi)


def plot_feature_space(
    points: ArrayInput,
    labels: Sequence[str],
    output_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    connect_order: bool = False,
    title: str = "Style–Geometry Feature Space",
    x_label: str = "Component 1",
    y_label: str = "Component 2",
    dpi: int = 180,
) -> Path:
    """绘制 PCA/UMAP 或直接 style-distance × geometry-distance 二维点图。"""

    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must have shape N x 2")
    if coordinates.shape[0] != len(labels):
        raise ValueError("labels length must match number of points")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("points must contain only finite values")
    if groups is not None and len(groups) != len(labels):
        raise ValueError("groups length must match labels length")

    group_values = list(groups) if groups is not None else ["all"] * len(labels)
    ordered_groups = list(dict.fromkeys(group_values))
    figure, axis = plt.subplots(figsize=(8.0, 6.4), facecolor=FIGURE_BACKGROUND)
    for group_index, group in enumerate(ordered_groups):
        indices = [index for index, value in enumerate(group_values) if value == group]
        subset = coordinates[indices]
        axis.scatter(
            subset[:, 0],
            subset[:, 1],
            s=62,
            color=ACCENT_COLORS[group_index % len(ACCENT_COLORS)],
            label=group if groups is not None else None,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    if connect_order and len(coordinates) > 1:
        axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color="#655f56",
            linewidth=1.0,
            alpha=0.65,
            zorder=1,
        )
        for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
            axis.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={"arrowstyle": "->", "color": "#655f56", "lw": 1.0, "alpha": 0.65},
            )
    for point, label in zip(coordinates, labels, strict=True):
        axis.annotate(
            label,
            xy=point,
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
            color=AXIS_TEXT,
        )
    _style_axis(axis)
    axis.grid(True, color=GRID_COLOR, alpha=0.50, linewidth=0.7)
    axis.set_xlabel(x_label, fontsize=11, color=AXIS_TEXT)
    axis.set_ylabel(y_label, fontsize=11, color=AXIS_TEXT)
    axis.set_title(title, fontsize=16, fontweight="bold", color=AXIS_TEXT, pad=13)
    if groups is not None:
        axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    return _save_figure(figure, output_path, dpi=dpi)


def plot_style_geometry_scatter(
    style_distances: Sequence[float],
    geometry_distances: Sequence[float],
    labels: Sequence[str],
    output_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    title: str = "Style Distance vs Spatial-Content Distance",
    dpi: int = 180,
) -> Path:
    """为项目核心的 style × geometry 对照提供语义明确的便利封装。"""

    points = np.column_stack([style_distances, geometry_distances])
    return plot_feature_space(
        points,
        labels,
        output_path,
        groups=groups,
        title=title,
        x_label="CNN style distance",
        y_label="Spatial-content distance",
        dpi=dpi,
    )


def _load_depth_map(depth_map: ImageInput, target_size: tuple[int, int]) -> np.ndarray:
    if isinstance(depth_map, np.ndarray):
        array = np.asarray(depth_map, dtype=np.float32)
    else:
        image = load_visual_image(depth_map, max_long_side=max(target_size))
        array = np.asarray(image.convert("L"), dtype=np.float32)
    if array.ndim == 3:
        array = np.mean(array[..., :3], axis=-1)
    if array.ndim != 2:
        raise ValueError("depth_map must be a two-dimensional map or an image")
    width, height = target_size
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_CUBIC)


def normalize_depth_map(depth: np.ndarray, *, invert: bool = False) -> np.ndarray:
    """以 2–98 百分位稳健归一化深度，白色默认表示更靠近观看者。"""

    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("depth contains no finite values")
    low, high = np.percentile(finite, [2.0, 98.0])
    if math.isclose(float(low), float(high), rel_tol=0.0, abs_tol=1e-8):
        normalized = np.full(array.shape, 0.5, dtype=np.float32)
    else:
        normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32)


def heuristic_vertical_depth(image_size: tuple[int, int]) -> np.ndarray:
    """生成明确标注为 heuristic 的纵向深度代理，仅用于测试视频管线。"""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive values")
    vertical = np.linspace(0.12, 1.0, height, dtype=np.float32)[:, None]
    horizontal_center = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    center_weight = 1.0 - 0.10 * np.square(horizontal_center)
    return np.clip(vertical * center_weight, 0.0, 1.0)


def _parallax_frame(
    rgb_array: np.ndarray,
    depth: np.ndarray,
    *,
    camera_x: float,
    camera_y: float,
) -> np.ndarray:
    height, width = depth.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    # 所有平面同向移动，近景额外移动；减去中位数可让画面整体保持在中心附近。
    relative_depth = 0.25 + 0.75 * depth
    relative_depth -= float(np.median(relative_depth))
    map_x = grid_x - camera_x * relative_depth
    map_y = grid_y - camera_y * relative_depth
    return cv2.remap(
        rgb_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def create_parallax_video(
    image: ImageInput,
    output_path: str | Path,
    *,
    depth_map: ImageInput | None = None,
    allow_heuristic_depth: bool = False,
    invert_depth: bool = False,
    duration_seconds: float = 4.0,
    fps: int = 18,
    horizontal_amplitude: float = 18.0,
    vertical_amplitude: float = 3.0,
    max_long_side: int = 960,
) -> Path:
    """用深度图生成短循环 2.5D 视差 GIF、MP4、MOV 或 AVI。

    默认必须提供真实模型输出的 ``depth_map``。只有显式启用
    ``allow_heuristic_depth`` 时才会使用纵向渐变测试管线，避免把启发式结果写成 CNN
    深度估计。
    """

    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise ValueError("duration_seconds must be a finite positive value")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if horizontal_amplitude < 0.0 or vertical_amplitude < 0.0:
        raise ValueError("parallax amplitudes must be non-negative")
    if depth_map is None and not allow_heuristic_depth:
        raise ValueError(
            "depth_map is required unless allow_heuristic_depth=True; the heuristic is only a "
            "pipeline test and must not be described as CNN output"
        )

    rgb_image = load_visual_image(image, max_long_side=max_long_side)
    width, height = rgb_image.size
    # 常见 MP4 编码器要求偶数尺寸；统一处理后 GIF 与视频输出也能直接互换。
    even_width = width - width % 2
    even_height = height - height % 2
    if even_width < 2 or even_height < 2:
        raise ValueError("image is too small for video output")
    if (even_width, even_height) != rgb_image.size:
        rgb_image = rgb_image.crop((0, 0, even_width, even_height))
    width, height = rgb_image.size

    if depth_map is None:
        raw_depth = heuristic_vertical_depth((width, height))
    else:
        raw_depth = _load_depth_map(depth_map, (width, height))
    depth = normalize_depth_map(raw_depth, invert=invert_depth)
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.2, sigmaY=1.2)
    rgb_array = np.asarray(rgb_image, dtype=np.uint8)
    frame_count = max(2, round(duration_seconds * fps))
    output = _ensure_output_parent(output_path)
    suffix = output.suffix.lower()
    if suffix not in {".gif", ".mp4", ".mov", ".avi"}:
        raise ValueError("parallax output must end in .gif, .mp4, .mov, or .avi")

    def render_frame(frame_index: int) -> np.ndarray:
        phase = 2.0 * math.pi * frame_index / (frame_count - 1)
        camera_x = horizontal_amplitude * math.sin(phase)
        camera_y = vertical_amplitude * (1.0 - math.cos(phase))
        return _parallax_frame(
            rgb_array,
            depth,
            camera_x=camera_x,
            camera_y=camera_y,
        )

    if suffix == ".gif":
        frames = [Image.fromarray(render_frame(index), mode="RGB") for index in range(frame_count)]
        frame_duration_ms = max(1, round(1000 / fps))
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
    else:
        codec = "MJPG" if suffix == ".avi" else "mp4v"
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"unable to initialize the {codec} video writer for {output}")
        try:
            for index in range(frame_count):
                writer.write(cv2.cvtColor(render_frame(index), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"video writer produced no output: {output}")
    return output


def _parse_label_path(value: str, *, separator: str = "=") -> tuple[str, Path]:
    if separator not in value:
        raise argparse.ArgumentTypeError(f"expected LABEL{separator}PATH")
    label, raw_path = value.split(separator, 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"expected non-empty LABEL{separator}PATH")
    return label.strip(), Path(raw_path.strip())


def _parse_feature_map_spec(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected ROW:LAYER=PATH.npy")
    labels, raw_path = value.split("=", 1)
    row, layer = labels.split(":", 1)
    if not row.strip() or not layer.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty ROW:LAYER=PATH.npy")
    return row.strip(), layer.strip(), Path(raw_path.strip())


def _comma_labels(value: str) -> list[str]:
    labels = [label.strip() for label in value.split(",") if label.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("label list must not be empty")
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成《宫娥》项目的统一图表与 2.5D 视频")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix = subparsers.add_parser("matrix", help="生成 2×2 transformation matrix")
    matrix.add_argument("--original", required=True, type=Path)
    matrix.add_argument("--style-only", required=True, type=Path)
    matrix.add_argument("--geometry-only", required=True, type=Path)
    matrix.add_argument("--combined", required=True, type=Path)
    matrix.add_argument("--output", required=True, type=Path)
    matrix.add_argument("--title", default="Controlled Transformation Matrix")

    comparison = subparsers.add_parser("compare", help="生成通用图像对比板")
    comparison.add_argument(
        "--image",
        required=True,
        action="append",
        type=_parse_label_path,
        metavar="LABEL=PATH",
    )
    comparison.add_argument("--output", required=True, type=Path)
    comparison.add_argument("--columns", type=int, default=3)
    comparison.add_argument("--title")

    feature_grid = subparsers.add_parser("feature-grid", help="从 .npy 特征图生成网格")
    feature_grid.add_argument(
        "--map",
        required=True,
        action="append",
        type=_parse_feature_map_spec,
        metavar="ROW:LAYER=PATH.npy",
    )
    feature_grid.add_argument("--output", required=True, type=Path)
    feature_grid.add_argument("--title", default="CNN Feature Maps")
    feature_grid.add_argument("--normalization", choices=("per-map", "global", "none"), default="per-map")

    heatmap = subparsers.add_parser("heatmap", help="从 CSV 数值矩阵生成距离热图")
    heatmap.add_argument("--matrix", required=True, type=Path)
    heatmap.add_argument("--row-labels", required=True, type=_comma_labels)
    heatmap.add_argument("--column-labels", required=True, type=_comma_labels)
    heatmap.add_argument("--output", required=True, type=Path)
    heatmap.add_argument("--title", default="Representation Distance by Layer and Image")

    metrics = subparsers.add_parser("metrics", help="从带表头 CSV 生成指标轨迹")
    metrics.add_argument("--csv", required=True, type=Path)
    metrics.add_argument("--x-column", default="intensity")
    metrics.add_argument("--output", required=True, type=Path)
    metrics.add_argument("--title", default="Metric Trajectories Across Transformation Strength")

    feature_space = subparsers.add_parser("feature-space", help="从 CSV 生成二维特征空间")
    feature_space.add_argument("--csv", required=True, type=Path)
    feature_space.add_argument("--output", required=True, type=Path)
    feature_space.add_argument("--x-column", default="x")
    feature_space.add_argument("--y-column", default="y")
    feature_space.add_argument("--label-column", default="label")
    feature_space.add_argument("--group-column", default="group")
    feature_space.add_argument("--connect-order", action="store_true")
    feature_space.add_argument("--title", default="Style–Geometry Feature Space")

    parallax = subparsers.add_parser("parallax", help="生成短 2.5D 视差视频")
    parallax.add_argument("--image", required=True, type=Path)
    parallax.add_argument("--depth-map", type=Path)
    parallax.add_argument("--heuristic-depth", action="store_true")
    parallax.add_argument("--invert-depth", action="store_true")
    parallax.add_argument("--output", required=True, type=Path)
    parallax.add_argument("--duration", type=float, default=4.0)
    parallax.add_argument("--fps", type=int, default=18)
    parallax.add_argument("--horizontal-amplitude", type=float, default=18.0)
    parallax.add_argument("--vertical-amplitude", type=float, default=3.0)
    parallax.add_argument("--max-long-side", type=int, default=960)
    return parser


def _read_metric_csv(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("metric CSV must contain a header row")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(float(row[name]))
    return list(columns), columns


def _read_feature_space_csv(path: Path) -> dict[str, list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("feature-space CSV must contain a header row")
        columns: dict[str, list[str]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(row[name])
    return columns


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output: Path
    if args.command == "matrix":
        output = plot_transformation_matrix(
            args.original,
            args.style_only,
            args.geometry_only,
            args.combined,
            args.output,
            title=args.title,
        )
    elif args.command == "compare":
        output = plot_image_comparison(
            dict(args.image),
            args.output,
            columns=args.columns,
            title=args.title,
        )
    elif args.command == "feature-grid":
        maps: dict[str, dict[str, np.ndarray]] = {}
        for row, layer, path in args.map:
            maps.setdefault(row, {})[layer] = np.load(path)
        output = plot_feature_map_grid(
            maps,
            args.output,
            title=args.title,
            normalization=args.normalization,
        )
    elif args.command == "heatmap":
        values = np.loadtxt(args.matrix, delimiter=",")
        output = plot_distance_heatmap(
            values,
            args.row_labels,
            args.column_labels,
            args.output,
            title=args.title,
        )
    elif args.command == "metrics":
        fieldnames, columns = _read_metric_csv(args.csv)
        if args.x_column not in columns:
            raise ValueError(f"x column {args.x_column!r} is absent from {fieldnames}")
        x = columns.pop(args.x_column)
        output = plot_metric_curves(x, columns, args.output, title=args.title)
    elif args.command == "feature-space":
        columns = _read_feature_space_csv(args.csv)
        required = {args.x_column, args.y_column, args.label_column}
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"feature-space CSV is missing columns: {sorted(missing)}")
        points = np.column_stack(
            [
                np.asarray(columns[args.x_column], dtype=float),
                np.asarray(columns[args.y_column], dtype=float),
            ]
        )
        groups = columns.get(args.group_column)
        output = plot_feature_space(
            points,
            columns[args.label_column],
            args.output,
            groups=groups,
            connect_order=args.connect_order,
            title=args.title,
        )
    elif args.command == "parallax":
        output = create_parallax_video(
            args.image,
            args.output,
            depth_map=args.depth_map,
            allow_heuristic_depth=args.heuristic_depth,
            invert_depth=args.invert_depth,
            duration_seconds=args.duration,
            fps=args.fps,
            horizontal_amplitude=args.horizontal_amplitude,
            vertical_amplitude=args.vertical_amplitude,
            max_long_side=args.max_long_side,
        )
    else:
        raise AssertionError(f"unhandled command: {args.command}")

    print(json.dumps({"output": str(output)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
