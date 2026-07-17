"""CNN 风格/空间描述符与几何分析的可重复性测试。"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"无法加载测试模块：{module_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


cnn = load_script_module("las_meninas_cnn_analysis", "src/cnn-analysis.py")
geometry = load_script_module(
    "las_meninas_geometry_analysis", "src/geometry-analysis.py"
)


def test_gram_matrix_uses_declared_normalization() -> None:
    features = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    gram = cnn.gram_matrix(features)
    expected = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    torch.testing.assert_close(gram, expected)


def test_gram_matrix_rejects_non_batched_input() -> None:
    with pytest.raises(ValueError, match="N, C, H, W"):
        cnn.gram_matrix(torch.ones((3, 8, 8), dtype=torch.float32))


def test_analysis_config_rejects_unknown_layers() -> None:
    config = cnn.AnalysisConfig(style_layers=("relu1_1", "unknown_layer"))
    with pytest.raises(ValueError, match="未知层"):
        config.validate()


def test_toy_descriptor_extractor_outputs_all_three_representations() -> None:
    config = cnn.AnalysisConfig(
        image_size=32,
        spatial_size=4,
        style_layers=("relu1_1",),
        spatial_layers=("relu1_1",),
        activation_layers=("relu1_1",),
        weights="none",
    )
    features = nn.Sequential(nn.Identity(), nn.ReLU())
    extractor = cnn.VGGDescriptorExtractor(features, config, torch.device("cpu"))
    input_tensor = torch.randn(
        (1, 3, 32, 32), generator=torch.Generator().manual_seed(9)
    )
    descriptors = extractor.extract(input_tensor)
    assert descriptors.grams["relu1_1"].shape == (1, 3, 3)
    assert descriptors.spatial["relu1_1"].shape == (3, 4, 4)
    assert descriptors.activations["relu1_1"].shape == (32, 32)
    assert torch.all(descriptors.activations["relu1_1"] >= 0)


def test_load_and_letterbox_preserves_aspect_ratio(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.jpg"
    Image.new("RGB", (400, 200), color=(120, 90, 60)).save(image_path, quality=95)
    loaded = cnn.load_and_letterbox(image_path, image_size=128)
    assert loaded.original_width == 400
    assert loaded.original_height == 200
    assert loaded.resized_width == 128
    assert loaded.resized_height == 64
    assert loaded.offset_x == 0
    assert loaded.offset_y == 32
    assert loaded.square_rgb.size == (128, 128)


def test_non_jpeg_pixel_limit_is_enforced(tmp_path: Path) -> None:
    image_path = tmp_path / "bounded.png"
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(image_path)
    with pytest.raises(ValueError, match="非 JPEG 输入像素过大"):
        cnn.load_and_letterbox(
            image_path,
            image_size=32,
            max_non_jpeg_pixels=100,
        )


def sample_bundle(scale: float = 1.0):
    return cnn.FeatureBundle(
        metadata={"format_version": cnn.FORMAT_VERSION},
        grams={
            "relu1_1": np.asarray([[[1.0, 0.25], [0.25, 0.5]]], dtype=np.float32)
            * scale
        },
        spatial={
            "relu3_1": np.asarray(
                [
                    [[1.0, 0.0], [0.5, 0.25]],
                    [[0.2, 0.4], [0.8, 1.0]],
                ],
                dtype=np.float32,
            )
            * scale
        },
        activations={"relu1_1": np.ones((4, 4), dtype=np.float32) * scale},
    )


def test_feature_bundle_npz_roundtrip_without_pickle(tmp_path: Path) -> None:
    cache_path = tmp_path / "bundle.npz"
    original = sample_bundle()
    cnn.save_feature_bundle(cache_path, original)
    loaded = cnn.load_feature_bundle(cache_path)
    assert loaded.metadata["format_version"] == cnn.FORMAT_VERSION
    np.testing.assert_allclose(loaded.grams["relu1_1"], original.grams["relu1_1"])
    np.testing.assert_allclose(loaded.spatial["relu3_1"], original.spatial["relu3_1"])
    np.testing.assert_allclose(
        loaded.activations["relu1_1"], original.activations["relu1_1"]
    )


def test_style_and_spatial_distances_are_zero_for_identity() -> None:
    bundle = sample_bundle()
    style = cnn.compute_style_distances(bundle, bundle)
    spatial = cnn.compute_spatial_distances(bundle, bundle)
    assert style["aggregate"]["relative_frobenius"] == pytest.approx(0.0)
    assert spatial["aggregate"]["relative_rms"] == pytest.approx(0.0)
    assert spatial["aggregate"]["mean_cosine_distance"] == pytest.approx(0.0)


def test_style_and_spatial_distances_detect_scaled_features() -> None:
    reference = sample_bundle()
    candidate = sample_bundle(scale=1.5)
    style = cnn.compute_style_distances(reference, candidate)
    spatial = cnn.compute_spatial_distances(reference, candidate)
    assert style["aggregate"]["relative_frobenius"] == pytest.approx(0.5)
    assert spatial["aggregate"]["relative_rms"] == pytest.approx(0.5)
    assert spatial["aggregate"]["mean_cosine_distance"] == pytest.approx(0.0, abs=1e-12)


def test_normalize_heatmap_handles_non_finite_and_constant_values() -> None:
    constant = np.ones((8, 8), dtype=np.float32)
    assert np.count_nonzero(cnn.normalize_heatmap(constant)) == 0
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    values[0, 0] = np.nan
    normalized = cnn.normalize_heatmap(values)
    assert normalized.dtype == np.uint8
    assert normalized.shape == (8, 8)
    assert int(normalized.max()) == 255


def test_cpu_device_is_always_selectable() -> None:
    assert str(cnn.choose_device("cpu")) == "cpu"


def test_cache_key_changes_with_image_or_configuration() -> None:
    first_config = cnn.AnalysisConfig(image_size=128, spatial_size=8)
    second_config = cnn.AnalysisConfig(image_size=256, spatial_size=8)
    first = cnn.make_cache_key("a" * 64, first_config, "model")
    repeated = cnn.make_cache_key("a" * 64, first_config, "model")
    changed_image = cnn.make_cache_key("b" * 64, first_config, "model")
    changed_config = cnn.make_cache_key("a" * 64, second_config, "model")
    assert first == repeated
    assert len(first) == 64
    assert first != changed_image
    assert first != changed_config


def test_cache_hit_keeps_current_experimental_condition_path(tmp_path: Path) -> None:
    """同像素的两个 0% 条件共享张量，但清单仍应保留各自文件名。"""

    first_path = tmp_path / "style-baseline-000.png"
    second_path = tmp_path / "geometry-000.png"
    Image.new("RGB", (32, 32), color=(50, 70, 90)).save(first_path)
    second_path.write_bytes(first_path.read_bytes())
    config = cnn.AnalysisConfig(
        image_size=32,
        spatial_size=4,
        style_layers=("relu1_1",),
        spatial_layers=("relu1_1",),
        activation_layers=("relu1_1",),
        weights="none",
    )
    extractor = cnn.VGGDescriptorExtractor(
        nn.Sequential(nn.Identity(), nn.ReLU()),
        config,
        torch.device("cpu"),
    )
    cache_dir = tmp_path / "cache"
    first = cnn.extract_with_cache(
        first_path,
        extractor,
        config,
        "toy-model",
        cache_dir,
    )
    second = cnn.extract_with_cache(
        second_path,
        extractor,
        config,
        "toy-model",
        cache_dir,
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert Path(str(second.bundle.metadata["image_path"])).name == "geometry-000.png"
    assert str(second.bundle.metadata["image_id"]).startswith("geometry-000-")


def test_line_intersection_recovers_exact_point() -> None:
    first = geometry.LineSegment.from_endpoints(0.0, 0.0, 100.0, 100.0)
    second = geometry.LineSegment.from_endpoints(0.0, 100.0, 100.0, 0.0)
    point = geometry.intersect_infinite_lines(first, second, 5.0)
    assert point is not None
    assert point[0] == pytest.approx(50.0)
    assert point[1] == pytest.approx(50.0)


def test_parallel_line_intersection_is_rejected() -> None:
    first = geometry.LineSegment.from_endpoints(0.0, 0.0, 100.0, 0.0)
    second = geometry.LineSegment.from_endpoints(0.0, 10.0, 100.0, 10.0)
    assert geometry.intersect_infinite_lines(first, second, 5.0) is None


def make_segment_through_point(
    vanishing_x: float,
    vanishing_y: float,
    angle_degrees: float,
    near_radius: float,
    far_radius: float,
) -> object:
    angle_radians = math.radians(angle_degrees)
    direction_x = math.cos(angle_radians)
    direction_y = math.sin(angle_radians)
    return geometry.LineSegment.from_endpoints(
        vanishing_x + near_radius * direction_x,
        vanishing_y + near_radius * direction_y,
        vanishing_x + far_radius * direction_x,
        vanishing_y + far_radius * direction_y,
    )


def test_ransac_recovers_vanishing_point_with_outliers() -> None:
    expected_x, expected_y = 320.0, 180.0
    inliers = [
        make_segment_through_point(expected_x, expected_y, angle, 45.0, 280.0)
        for angle in (-72.0, -51.0, -31.0, -12.0, 13.0, 29.0, 48.0, 70.0)
    ]
    outliers = [
        geometry.LineSegment.from_endpoints(5.0, 15.0, 230.0, 28.0),
        geometry.LineSegment.from_endpoints(410.0, 40.0, 600.0, 55.0),
        geometry.LineSegment.from_endpoints(60.0, 330.0, 85.0, 80.0),
    ]
    config = geometry.GeometryConfig(
        ransac_iterations=1500,
        inlier_distance_ratio=0.005,
        seed=7,
    )
    estimate = geometry.estimate_vanishing_point_ransac(
        inliers + outliers,
        image_width=640,
        image_height=360,
        config=config,
    )
    assert estimate is not None
    assert estimate.x == pytest.approx(expected_x, abs=0.1)
    assert estimate.y == pytest.approx(expected_y, abs=0.1)
    assert estimate.inlier_count >= len(inliers)
    assert estimate.weighted_inlier_ratio > 0.7


def test_ransac_honors_independently_supplied_normalized_roi() -> None:
    first_cluster = [
        make_segment_through_point(160.0, 90.0, angle, 20.0, 130.0)
        for angle in (-65.0, -35.0, 5.0, 35.0, 65.0)
    ]
    second_cluster = [
        make_segment_through_point(480.0, 250.0, angle, 20.0, 160.0)
        for angle in (-70.0, -40.0, -10.0, 20.0, 50.0, 75.0)
    ]
    config = geometry.GeometryConfig(
        ransac_iterations=1200,
        inlier_distance_ratio=0.004,
        vanishing_roi=(0.10, 0.10, 0.40, 0.40),
        seed=19,
    )
    estimate = geometry.estimate_vanishing_point_ransac(
        first_cluster + second_cluster,
        image_width=640,
        image_height=360,
        config=config,
    )
    assert estimate is not None
    assert estimate.x == pytest.approx(160.0, abs=0.1)
    assert estimate.y == pytest.approx(90.0, abs=0.1)


def test_vanishing_roi_parser_rejects_reversed_bounds() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="最小值小于最大值"):
        geometry.parse_vanishing_roi("0.8,0.2,0.3,0.7")


def test_hough_extracts_long_synthetic_lines() -> None:
    edges = np.zeros((240, 320), dtype=np.uint8)
    cv2.line(edges, (10, 220), (160, 80), 255, 2, cv2.LINE_AA)
    cv2.line(edges, (310, 220), (160, 80), 255, 2, cv2.LINE_AA)
    cv2.line(edges, (20, 40), (300, 40), 255, 2, cv2.LINE_AA)
    config = geometry.GeometryConfig(
        max_side=320,
        hough_threshold=20,
        min_line_length_ratio=0.08,
        max_line_gap_ratio=0.02,
        max_lines=50,
    )
    segments = geometry.extract_hough_segments(edges, config)
    assert len(segments) >= 3
    assert segments[0].length > 150.0


def test_orientation_histogram_is_length_normalized() -> None:
    segments = [
        geometry.LineSegment.from_endpoints(0.0, 0.0, 100.0, 0.0),
        geometry.LineSegment.from_endpoints(0.0, 0.0, 0.0, 50.0),
    ]
    histogram = geometry.orientation_histogram(segments, bin_count=6)
    assert sum(item["proportion"] for item in histogram) == pytest.approx(1.0)
    assert sum(item["length_weight"] for item in histogram) == pytest.approx(150.0)


def test_canny_control_returns_unit_interval() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (108, 108), (255, 255, 255), 3)
    probability = geometry.detect_edges_canny(image)
    assert probability.dtype == np.float32
    assert probability.shape == (128, 128)
    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0
    assert np.count_nonzero(probability) > 0


def test_missing_hed_assets_fail_when_download_disabled(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="下载已禁用"):
        geometry.ensure_hed_assets(tmp_path, allow_download=False)


def test_geometry_bounded_loader_preserves_original_size(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (300, 600), color=(80, 100, 120)).save(image_path, quality=90)
    image_bgr, original_size = geometry.load_bounded_bgr(image_path, max_side=120)
    assert original_size == (300, 600)
    assert image_bgr.shape[:2] == (120, 60)
    assert image_bgr.dtype == np.uint8
