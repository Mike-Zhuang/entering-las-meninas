from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.transformations import (
    DEFAULT_LEVELS,
    FIXED_CROPS,
    MIRROR_CONTENT_BOX,
    MIRROR_CONTROL_BOX,
    NormalizedBox,
    combined_transform,
    create_working_proxy,
    deterministic_style_transform,
    extract_fixed_crops,
    generate_geometry_sequence,
    generate_style_sequence,
    generate_topology_variants,
    matched_control,
    mirror_deletion,
    parse_levels,
    perspective_depth_transform,
    perspective_homography,
    run_controlled_transformations,
    sham_warp,
)
from src.visualization import (
    create_parallax_video,
    heuristic_vertical_depth,
    normalize_depth_map,
    plot_distance_heatmap,
    plot_feature_map_grid,
    plot_feature_space,
    plot_image_comparison,
    plot_metric_curves,
    plot_style_geometry_scatter,
    plot_transformation_matrix,
)


def synthetic_painting(width: int = 180, height: int = 220) -> Image.Image:
    """生成包含渐变、线条和局部人物块的确定性测试图。"""

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    red = np.broadcast_to(35.0 + 160.0 * x, (height, width))
    green = np.broadcast_to(25.0 + 145.0 * y, (height, width))
    blue = 20.0 + 115.0 * (0.58 * x + 0.42 * y)
    array = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, width - 20, height - 25), outline=(224, 194, 138), width=3)
    draw.line((0, height - 1, width // 2, height // 2), fill=(245, 238, 215), width=2)
    draw.line((width - 1, height - 1, width // 2, height // 2), fill=(245, 238, 215), width=2)

    mirror = MIRROR_CONTENT_BOX.to_pixels(image.size)
    draw.rectangle(mirror, fill=(40, 60, 88), outline=(214, 180, 108), width=1)
    mirror_left, mirror_top, mirror_right, mirror_bottom = mirror
    draw.ellipse(
        (
            mirror_left + 1,
            mirror_top + 1,
            max(mirror_left + 2, mirror_right - 4),
            max(mirror_top + 2, mirror_bottom - 2),
        ),
        fill=(230, 190, 142),
    )
    control = MIRROR_CONTROL_BOX.to_pixels(image.size)
    draw.rectangle(control, fill=(64, 55, 44))
    for offset in range(0, max(1, control[2] - control[0]), 3):
        draw.line(
            (control[0] + offset, control[1], control[0], min(control[3], control[1] + offset)),
            fill=(85, 72, 58),
            width=1,
        )
    return image


def assert_nonempty_file(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0


def test_normalized_box_validation_and_pixel_conversion() -> None:
    box = NormalizedBox(0.1, 0.2, 0.5, 0.8)
    assert box.to_pixels((100, 50)) == (10, 10, 50, 40)
    with pytest.raises(ValueError):
        NormalizedBox(0.5, 0.1, 0.4, 0.8)
    with pytest.raises(ValueError):
        NormalizedBox(0.1, -0.1, 0.4, 0.8)


def test_parse_levels_is_ordered_and_strict() -> None:
    assert parse_levels("100,0,50,25,75") == DEFAULT_LEVELS
    assert parse_levels([0, 100]) == (0, 100)
    with pytest.raises(ValueError):
        parse_levels("0,25,25")
    with pytest.raises(ValueError):
        parse_levels("0,101")
    with pytest.raises(ValueError):
        parse_levels(())


def test_proxy_and_fixed_crops_are_reproducible(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    proxy_path = tmp_path / "proxy.png"
    synthetic_painting(420, 300).save(source_path, quality=95)

    proxy = create_working_proxy(source_path, proxy_path, max_long_side=160)
    assert proxy.mode == "RGB"
    assert max(proxy.size) == 160
    assert_nonempty_file(proxy_path)

    crop_paths = extract_fixed_crops(proxy, tmp_path / "crops")
    assert set(crop_paths) == set(FIXED_CROPS)
    for name, path in crop_paths.items():
        assert path.name == f"{name}.png"
        assert_nonempty_file(path)
        with Image.open(path) as crop:
            assert crop.width > 0
            assert crop.height > 0


def test_proxy_enforces_source_pixel_limit(tmp_path: Path) -> None:
    source_path = tmp_path / "too-large-for-test-limit.png"
    synthetic_painting(80, 80).save(source_path)
    with pytest.raises(ValueError, match="exceeding the configured limit"):
        create_working_proxy(source_path, max_long_side=64, source_pixel_limit=1_000)


def test_perspective_geometry_sequence_has_exact_baseline_and_continuity() -> None:
    image = synthetic_painting()
    original = np.asarray(image)
    baseline = np.asarray(perspective_depth_transform(image, 0.0))
    assert np.array_equal(baseline, original)

    matrix = perspective_homography(image.size, 1.0)
    assert matrix.shape == (3, 3)
    assert np.all(np.isfinite(matrix))
    assert not np.allclose(matrix, np.eye(3))

    sequence = generate_geometry_sequence(image)
    assert tuple(sequence) == DEFAULT_LEVELS
    differences = [
        float(np.mean(np.abs(np.asarray(sequence[level], dtype=float) - original)))
        for level in DEFAULT_LEVELS
    ]
    assert differences[0] == 0.0
    assert all(
        later > earlier for earlier, later in zip(differences, differences[1:], strict=False)
    )
    assert np.array_equal(
        np.asarray(perspective_depth_transform(image, 0.75)),
        np.asarray(perspective_depth_transform(image, 0.75)),
    )


def test_perspective_rejects_invalid_parameters() -> None:
    image = synthetic_painting()
    with pytest.raises(ValueError):
        perspective_depth_transform(image, -0.01)
    with pytest.raises(ValueError):
        perspective_depth_transform(image, 1.01)
    with pytest.raises(ValueError):
        perspective_homography(image.size, 1.0, vanishing_point=(1.2, 0.5))


def test_style_sequence_is_deterministic_non_geometric_baseline() -> None:
    image = synthetic_painting()
    original = np.asarray(image)
    baseline = np.asarray(deterministic_style_transform(image, 0.0))
    full_a = np.asarray(deterministic_style_transform(image, 1.0))
    full_b = np.asarray(deterministic_style_transform(image, 1.0))
    assert np.array_equal(baseline, original)
    assert np.array_equal(full_a, full_b)
    assert full_a.shape == original.shape
    assert float(np.mean(np.abs(full_a.astype(float) - original.astype(float)))) > 2.0

    sequence = generate_style_sequence(image)
    differences = [
        float(np.mean(np.abs(np.asarray(sequence[level], dtype=float) - original)))
        for level in DEFAULT_LEVELS
    ]
    assert differences[0] == 0.0
    assert all(
        later > earlier for earlier, later in zip(differences, differences[1:], strict=False)
    )


def test_combined_transform_changes_both_dimensions() -> None:
    image = synthetic_painting()
    combined = combined_transform(image, geometry_amount=1.0, style_amount=1.0)
    assert combined.size == image.size
    assert not np.array_equal(np.asarray(combined), np.asarray(image))
    assert np.array_equal(
        np.asarray(combined_transform(image, geometry_amount=0.0, style_amount=0.0)),
        np.asarray(image),
    )


def test_topology_variants_localize_the_declared_changes() -> None:
    image = synthetic_painting()
    original = np.asarray(image, dtype=np.int16)
    mirror_pixels = MIRROR_CONTENT_BOX.to_pixels(image.size)
    ml, mt, mr, mb = mirror_pixels
    control_pixels = MIRROR_CONTROL_BOX.to_pixels(image.size)
    cl, ct, cr, cb = control_pixels

    deletion = np.asarray(mirror_deletion(image), dtype=np.int16)
    deletion_delta = np.abs(deletion - original).sum(axis=2)
    assert deletion_delta[mt:mb, ml:mr].mean() > 0.0
    outside = deletion_delta.copy()
    outside[mt:mb, ml:mr] = 0
    assert outside.max() == 0

    control = np.asarray(matched_control(image), dtype=np.int16)
    control_delta = np.abs(control - original).sum(axis=2)
    assert control_delta[ct:cb, cl:cr].mean() > 0.0
    outside_control = control_delta.copy()
    outside_control[ct:cb, cl:cr] = 0
    assert outside_control.max() == 0
    deletion_patch_delta = (deletion - original)[mt:mb, ml:mr].astype(float)
    control_patch_delta = (control - original)[ct:cb, cl:cr].astype(float)
    deletion_rms = np.sqrt(np.mean(np.square(deletion_patch_delta)))
    control_rms = np.sqrt(np.mean(np.square(control_patch_delta)))
    assert control_rms == pytest.approx(deletion_rms, rel=0.05, abs=0.5)

    zero_sham = np.asarray(sham_warp(image, 0.0))
    full_sham = np.asarray(sham_warp(image, 1.0))
    assert np.array_equal(zero_sham, np.asarray(image))
    assert full_sham.shape == np.asarray(image).shape
    assert not np.array_equal(full_sham, np.asarray(image))

    variants = generate_topology_variants(image)
    assert set(variants) == {
        "topology-original",
        "topology-mirror-deletion",
        "topology-matched-control",
        "topology-sham-warp",
    }


def test_controlled_pipeline_writes_all_declared_artifacts(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    output_dir = tmp_path / "controlled"
    synthetic_painting(240, 300).save(source_path)

    manifest = run_controlled_transformations(
        source_path,
        output_dir,
        levels=(0, 50, 100),
        proxy_long_side=160,
    )
    manifest_path = Path(str(manifest["manifest_path"]))
    assert_nonempty_file(manifest_path)
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["levels"] == [0, 50, 100]
    assert loaded_manifest["method_note"].startswith("The style-only sequence is a deterministic")
    assert len(loaded_manifest["records"]) == 11
    assert set(loaded_manifest["matrix_inputs"]) == {
        "original",
        "style_only",
        "geometry_only",
        "combined",
    }
    for path in loaded_manifest["matrix_inputs"].values():
        assert_nonempty_file(Path(path))
    assert len(list((output_dir / "geometry").glob("*.png"))) == 3
    assert len(list((output_dir / "style").glob("*.png"))) == 3
    assert len(list((output_dir / "topology").glob("*.png"))) == 4
    assert len(list((output_dir / "crops").glob("*.png"))) == len(FIXED_CROPS)


def test_visualization_functions_write_complete_figures(tmp_path: Path) -> None:
    original = synthetic_painting(120, 150)
    style = deterministic_style_transform(original, 1.0)
    geometry = perspective_depth_transform(original, 1.0)
    combined = combined_transform(original, geometry_amount=1.0, style_amount=1.0)

    matrix_path = plot_transformation_matrix(
        original,
        style,
        geometry,
        combined,
        tmp_path / "matrix.png",
        dpi=80,
    )
    comparison_path = plot_image_comparison(
        {"Original": original, "Mirror deleted": mirror_deletion(original), "Control": matched_control(original)},
        tmp_path / "comparison.png",
        columns=2,
        title="Topology conditions",
        dpi=80,
    )
    for path in (matrix_path, comparison_path):
        assert_nonempty_file(path)
        with Image.open(path) as figure:
            assert figure.width > 300
            assert figure.height > 300


def test_feature_and_metric_plotting_functions(tmp_path: Path) -> None:
    grid_path = plot_feature_map_grid(
        {
            "Original": {
                "conv1": np.arange(64, dtype=float).reshape(8, 8),
                "conv3": np.ones((4, 8, 8), dtype=float),
            },
            "Geometry 100%": {
                "conv1": np.flipud(np.arange(64, dtype=float).reshape(8, 8)),
                "conv3": np.stack([np.eye(8)] * 4),
            },
        },
        tmp_path / "feature-grid.png",
        normalization="global",
        dpi=80,
    )
    heatmap_path = plot_distance_heatmap(
        [[0.0, 0.2, 0.8], [0.0, 0.5, 0.9]],
        ["conv1", "conv3"],
        ["Original", "50%", "100%"],
        tmp_path / "heatmap.png",
        dpi=80,
    )
    curves_path = plot_metric_curves(
        [0, 25, 50, 75, 100],
        {
            "Style distance": [0.0, 0.18, 0.41, 0.70, 1.0],
            "Geometry distance": [0.0, 0.22, 0.45, 0.67, 0.91],
        },
        tmp_path / "curves.png",
        dpi=80,
    )
    feature_space_path = plot_feature_space(
        [[0.0, 0.0], [0.3, 0.1], [0.2, 0.7]],
        ["Original", "Style", "Geometry"],
        tmp_path / "feature-space.png",
        groups=["baseline", "style", "geometry"],
        connect_order=True,
        dpi=80,
    )
    semantic_scatter_path = plot_style_geometry_scatter(
        [0.0, 0.8, 0.2],
        [0.0, 0.1, 0.9],
        ["Original", "Style", "Geometry"],
        tmp_path / "style-geometry.png",
        dpi=80,
    )
    for path in (
        grid_path,
        heatmap_path,
        curves_path,
        feature_space_path,
        semantic_scatter_path,
    ):
        assert_nonempty_file(path)


def test_parallax_video_requires_explicit_depth_provenance(tmp_path: Path) -> None:
    image = synthetic_painting(80, 100)
    with pytest.raises(ValueError, match="depth_map is required"):
        create_parallax_video(image, tmp_path / "missing-depth.gif", duration_seconds=0.5, fps=6)


def test_parallax_gif_and_depth_helpers(tmp_path: Path) -> None:
    image = synthetic_painting(80, 100)
    depth = heuristic_vertical_depth(image.size)
    normalized = normalize_depth_map(depth)
    inverted = normalize_depth_map(depth, invert=True)
    assert normalized.shape == (image.height, image.width)
    assert np.allclose(normalized + inverted, 1.0)
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

    output = create_parallax_video(
        image,
        tmp_path / "parallax.gif",
        depth_map=depth,
        duration_seconds=0.6,
        fps=6,
        horizontal_amplitude=5.0,
        vertical_amplitude=1.0,
        max_long_side=100,
    )
    assert_nonempty_file(output)
    with Image.open(output) as animation:
        assert animation.n_frames >= 3


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (plot_metric_curves, ([0, 1], {"bad": [0.0]}, "unused.png")),
        (plot_feature_space, ([[0.0, 1.0, 2.0]], ["bad"], "unused.png")),
        (plot_distance_heatmap, ([[1.0, 2.0]], ["row"], ["only-one"], "unused.png")),
    ],
)
def test_plotting_validation_rejects_mismatched_inputs(function: object, arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
