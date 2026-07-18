"""neural-style-transfer.py 的快速单元与端到端烟雾测试。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "neural-style-transfer.py"


def load_module() -> ModuleType:
    """加载带短横线文件名的 CLI 模块。"""

    module_name = "las_meninas_neural_style_transfer"
    specification = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"无法加载模块：{MODULE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


nst = load_module()


def make_test_image(path: Path, size: tuple[int, int], *, phase: int = 0) -> Path:
    """生成同时含水平、垂直与通道变化的确定性 RGB 测试图。"""

    width, height = size
    x_grid = np.arange(width, dtype=np.uint16)[None, :]
    y_grid = np.arange(height, dtype=np.uint16)[:, None]
    red = np.broadcast_to((5 * x_grid + phase) % 256, (height, width))
    green = np.broadcast_to((7 * y_grid + 2 * phase) % 256, (height, width))
    blue = (3 * x_grid + 11 * y_grid + 3 * phase) % 256
    array = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    Image.fromarray(array).save(path)
    return path


def make_identity_extractor(config: object) -> nn.Module:
    """构造与 VGG 层索引兼容的轻量可微测试网络。"""

    features = nn.Sequential(*[nn.Identity() for _ in range(30)])
    return nst.VGG19FeatureExtractor(
        features,
        content_layer=config.content_layer,
        style_layers=config.style_layers,
    )


def test_default_original_style_reference_exists() -> None:
    assert nst.DEFAULT_STYLE_IMAGE == (
        PROJECT_ROOT / "assets" / "style" / "cognitive-map-style-reference.png"
    )
    assert nst.DEFAULT_STYLE_IMAGE.is_file()


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1,0.25,0.5", (0.25, 0.5, 1.0)),
        ([2.0, 0.0, 1.0], (0.0, 1.0, 2.0)),
        ("0", (0.0,)),
    ],
)
def test_parse_style_strengths(raw_value: object, expected: tuple[float, ...]) -> None:
    assert nst.parse_style_strengths(raw_value) == expected


@pytest.mark.parametrize("raw_value", ["", "0.5,", "-0.1,1", "nan", "1,1", "abc"])
def test_parse_style_strengths_rejects_invalid_values(raw_value: str) -> None:
    with pytest.raises(ValueError):
        nst.parse_style_strengths(raw_value)


def test_gram_matrix_has_expected_normalization_and_gradient() -> None:
    features = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [1.0, 3.0]]]],
        requires_grad=True,
    )
    actual = nst.gram_matrix(features)
    flattened = features.reshape(1, 2, 4)
    expected = torch.bmm(flattened, flattened.transpose(1, 2)) / 8.0
    assert actual.shape == (1, 2, 2)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert features.grad is not None
    assert torch.all(torch.isfinite(features.grad))


def test_total_variation_is_zero_for_constant_image_and_positive_for_edge() -> None:
    constant = torch.full((1, 3, 8, 10), 0.5)
    assert float(nst.total_variation_loss(constant)) == pytest.approx(0.0)
    with_edge = constant.clone()
    with_edge[:, :, :, 5:] = 1.0
    assert float(nst.total_variation_loss(with_edge)) > 0.0


def test_load_resized_rgb_preserves_aspect_ratio_without_crop(tmp_path: Path) -> None:
    input_path = tmp_path / "wide.png"
    array = np.zeros((40, 80, 3), dtype=np.uint8)
    array[:, :10, 0] = 255
    array[:, -10:, 2] = 255
    Image.fromarray(array).save(input_path)

    loaded = nst.load_resized_rgb(
        input_path,
        long_side=64,
        max_source_pixels=10_000,
    )

    assert (loaded.original_width, loaded.original_height) == (80, 40)
    assert (loaded.working_width, loaded.working_height) == (64, 32)
    resized = np.asarray(loaded.image)
    # 左右两侧的特征都仍存在，说明缩放没有通过中心裁剪丢掉画面边缘。
    assert resized[:, 0, 0].mean() > 200
    assert resized[:, -1, 2].mean() > 200


def test_choose_device_auto_returns_mps_or_cpu() -> None:
    assert nst.choose_device("cpu").type == "cpu"
    assert nst.choose_device("auto").type in {"mps", "cpu"}


def test_prepare_output_directory_refuses_implicit_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        nst.prepare_output_directory(output_dir, overwrite=False)
    nst.prepare_output_directory(output_dir, overwrite=True)


def test_run_style_transfer_writes_reproducible_complete_artifacts(tmp_path: Path) -> None:
    content_path = make_test_image(tmp_path / "content.png", (48, 32), phase=3)
    style_path = make_test_image(tmp_path / "style.png", (40, 40), phase=29)

    def run_once(output_dir: Path) -> dict[str, object]:
        config = nst.StyleTransferConfig(
            content_image=content_path,
            style_image=style_path,
            output_dir=output_dir,
            style_strengths=(0.0, 0.5),
            long_side=64,
            steps=2,
            learning_rate=0.01,
            content_weight=1.0,
            style_weight=20.0,
            tv_weight=0.01,
            initial_noise=0.03,
            seed=2026,
            device="cpu",
            weights="none",
            quiet=True,
        )
        extractor = make_identity_extractor(config)
        return nst.run_style_transfer(
            config,
            extractor_override=extractor,
            model_signature_override="test-identity-feature-network",
        )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_manifest = run_once(first_dir)
    second_manifest = run_once(second_dir)

    assert (first_dir / "manifest.json").is_file()
    on_disk_manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest == first_manifest
    assert first_manifest["parameters"]["center_crop"] is False
    assert first_manifest["parameters"]["aspect_ratio_preserved"] is True
    assert first_manifest["inputs"]["content"]["working_size"] == [64, 43]
    assert len(first_manifest["outputs"]) == 2

    for strength in (0.0, 0.5):
        slug = nst.style_strength_slug(strength)
        image_name = f"neural-style-strength-{slug}.png"
        csv_name = f"loss-strength-{slug}.csv"
        json_name = f"loss-strength-{slug}.json"
        first_image = first_dir / image_name
        second_image = second_dir / image_name
        assert first_image.is_file()
        assert first_image.read_bytes() == second_image.read_bytes()
        with Image.open(first_image) as image:
            assert image.size == (64, 43)
            assert image.format == "PNG"

        with (first_dir / csv_name).open(encoding="utf-8", newline="") as file_handle:
            csv_rows = list(csv.DictReader(file_handle))
        json_payload = json.loads((first_dir / json_name).read_text(encoding="utf-8"))
        assert [int(row["step"]) for row in csv_rows] == [0, 1, 2]
        assert [row["step"] for row in json_payload["records"]] == [0, 1, 2]
        assert len(json_payload["records"]) == 3
        assert all(np.isfinite(float(row["total_loss"])) for row in csv_rows)

    first_final_losses = [output["final_loss"] for output in first_manifest["outputs"]]
    second_final_losses = [output["final_loss"] for output in second_manifest["outputs"]]
    assert first_final_losses == second_final_losses


def test_cli_main_uses_requested_inputs_and_emits_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content_path = make_test_image(tmp_path / "content.png", (48, 32), phase=5)
    style_path = make_test_image(tmp_path / "style.png", (32, 48), phase=17)
    output_dir = tmp_path / "cli-output"

    def fake_create_extractor(
        config: object,
        device: torch.device,
    ) -> tuple[nn.Module, str]:
        return make_identity_extractor(config).to(device), "test-cli-identity-feature-network"

    monkeypatch.setattr(nst, "create_vgg19_extractor", fake_create_extractor)
    exit_code = nst.main(
        [
            str(content_path),
            "--style-image",
            str(style_path),
            "--output-dir",
            str(output_dir),
            "--style-strengths",
            "0.25,0.75",
            "--long-side",
            "64",
            "--steps",
            "1",
            "--learning-rate",
            "0.01",
            "--style-weight",
            "10",
            "--initial-noise",
            "0",
            "--device",
            "cpu",
            "--quiet",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["style_strengths"] == [0.25, 0.75]
    assert manifest["implementation"]["model_signature"] == ("test-cli-identity-feature-network")
    assert len(manifest["outputs"]) == 2
