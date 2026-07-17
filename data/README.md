# 数据与素材说明

## 原作

项目主输入是 Diego Velázquez 的 *Las Meninas* 高清数字图。原文件为 `26065 × 30000` 像素、约 255 MB，超过 GitHub 普通文件的单文件限制，因此不直接提交到仓库。

运行以下命令可下载并校验完全相同的原图：

```bash
./scripts/download-source-image.sh
```

脚本会把文件保存为 `data/raw/las-meninas-original.jpg`，并验证 SHA-256：

```text
dd0cab7a6bebcee8c492f3181b324b91df8a8f23f1794dcfae45e454efa3fda0
```

来源：[Wikimedia Commons — *Las Meninas, by Diego Velázquez, from Prado in Google Earth*](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg)

Commons 文件页把 1656 年原作及这一公版二维作品的忠实复制标为 public domain，并将数字文件来源写为 “The Prado in Google Earth”。项目保留来源与校验值以保证可追溯性；不同司法辖区对 public domain 的判断可能不同，应以 Commons 完整说明和当地法律为准。

公开仓库中的两个直接代理图均由上述已校验原图缩小得到：

| 文件 | 尺寸 | SHA-256 |
| --- | --- | --- |
| `data/processed/las-meninas-2048.jpg` | 1779 × 2048 | `b422a84456946bf6d82a289496cba2af8ff86adf690114c04e6c2450ab872c31` |
| `outputs/reference/las-meninas-reference.jpg` | 1042 × 1200 | `f1d43107c86a636938ef8e93ed6cf20a8e201c96a2ca3704be752b229a5c57fd` |

## 辅助素材

本地 `pics/` 中的其余图片可用于人工对照、局部分析和艺术史比较，不构成神经网络训练集：

- `perception.png`：透视、地平线与消失点示意；
- `location.png`：人物、画布、镜子与观看位置示意；
- `mirror.png`：原作中画家和后墙镜像的局部；
- `picasso.png`：Pablo Picasso 对 *Las Meninas* 的重构，用作外部比较案例。

本仓库将公开发布，因此 `.gitignore` 会排除整个本地 `pics/` 目录。仓库不会重新分发来源未完整追溯的示意图或仍受保护的作品复制图。公开成果只包含公版原作的下载方法、可复现分析、项目生成变换及必要的文字引用。

`data/raw/` 同样被排除，因为原文件约 255 MB，超过 GitHub 普通文件限制且可由下载脚本重建。许可范围与模型权重来源的完整说明见 [`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md)。
