# 第三方来源、权利与再分发说明

本文件记录公有仓库实际包含或运行时获取的第三方材料。它是来源审计记录，不是法律意见，也不会把第三方材料改授为本项目的 MIT License。

## 1. Diego Velázquez, *Las Meninas*

- 作品：Diego Velázquez, *Las Meninas*, 1656，布面油画，Museo Nacional del Prado，馆藏号 P001174。
- 馆藏记录：[Museo Nacional del Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)。
- 数字输入：[Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg)；文件页将原作以及公版二维作品的忠实复制标为 public domain，并将数字文件来源写为 “The Prado in Google Earth”。
- 原文件 SHA-1：`5ae2431dd56805c8f637819a63c7a36cda0c30bf`。
- 原文件 SHA-256：`dd0cab7a6bebcee8c492f3181b324b91df8a8f23f1794dcfae45e454efa3fda0`。

仓库不提交 266,867,780-byte 的原始 JPEG。`scripts/download-source-image.sh` 从上述 Commons 文件的直链下载并校验完全相同的文件。`data/processed/las-meninas-2048.jpg` 与 `outputs/reference/las-meninas-reference.jpg` 是该公版输入的缩小版本；`outputs/transformations/`、`outputs/geometry-analysis/`、`outputs/cnn-analysis/` 和 `outputs/neural-style-transfer/` 中的部分结果也以它为输入。底层公版内容不会因缩放、分析或本项目的软件许可而变成专有材料。

Public-domain 判断可能因司法辖区而异。使用者应以 Commons 文件页的完整权利说明及自己所在地的法律为准。

## 2. 明确不在公有仓库再分发的本地图片

整个 `/pics/` 目录被 `.gitignore` 排除，且公开发布检查会拒绝其中任何文件进入 Git 索引：

| 本地文件 | SHA-256 | 已核验状态 | 公有仓库处理 |
| --- | --- | --- | --- |
| `perception.png` | `070d97f487755826289867b6c15919ec631c2d3dd5c911989bbb20655844dacc` | 未找到可验证的原始出版物或许可 | 不提交；只作本地构思参考 |
| `location.png` | `78752b3439a250e3bacff898f7f6d776e55a22f75e45cfa16b509a6e6b33388a` | 未找到可验证的原始出版物或许可 | 不提交；只作本地构思参考 |
| `mirror.png` | `807ba7e96546762b6e59b343c41bb88a25769ae7500df9959b888f0950304eec` | 《宫娥》的局部裁切，但该本地副本无需单独发布 | 不提交；需要时由已注明来源的公版输入重建 |
| `picasso.png` | `fd3932138def185693b78036659f5c01602b3ad492316dd56d0e326141805771` | 本地下载来源与授权链未建立 | 不提交；只链接收藏机构页面 |

Picasso 的 1957 年 *Las Meninas* 馆藏记录位于 [Museu Picasso Barcelona](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)。该页面明确标示作品复制图权利由 Sucesión Picasso / VEGAP 保留。因此，Velázquez 原作的公版状态不能延伸到 Picasso 的重构作品。

## 3. 运行时模型与权重

模型文件均在运行时下载到用户缓存或由用户提供，不纳入仓库，也不属于本项目的 MIT License：

- **Torchvision VGG-19 / `VGG19_Weights.IMAGENET1K_V1`**：由 Torchvision 下载；[官方模型文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19)。预训练权重与其训练数据可能有独立条款，使用者应自行核对上游说明。
- **Xie–Tu HED BSDS 模型**：模型从 [UCSD 原作者站点](https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel) 获取，SHA-256 为 `4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7`；网络定义从 [作者仓库](https://github.com/s9xie/hed/blob/master/examples/hed/deploy.prototxt) 获取，SHA-256 为 `378a9246383da889cf8e0290c47554d75dcf9c5b6bbabd8ab6c481c34aa12b8a`。本项目只记录来源与校验值，不对模型权重作许可保证。

依赖清单中的 PyTorch、Torchvision、OpenCV、NumPy、SciPy、scikit-image、scikit-learn、Pillow、pandas、Matplotlib 与 tqdm 不在仓库中复制；包管理器取得的每个依赖仍受各自上游许可约束。

## 4. 论文、博物馆文字与网页

`report/sources.md` 引用的论文、博物馆说明和官方文档仅以短引、释义与链接使用，并未作为全文复制到仓库。其版权与使用条款仍归各自权利人所有。引用不表示作者或机构认可本项目。

## 5. 项目生成图片与分析结果

以下文件由本项目通过图像生成工具制作，具体提示词、用途与 SHA-256 见相邻说明文件：

- `assets/style/cognitive-map-style-reference.png`；
- `assets/installation/the-missing-viewer-mockup.png`；
- `outputs/artwork/entering-las-meninas-final.png`。

它们不是第三方馆藏复制图，也不在 MIT License 范围内。本仓库没有通过 `LICENSE` 向这些图片授予通用再利用许可；同时，本说明不主张任何特定 AI 输出必然在所有司法辖区构成可受版权保护的作品。底层的 Velázquez 公版元素继续保持其原有状态。

由代码生成的分析图、变换图、表格、演示文稿和报告同样不因代码采用 MIT License 而自动获得 MIT 授权。任何人都可以依照 MIT License 运行代码，重新生成自己的分析结果。
