# 仓库导览与保留策略

这个仓库记录了同一课程项目在不同阶段形成的多条工作线。重构的目标不是删除旧成果，而是明确主次：老师从最终演示进入，核心代码紧随其后，其他研究作为支持实验或扩展材料保留。

## 入口顺序

1. `entering-las-meninas-10-minute-presentation.pptx`
2. `README.md`
3. `OH_Programming/submission/README-en.md`
4. `OH_Programming/README-zh.md`
5. `entering-las-meninas-10-minute-presentation/README.md`

## 目录分类

| 路径 | 分类 | 用途 | 是否保持原路径 |
| --- | --- | --- | --- |
| `entering-las-meninas-10-minute-presentation.pptx` | 主交付 | 10 分钟英文最终演示 | 是，根目录唯一主 PPT |
| `entering-las-meninas-10-minute-presentation/` | 主交付源工程 | PPTD、页面、媒体、QA、讲稿与历史导出 | 是 |
| `OH_Programming/` | 主实验 | 几何自编码器、模型、全部输出和提交 notebook | 是，代码和 notebook 有路径依赖 |
| `outputs/artwork/` | 主项目媒体 | 最终静态作品及说明 | 是，被多份文档引用 |
| `outputs/video/` | 主项目媒体 | 最终视差视频及说明 | 是，被扩展流水线引用 |
| `style_transfer/` | 支持实验 | 独立 VGG-19 风格迁移包 | 是，可单独交给老师运行 |
| `EXTENDED-CNN-STUDY.md` | 扩展研究 | 原根 README，保留 VGG、几何、关系拓扑和镜像消融叙事 | 是，根目录相对链接可继续工作 |
| `src/` | 扩展研究代码 | VGG 表示、几何分析、变换、汇总和可视化 | 是 |
| `scripts/` | 扩展研究工具 | 下载、全流水线、结果验证和公开发布检查 | 是 |
| `tests/` | 扩展研究验证 | 单元测试与发布安全测试 | 是 |
| `report/` | 扩展研究文稿 | 原长篇报告、来源和中文版本 | 是 |
| `outputs/figures/`、`outputs/metrics/` | 扩展研究结果 | 原定量分析图和表 | 是 |
| `presentation/`、`outputs/presentation/` | 历史演示 | 扩展 CNN 研究的早期演示和构建器 | 是 |
| `assignment-1-first-delivery-zh.md` | 早期课程材料 | 第一次作业的中文稿 | 是 |
| `assignment-1-submission/` | 本地课程归档 | 中英文作业包，含不适合公开再分发的材料 | 是；继续由 `.gitignore` 排除 |
| `pics/` | 本地研究素材 | 超大原图和权利状态不统一的比较素材 | 是；继续由 `.gitignore` 排除 |

## 为什么没有大规模移动代码

以下路径具有真实依赖，移动会增加无必要的回归风险：

- `OH_Programming/autoencoder-experiment.py` 使用脚本所在目录解析作品、输入、输出和模型路径。
- `OH_Programming/build-submission-notebooks.py` 与两份 notebook 会搜索 `OH_Programming/submission/`。
- 根目录扩展流水线默认向 `outputs/` 写入，并由测试和验证脚本检查固定产物。
- `presentation/build-deck.mjs` 默认导出到 `outputs/presentation/`。
- PPTD manifest、十个 `.page` 文件和 `media/` 使用演示工程内部的相对路径。

因此本次重构主要发生在导航层：根 README、主 PPT、演示工程内部归档和仓库导览。运行目录保留原位。

## 演示版本说明

- 根目录 `entering-las-meninas-10-minute-presentation.pptx`：最终主版本。
- 演示工程内同名 PPTX：源工程生成时的导出快照。
- `entering-las-meninas-10-minute-presentation/archive/earlier-editable-export.pptx`：更早的可编辑导出，保留用于追溯。

三者都保留，但只有根目录版本被标记为老师应首先打开的文件。

## 本地但不公开的内容

`.gitignore` 中的 `.cache/`、`.venv/`、`pics/` 和 `assignment-1-submission/` 继续留在本地。它们不是未提交改动，也不应为了“目录干净”而删除：缓存和虚拟环境可重建，课程归档与受限图片则有明确的保留或再分发边界。
