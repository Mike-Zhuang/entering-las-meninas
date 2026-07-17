# Entering *Las Meninas*: How a Painting Changed the Way I Look at Art

## 中文对应报告

## 摘要

本项目研究卷积神经网络能够从委拉斯开兹的《宫娥》中分离出什么，又遗漏了什么。我没有把 CNN 当作自动艺术评论家，而把它当作一个“不完整的观看者”。冻结的 ImageNet 预训练 VGG-19 提供两类测量：以多层 Gram matrix 距离作为有限的 style proxy，以保留二维位置的 feature map 距离描述构图和空间变化；HED、Hough 与 RANSAC 则构成边界到候选消失点的几何链路。

结果表明两类主要 transformation 在模型表征中可以部分分离。100% 非神经 style baseline 的 style distance 为 0.3025、spatial cosine distance 仅 0.0333；100% geometry transformation 分别为 0.1267 与 0.3525。最关键的实验不是最剧烈的变换，而是最小的镜像删除：它让人工定义的五条关系从 5/5 降为 4/5，但 VGG 距离只有 0.0219（style）和 0.0025（spatial）。保留全部五条关系的 matched control 反而产生更大的 style distance 0.0249。这说明 CNN 可以检测局部视觉变化，却没有证据表明它理解了“镜子为什么重要”。几何分支同样没有发现唯一稳定的消失点；估计结果随 ROI 明显改变。

最终作品 *Entering Las Meninas* 保留原作的主要关系结构，并把镜中王室人物换成当代观看者。项目没有招募人类参与者，也没有进行问卷、行为实验或脑活动测量；因此报告只陈述模型行为、人工关系审查和我在实际项目过程中的理解，不把它扩张为对一般人类知觉的结论。

## 1. 从“图像表面”到“观看位置”

在项目开始时，我很容易把《宫娥》概括成“一幅带有奇特镜子的著名群像”。当我真正建立实验变量时，这种描述显得不够。我必须明确写出：画家面对一个画外位置；左侧大画布遮挡房间；José Nieto 位于明亮后门；Infanta 被侍女围绕；小小的后墙镜面把画内空间连接到没有以实体出现在画面中的人物。

Prado 将作品记录为委拉斯开兹 1656 年的布面油画，高 320.3 cm、宽 279.1 cm，馆藏号 P001174；馆方辨认出镜中的 Philip IV 与 Mariana of Austria，并指出画中空间同时依靠科学透视、空气透视和多重光源建立。Prado 还强调，镜子使作品成为对观看、再现以及观看者自身角色的思考（[Museo Nacional del Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)）。因此，项目不能把“检测到多少纹理、边缘或人物”误写成对整件作品的理解。

主输入来自 Wikimedia Commons 所保存的公版高清忠实复制。本地实验使用 1,779 × 2,048 工作代理图，仓库同时保存来源、校验值和下载脚本（[Wikimedia Commons 文件与授权页](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg)）。

![公版原作工作参考图](../outputs/reference/las-meninas-reference.jpg)

最终研究问题是：**CNN 能在多大程度上把《宫娥》的 style、geometry 与 relational topology 分开？这种分离又如何显示“检测视觉特征”和“理解观看结构”之间的差别？** 本项目的 “relational topology” 只是操作性定义：它是一张由少量定性关系组成的图，不是完整的数学拓扑证明，也不是对人脑的测量。

## 2. Ready-made：*The Missing Viewer*

我的 ready-made 方案是一面普通矩形镜子，题为 *The Missing Viewer*。镜子位于《宫娥》复制图或投影前方的暗示观看轴线上。观众照镜子时，会暂时占据画家、镜面与多个人物所朝向的画外位置。材料本身没有特别之处，意义来自 painting–mirror–viewer 三者的对齐关系。

![AI 生成的装置概念效果图；不是现场纪实照片](../assets/installation/the-missing-viewer-mockup.png)

上图只是用于检查尺度、轴线与展厅可行性的概念 mockup，不声称装置已经真实搭建。这个方案也为数字实验提供了检验标准：即使只删除镜面内很少的像素、CNN 距离几乎不变，支撑这个 ready-made 的画内–画外连接仍可能已经消失。

## 3. 实验设计与操作性定义

项目把视觉信息分成三层：

1. **Style**：色彩、纹理、局部笔触和 CNN 通道共激活；
2. **Geometry**：坐标、尺度、角度、透视、纵深和空间布局；
3. **Relational topology**：`reflects`、`contains`、`occludes`、`faces`、`surrounds` 等定性连接。

首先建立 style 保留/改变 × geometry 保留/改变的 2 × 2 矩阵：G0S0 是未改变的工作代理图；G0S1 包含确定性色彩纹理基线和独立的 neural-style 分支；G1S0 是透视单应变换；G1S1 把 geometry-100 图作为 content 再进行强度 1.0 的神经风格迁移。style 与 geometry 还分别生成 0、25、50、75、100% 连续序列，避免只比较一对前后图。

![确定性版本的 2 × 2 变换矩阵](../outputs/figures/transformation-matrix.png)

geometry 变换使用归一化 warp pivot (0.56, 0.53)、convergence 0.14、depth expansion 0.08。它们是预先声明的干预参数，不是算法发现的历史消失点。确定性 style baseline 只改变色调、色度、细节与轻度量化，不移动像素坐标，并在所有文档中明确标注为 non-neural。

在 2 × 2 之外，我预先定义五条关系：

1. 后墙镜面连接画内空间与画外观看位置；
2. 画家面向画外位置；
3. 后门框包含 José Nieto；
4. 左侧大画布遮挡房间；
5. Infanta 被侍女群围绕。

`mirror-deletion` 只用局部墙面纹理替换镜面内部，保留镜框。`matched-control` 在完全相同的区域施加羽化色偏，并把 RMS 像素改动能量匹配到镜像删除，同时保留镜中人物。`sham-warp` 通过亚像素平移再逆平移，控制重采样本身造成的变化。关系表由人工逐项视觉核验，不是 CNN 自动推理的输出。

## 4. VGG-19 style 与 spatial 方法

VGG-19 使用连续 3 × 3 卷积，便于从同一个冻结网络读取不同深度的 feature map（[Simonyan 与 Zisserman](https://arxiv.org/abs/1409.1556)；[Torchvision VGG-19 官方文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19)）。本项目没有用一张画的数据增强去训练分类器，而是把预训练 VGG 当作固定测量工具。

所有比较图保持纵横比，letterbox 到 512 × 512；实际画面区域为 445 × 512，避免中心裁剪丢失大画布和房间边缘。style layers 为 `relu1_1`、`relu2_1`、`relu3_1`、`relu4_1`、`relu5_1`。对展平空间维度后的特征 (F_l)，计算

\[
G_l = \frac{F_lF_l^\top}{C_lH_lW_l}.
\]

这与 Gatys neural style transfer 中以通道相关性表示风格的思路一致（[Gatys、Ecker 与 Bethge](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)；[PyTorch 官方教程](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)）。报告中的 **style distance** 是五层 Gram matrix 相对 Frobenius 距离的平均值；它只是模型定义的纹理/共激活统计，不等于完整艺术史风格。

spatial descriptor 来自 `relu3_1`、`relu4_2` 与 `relu5_1`，并保留为 32 × 32 网格。主文把对应空间位置的平均 cosine distance 称为 **spatial distance**，同时保留 relative RMS 作为幅度指标。二者仍可能响应色彩和纹理，不能被称为纯 geometry。正式 reference 与 0% 条件现在使用完全相同的工作代理图，因此所有 0% 距离严格为 0，不再混入 JPEG/PNG 编码差异。

神经风格迁移使用冻结 VGG-19、Adam 优化 500 步，long side 512；content layer 为 `relu4_2`，content weight 1，style weight (10^6)，TV weight (10^{-4})，学习率 0.02，style strength 为 0.25、0.5、1.0，随机种子 139，并为三种强度共享相同初始状态。style donor 是本项目生成的抽象图，只包含嵌套画框、构造线和镜面般的亮矩形，没有提示模仿任何具名现代艺术家。

## 5. HED、Hough 与 RANSAC 几何链路

HED 是带深监督的全卷积边缘网络，本项目用它输出 boundary probability map（[Xie 与 Tu](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html)）。经过阈值化和细化后，probabilistic Hough 提取候选线段，RANSAC 再选择“主导线段交点候选”；Canny + Hough 是非神经对照（[OpenCV Hough 官方文档](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html)）。工作图在此分支缩放为 1,112 × 1,280；检测到 334 条 HED 线段，Canny 最多保留 400 条，RANSAC 运行 5,000 次。

我比较了不加 ROI 的全画面自动运行，以及 art-history-motivated ROI 与 rear-door ROI 两个显式先验窗口。ROI 是分析者给出的条件，不能包装成模型自动发现。

## 6. 实际结果

### 6.1 Style 与 geometry 可以部分分离

确定性 style trajectory 在 25、50、75、100% 时的 style distance 为 0.0795、0.1569、0.2300、0.3025，而 spatial cosine 只从 0.0059 上升到 0.0333。geometry trajectory 的 style distance 为 0.0885、0.1043、0.1185、0.1267，spatial cosine 则从 0.0971 上升到 0.3525。100% geometry 的 spatial distance 是 100% style baseline 的 10.6 倍；反过来，style baseline 的 Gram distance 是 geometry 的 2.4 倍。它不是完美正交分离，因为 warp 会改变局部统计、style 也会影响空间 feature，但方向清楚且可复核。

neural-style 0.25、0.5、1.0 的 style distance 为 0.5030、0.5303、0.5549，spatial cosine 为 0.1045、0.1148、0.1298。随着强度增加，最终 content loss 从 0.2298 上升到 0.5764；raw style loss 从 (1.437 \times 10^{-6}) 降到 (5.931 \times 10^{-7})，但乘上强度后的 weighted style loss 从 0.3593 上升到 0.5931。

![神经风格迁移连续序列](../outputs/figures/neural-style-sequence.png)

| 2 × 2 条件 | Style distance | Spatial cosine | Spatial relative RMS |
| --- | ---: | ---: | ---: |
| G0S0：原始工作代理图 | 0.0000 | 0.0000 | 0.0000 |
| G0S1：neural style 1.0 | 0.5549 | 0.1298 | 0.4822 |
| G1S0：geometry 100% | 0.1267 | 0.3525 | 0.9189 |
| G1S1：geometry-100 content + neural style 1.0 | 0.5676 | 0.4004 | 0.9083 |

G1S1 的 spatial cosine 高于两个单因素条件，同时保留很强的 Gram displacement。确定性 combined endpoint 则为 0.3210（style）、0.3631（spatial cosine）和 0.9807（spatial RMS）。cosine 与 RMS 的排序不完全相同，说明报告必须明确写出指标，而不能只给一个含义不清的“相似度”。

### 6.2 破坏关键关系几乎没有移动 VGG

人工关系审查中，original、matched-control、geometry-100、neural-style-1 和 combined-neural-geometry 均保留 5/5 关系；mirror-deletion 为 4/5，因为它单独切断了镜面与画外观看位置的连接。

![原图、matched control、镜像删除与 sham warp](../outputs/figures/topology-controls.png)

然而，mirror-deletion 的 VGG style distance 只有 0.0219，spatial cosine 只有 0.0025。保留五条关系的 matched-control 反而具有更大的 style distance 0.0249。镜像删除的 spatial cosine 比 matched edit 高约 66%（0.0025 对 0.0015），但只控制重采样的 sham-warp 更高，达到 0.0036。因此，现有结果不能证明 CNN 编码了缺失的“反射连接”；更合理的解释是模型响应局部外观和采样差异，而人工判断的关系损失几乎不影响其距离排序。

这改变了我对 “extract topology” 的理解：CNN 可以提供候选区域与局部特征，但 `mirror connects depicted room to an unseen position` 并没有自动作为图的一条边出现。项目必须先声明、再干预、最后人工审查这条边。

### 6.3 消失点估计高度依赖 ROI

不加 ROI 的全画面运行得到 HED/RANSAC 候选 (0.720, 0.697)，57 条 inlier，weighted inlier ratio 0.1728；Canny 却给出 (0.217, 0.619)，两者的归一化图像对角线距离为 0.3348。这里的 medium 证据等级只描述 HED 候选内部的线段共识，并不表示两种方法彼此一致。全画面的明显分歧本身就说明估计不稳定。

![HED 边界、Hough 线段与全画面 RANSAC 候选](../outputs/geometry-analysis/original/cnn-lines-vanishing-point.png)

art-history-motivated ROI 中，HED 移到 (0.529, 0.439)，只有 19 条 inlier、ratio 0.0552；Canny 为 (0.518, 0.472)，证据等级 low。rear-door ROI 中，HED 又移到 (0.626, 0.612)，ratio 0.1063；Canny 为 (0.634, 0.600)，仍为 low。也就是说，两种 edge method 在无约束全画面上明显分歧，却会在各个人工选定窗口内偶然靠近；而不同窗口本身又给出相距很远的候选点。因此，无论是 HED 内部线段共识，还是带条件的 HED–Canny 局部一致，都不能验证一个唯一的艺术史消失点。

因此，几何分支的诚实结论是一次有信息量的自动发现失败：HED 的确提取了多尺度边界，但 Hough/RANSAC 会偏向 ROI 中占优势的画框、门框、人物轮廓和笔触边界。它不能证明委拉斯开兹采用了某个唯一算法点，也不能取代 Prado 对线性透视、空气透视和光线的综合解释。

## 7. Picasso 的本地外部对照

本地分析使用了一张与 Picasso 1957 年 *Las Meninas* 大型版本相符的复制图。Museu Picasso 记录该作为 194 × 260 cm、馆藏号 MPB 70.433，并指出 Picasso 把竖幅变为横幅、放大画家、平面化和重组空间，同时保留人物群与门中人物（[Museu Picasso Barcelona](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)）。它因此比随机滤镜更适合作为“外观和几何剧变，但角色关系仍可对应”的概念压力测试。

该本地复制图相对 reference 的 style distance 为 1.4196、spatial cosine 为 0.7139，都是本组最高值。这些数字只是该 VGG 表征下的距离，不是识别率。输入还存在画幅、复制条件和授权状态差异。由于没有采集人类识别判断，我不能据此声称“人类比 CNN 更会泛化”。我只能指出：博物馆已经建立两件作品的艺术史关系，而本项目的 VGG 表征把两张具体输入放得很远。由于复制图授权链未确认且官方页面保留图像权利，项目只在本地计算，不在公有仓库重新分发图片。

## 8. 最终作品：*Entering Las Meninas*

最终作品把实验分开的层重新组合：保留画家与大画布、Infanta 与侍女、狗、后门、竖幅和主要纵深；把透视线和节点关系转化为墙面与地面的炭笔构造痕迹；最重要的是，它没有删除镜面连接，而把镜中王室人物换成一个当代观看者剪影。

![最终 AI-assisted 作品 *Entering Las Meninas*](../outputs/artwork/entering-las-meninas-final.png)

人工审查因此仍给最终作品 5/5，但明确记录镜中身份改变。它的 VGG style distance 为 0.4614、spatial cosine 为 0.3638：两个表征都远离原作，却有意保持五条关系。这张图不是 CNN 数值实验的直接输出，而是受实验问题启发的独立生成式编辑。测量帮助我决定“什么必须保留”，最终创作则做出了 VGG 无法代替的解释选择。

## 9. 局限与证据边界

这是单件作品的计算案例研究，不是一般艺术知觉理论。VGG-19 与 HED 主要在自然图像数据上训练；Gram matrix 有意弱化位置，spatial feature 仍响应 style；homography 会引入裁切和插值，neural style 也可能改变边界；32 × 32 网格限制细节定位。

topology score 只有五条人工二值关系，没有第二位标注者和 inter-rater reliability。它使项目假设可审计，但不是客观 ground truth。HED/Hough/RANSAC 也没有人工确认的消失点真值，ROI 依赖意味着结果只能称为 candidate。

本项目没有招募任何人类参与者，没有问卷、行为实验、眼动或神经记录。因此我不声称某个 transformation 会普遍增强或削弱“进入感”，不声称实验已经证明人类与 CNN 分歧，也不把 VGG feature 称为人类 cognitive map。Tolman 的 cognitive map 来自空间学习问题，O'Keefe 与 Dostrovsky 的证据来自海马单元记录；本项目只借用“组织关系”的思想（[Tolman, 1948](https://doi.org/10.1037/h0061626)；[O'Keefe 与 Dostrovsky, 1971](https://doi.org/10.1016/0006-8993(71)90358-1)）。

## 10. 我学到了什么

项目最重要的结果不是 CNN “解决了”《宫娥》，而是模型的不完整性变得可以测量。style 与 geometry trajectory 说明，同一 CNN 在保留或弱化空间坐标时能提供不同但有用的摘要；HED 说明学习到的 boundary map 可以支持几何后处理，却不会自动理解透视；mirror matched control 则揭示更深的边界：一个对我的解读至关重要的关系变化，可以在 VGG 距离中几乎不可见，甚至被普通控制变换超过。

这次实际过程改变了我观看艺术的方式，因为它迫使我不再把构图看成“装着可识别物体的容器”。我开始寻找那些让作品延伸到画框之外的关系：谁面向谁、什么遮挡什么、哪个开口包含另一层空间、作品把我的身体放在哪里。在《宫娥》中，镜子占据的像素很少，却在观看结构中具有很大权重。最终作品保留这条连接，正是因为实验让我意识到：对一件作品的忠实有时存在于关系，而不在表面。

CNN 的价值恰恰来自它没有替我观看。它分离统计、暴露混淆因素，并让我更清楚地说明自己的解释。“进入画中”最终不只是模拟更强的纵深，而是意识到原作早已为观看者安排位置，并在最终作品中把这个位置变得可见。

## AI 与素材披露

项目使用冻结 VGG-19、Gatys-style 像素优化、预训练 HED、OpenCV 几何和确定性图像变换。代码、参数、校验值、CSV、JSON manifest 与输出均保留以供审计。style-reference、装置 mockup 与最终作品由 Codex 图像生成工具根据项目自行撰写的 prompt 生成。工具没有返回可记录的模型版本或随机种子，因此这三张 raster image 不能保证逐像素复现；完整 prompt 与 SHA-256 分别记录在 `assets/README.md` 和 `outputs/artwork/README.md`。mockup 明确标注为生成图，最终作品明确标注为 AI-assisted。委拉斯开兹原作复制图为公版；Picasso 复制图和未核实来源的课堂辅助图保留在本地，不在公有仓库重新分发。

## 参考资料

- [Museo Nacional del Prado：*The Family of Felipe IV, or Las Meninas*](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)
- [Museu Picasso Barcelona：*Las Meninas*](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)
- [Simonyan & Zisserman：VGG](https://arxiv.org/abs/1409.1556)
- [Gatys, Ecker, & Bethge：Image Style Transfer Using CNNs](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)
- [Xie & Tu：Holistically-Nested Edge Detection](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html)
- [OpenCV：Hough Line Transform](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html)
- [Tolman：Cognitive Maps in Rats and Men](https://doi.org/10.1037/h0061626)
- [O'Keefe & Dostrovsky：The Hippocampus as a Spatial Map](https://doi.org/10.1016/0006-8993(71)90358-1)
