# 《宫娥》项目：研究证据、来源与可引用表述

> 项目题目：*Entering Las Meninas: How a Painting Changed the Way I Look at Art*
> 内部研究副标题：*The CNN as an Incomplete Viewer: Separating Style, Geometry, and Relational Topology*
> 核验日期：2026-07-17
> 本文用途：为最终报告、图注、README 和演示稿提供可追溯的事实、方法依据与引用边界。

## 1. 证据标准

本文只采用以下三类来源：

1. 作品事实与艺术史解释：作品收藏机构或博物馆的第一方页面；
2. CNN、风格迁移与边缘检测：原始论文、作者发布的代码/模型页面或框架官方文档；
3. 认知地图：提出相关概念或提供原始实验依据的论文。

本文不把搜索摘要、百科页面、博客或无来源图片当作论证依据。凡是超出来源直接陈述的内容，均明确标为“本项目的推论”或“操作性定义”。这一区分尤其重要，因为《宫娥》的镜像与观看位置长期存在多种解释；本项目可以研究这种不确定性，但不能把某一种空间重建写成无争议事实。[Museo Nacional del Prado：作品页面](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

## 2. 原作事实与高清输入图来源

### 2.1 作品身份

Prado 的馆藏记录将作品确定为 Diego Rodríguez de Silva y Velázquez 的 *Las Meninas*，创作于 1656 年，媒介为布面油画，馆藏号为 P001174；馆方技术数据给出的尺寸为高 320.3 cm、宽 279.1 cm。[Museo Nacional del Prado：作品记录与技术数据](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

可直接用于报告的英文图注：

> Diego Velázquez, *Las Meninas*, 1656, oil on canvas, 320.3 × 279.1 cm, Museo Nacional del Prado, Madrid, inv. P001174.

### 2.2 本地高清文件的可追溯来源

本项目的主输入文件 `pics/Las_Meninas,_by_Diego_Velázquez,_from_Prado_in_Google_Earth.jpg` 已通过文件尺寸与 SHA-1 校验确认和 Wikimedia Commons 同名原始文件完全一致。本地文件为 26,065 × 30,000 像素、266,867,780 bytes，SHA-1 为 `5ae2431dd56805c8f637819a63c7a36cda0c30bf`；这些值与 Commons 文件页的结构化数据一致。Commons 将来源说明为 “The Prado in Google Earth”。[Wikimedia Commons：高清文件页与结构化数据](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg)

Commons 将该文件说明为公版二维作品的忠实摄影复制，并标注原作及该忠实复制的 public-domain 状态。[Wikimedia Commons：Licensing](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg#Licensing)

建议在代码仓库和最终报告保留以下来源行，即使公版材料在许多司法辖区不强制署名，也能保证研究可追溯性：

> Source image: Diego Velázquez, *Las Meninas* (1656), Museo Nacional del Prado; high-resolution faithful reproduction from “The Prado in Google Earth,” via Wikimedia Commons, public domain.

### 2.3 本地辅助图片的来源状态

| 本地文件 | 当前可验证状态 | 可提交用法 | 不应声称的内容 |
| --- | --- | --- | --- |
| `mirror.png` | 从《宫娥》画面裁出的画家与后墙镜像局部；可由公版高清原图重新生成 | 重新从已注明来源的高清原图裁切，并在图注写明 “detail” | 不应把它当作独立训练样本或独立艺术作品 |
| `perception.png` | 文件本身没有可见署名，当前未核实原始出版物 | 只可作为待核验的人工透视假说参考；正式提交优先使用项目自己从原图检测并绘制的透视叠图 | 不应称其为官方 Prado 分析、精确 ground truth 或 CNN 输出 |
| `location.png` | 文件本身没有可见署名，当前未核实原始出版物 | 只可作为构思关系图的内部参考；正式提交应自行重画并标注哪些关系是项目定义 | 不应称其为已证实的房间实测图或唯一正确的观看位置 |
| `picasso.png` | 画面与 Museu Picasso 馆藏页中的 Picasso 1957 年大型群像版本视觉上相符，但本地文件的下载来源与授权链未建立 | 用作课堂比较时必须附馆藏链接、作品信息和权利说明；公开仓库应避免直接再分发该文件，除非另有许可 | 不应因为主题源于 1656 年原作，就把 Picasso 1957 年作品复制图也视作公版 |

这里的来源缺口是研究记录的一部分。辅助图可以启发实验，但不能在没有出处的情况下升级为证据。

### 2.4 公有仓库的再分发决定

公有仓库排除整个 `pics/`、可重建的 255 MB 原始 JPEG、模型权重与大型 CNN descriptor 张量。仓库保留公版原作的下载/校验脚本、缩小代理图、可复现代码、汇总指标、项目生成图和完整 AI 披露。这个决定既避免把 Picasso 复制图误当公版，也避免让代码采用 MIT License 造成“所有图片与模型均获 MIT 授权”的错误印象。逐文件边界见根目录 `THIRD-PARTY-NOTICES.md`。

## 3. 镜子、透视与观看者：可靠的艺术史依据

### 3.1 画中人物与镜像

Prado 明确识别画中核心人物：Infanta Margarita 与两位 meninas；画家本人站在大画布前；José Nieto 位于背景门口；后墙镜中出现的是 Philip IV 与 Mariana of Austria，馆方描述他们为正在观看现场的 Infanta 的父母。[Museo Nacional del Prado：人物与镜像说明](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

因此，报告可以安全地写：

> 后墙镜中可见 Philip IV 与 Mariana of Austria；镜像让画面引入了并未以实体直接出现在房间前景中的王室人物。[Museo Nacional del Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

报告不应把“镜中一定反射现实空间里的国王王后”与“镜中一定反射画家正在绘制的双人肖像”中的任何一个写成已经由 Prado 解决的事实。Prado 支持的是镜中人物身份以及镜子对观看与再现问题的重要性，而不是本项目所能证明的唯一光路。

### 3.2 空间与透视

Prado 指出，画中空间不只通过科学透视构造，也通过空气透视与多重光源建立；馆方同时将这一场景称为西方绘画中极具可信度的空间之一。[Museo Nacional del Prado：空间说明](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

这支持本项目同时分析两类几何线索：

- 可由边缘与直线检测描述的投影线、门框、画框、天花板接缝与消失方向；
- 不能简化为直线交点的明暗梯度、遮挡、尺度变化与空气感。

“CNN/HED 检出了多少直线”不能替代完整的空间分析，因为馆方对画中空间的解释本身就不只依赖线性透视。[Museo Nacional del Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

### 3.3 观看者角色

Prado 的解释直接指出：镜子的加入使作品成为对“观看行为”的思考，并促使观看者反省再现法则、绘画与现实之间的边界，以及自己在画中的角色。[Museo Nacional del Prado：镜子与观看者](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

这为本项目标题中的 *Entering* 提供了第一方艺术史依据。报告可以说“作品迫使我思考自己在再现结构中的位置”，但“现实观看者与国王王后占据完全相同的物理坐标”应写为待检验的空间解释，而不是馆方确认的测量事实。

### 3.4 可直接采用的艺术史段落

> 《宫娥》并不是一张可以通过人物识别完全解释的群像。Prado 既指出画面通过科学透视、空气透视和多重光源建立可信空间，也强调镜子把作品转化为对观看、再现以及观看者自身角色的思考。因此，本项目把镜子、背景门、画布、遮挡和凝视组织成一套关系问题：CNN 可能检测到这些区域，却未必能说明它们为何共同改变观看者的位置感。[Museo Nacional del Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

## 4. Picasso 的 *Las Meninas*：Transformation 的艺术史对照

### 4.1 馆藏事实

Museu Picasso Barcelona 将本项目所比较的大型版本记录为 Pablo Picasso 的 *Las Meninas*，1957 年，布面油画，194 × 260 cm，馆藏号 MPB 70.433，1968 年由 Picasso 捐赠。[Museu Picasso Barcelona：作品记录](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)

馆方把它说明为 Picasso 在该系列中第一幅大型完整群像。与 Velázquez 的竖幅相比，这一版本改为横幅；原作人物仍被保留，但画家形象被显著放大，José Nieto 仍被安排在门框与透视线汇聚的关键位置，空间与人物则被进一步平面化、简化和重新组织。[Museu Picasso Barcelona：作品分析](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)

馆方还指出，该系列中有十六幅描绘完整构图的作品，Picasso 在这些作品中反复改变空间与人物的处理方式。[Museu Picasso Barcelona：作品分析](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)

### 4.2 对本项目的意义

由上述馆方事实可以提出一个明确但仍需实验检验的推论：Picasso 的版本改变了格式、色彩、线条、人物尺度与空间分层，却保留了足够多的角色和关系，使其仍被博物馆明确归入对 Velázquez *Las Meninas* 的解释系列。[Museu Picasso Barcelona](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)

这使 Picasso 版本成为比随机滤镜更有价值的 transformation 对照：

- 风格与像素距离可以很大；
- 画家、门中人物、人物群和画室结构仍存在可对应关系；
- CNN 是否把这种“关系保留但外观剧变”的图像判断为接近原作，是一个可测量问题，而不是预设结论。

### 4.3 复制图权利边界

Museu Picasso 页面明确标示，Picasso 作品复制图的权利由 Sucesión Picasso / VEGAP 保留。[Museu Picasso Barcelona：页面版权声明](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)

因此：

- 可在报告中链接官方作品页并完整引用作品信息；
- 不能把 `picasso.png` 按照 Velázquez 原作的 public-domain 状态处理；
- 面向课堂的图像引用与面向公开 GitHub 的再分发不是同一使用场景；公开仓库若无明确许可，建议不提交该二进制文件，而在文档中放官方链接和文字分析；
- 本文只是来源与风险记录，不替代学校政策或法律意见。

## 5. CNN 方法的原始依据

### 5.1 VGG：多层视觉特征主干

Simonyan 与 Zisserman 的 VGG 论文系统比较了网络深度，并以连续的 3 × 3 卷积构建最高 16–19 个带权层的网络；VGG-19 对应论文中的 19 层配置。[Simonyan & Zisserman, 2015](https://arxiv.org/abs/1409.1556)

Torchvision 官方文档提供 `vgg19` 及 ImageNet-1K 预训练权重；文档同时给出权重对应的归一化参数与分类默认预处理。[Torchvision：VGG-19 官方文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19)

对本项目而言，VGG 的合理角色是“冻结的视觉测量工具”，而不是针对艺术史训练出的解释器：

- 读取浅层、中层和深层 feature maps；
- 比较同一原作经过受控 transformation 后的表征变化；
- 用固定模型、固定层、固定归一化保证版本之间可比较。

由于官方预训练权重来自 ImageNet-1K，对《宫娥》的分析属于跨域使用。报告应写“VGG feature distance”或“CNN-defined similarity”，不能写成“神经网络客观测得艺术价值”或“VGG 理解了画中人物关系”。这一限制是从模型训练来源推导出的项目方法边界。[Torchvision：VGG-19 weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19)

官方分类权重的默认 transform 包含缩放与中心裁剪；中心裁剪会丢失《宫娥》外围构图，因此几何实验必须记录并采用保持完整画幅和纵横比的预处理，同时沿用权重要求的通道归一化。[Torchvision：VGG-19 transforms](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19)

### 5.2 Gatys 风格迁移：分离“模型定义的内容与风格”

Gatys、Ecker 与 Bethge 的方法使用为物体识别优化的 CNN 表征，把一张图的内容表征与另一张图的外观表征组合起来，生成新的图像。[Gatys, Ecker, & Bethge, 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)

该方法用 CNN 某层 feature maps 的通道相关性构成风格表示。设第 \(l\) 层特征在展平空间维度后为 \(F_l \in \mathbb{R}^{C_l \times H_lW_l}\)，项目使用归一化 Gram matrix：

\[
G_l = \frac{F_lF_l^\top}{C_lH_lW_l}.
\]

PyTorch 官方 neural style tutorial 给出了同一计算方式：将 feature maps 展平、与转置相乘得到 Gram matrix，再按元素数量归一化；教程使用预训练 VGG-19 的卷积特征计算 content loss 与 style loss。[PyTorch：Neural Transfer Using PyTorch](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)

在本项目中，Gram matrix 应被称为“CNN style proxy”或“模型定义的纹理/特征相关性”，原因是这一表征弱化了特征出现的精确位置。Gatys 论文展示的 style reconstruction 正是通过多层特征相关性匹配外观，同时舍弃场景的全局排列。[Gatys, Ecker, & Bethge, 2016：原文 PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

这支持以下实验设计：

- 用 Gram distance 测量 style-only transformation；
- 用仍保留 \(H \times W\) 坐标的 feature maps 测量 spatial/content change；
- 不用 Gram matrix 单独判断消失点、人物相对位置或镜像关系，因为它有意弱化空间排列。

可安全写入报告的结论形式是：“在本实验定义的 VGG 层与 Gram 表征下，版本 A 的 style distance 高于版本 B。”不能把它扩张为“版本 A 在艺术史意义上拥有更多或更少风格”。

### 5.3 HED：由 CNN 提取多尺度边界

Xie 与 Tu 提出的 Holistically-Nested Edge Detection（HED）把逐像素边缘分类转化为 image-to-image prediction，并结合全卷积网络、deep supervision 与多尺度特征学习。[Xie & Tu, 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html)

作者仓库提供论文代码、BSDS 预训练模型和五个尺度的 side outputs，可作为模型与权重来源记录。[Xie：HED 作者仓库](https://github.com/s9xie/hed)

HED 在本项目中的可验证角色是输出“边界概率图”。从边界图继续估计建筑直线、交点和消失方向，需要额外的几何后处理。OpenCV 官方文档说明，`HoughLines` 与 `HoughLinesP` 接收边缘图并检测直线参数或线段端点。[OpenCV：Hough Line Transform](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html)

因此最终报告必须把处理链写清楚：

> HED CNN → boundary probability map → threshold/edge selection → Hough line detection → candidate vanishing directions → human verification against the painting.

安全的表述是“CNN 提取边界，Hough 变换从这些边界估计候选直线”。不安全的表述是“HED 自己理解了房间透视”或“HED 证明了唯一正确的消失点”。画作中的笔触、人物轮廓、画框和明暗交界都会产生强边缘，故检测结果必须与人工标注和传统 Canny + Hough 对照一起解释。

### 5.4 CNN 的 texture bias：项目假设的实验依据

Geirhos 等人使用 shape–texture cue-conflict 图像比较人类与多种 ImageNet 预训练 CNN。他们报告，在该实验条件下，VGG-16 等网络更常按纹理线索而非整体形状作答；同一研究也显示，改变训练数据为 Stylized-ImageNet 后，同一类架构可以显著提高 shape bias。[Geirhos et al., 2019](https://arxiv.org/abs/1811.12231)

这项证据支持本项目提出而不是预先宣布以下假设：当一个 transformation 保留局部颜色/纹理却破坏镜子、门、人物位置或整体构图时，ImageNet 预训练 CNN 的相似度判断可能与人类判断分离。[Geirhos et al., 2019：ICLR record](https://openreview.net/forum?id=Bygh9j09KX)

必须保留三个限定：

1. 论文结论针对特定训练分布、网络和 cue-conflict 测试，不等于“所有 CNN 天生只看纹理”；
2. 论文显示训练数据可以改变 shape/texture bias，因此 bias 不是仅由 CNN 架构名称决定的常数；
3. 本项目如果没有收集人类判断数据，只能展示 CNN 的模型内部距离，不能声称已经证明“人类与 CNN 分歧”。

### 5.5 方法—输出—主张边界

| 方法 | 项目输出 | 可以主张 | 不能主张 |
| --- | --- | --- | --- |
| VGG spatial feature maps | 各层 activation、content/spatial distance | 固定 VGG 表征对某种 transformation 的敏感度 | 大脑视觉皮层的真实反应或作品的客观意义 |
| VGG Gram matrices | 多层 style distance、style trajectory | 模型定义的特征相关性或纹理统计变化 | 完整艺术史风格已被提取或理解 |
| Gatys optimization | 保持 content loss、改变 style loss 的生成图 | 在给定损失与权重下完成了可复现的 style transformation | 生成图只改变风格而绝对不改变几何；必须用 landmark/edge 指标验证 |
| HED | 边界概率图、side/fused edges | CNN 对多尺度视觉边界的响应 | 透视、镜像或人物身份已被理解 |
| Hough on HED/Canny | 直线与候选汇聚方向 | 给定阈值下检测到的几何线索 | 由算法证明画家采用了某一种历史绘制技术 |
| 人工关系图 | mirror/reflection、contains、occludes、looks-toward 等节点与边 | 本项目明确声明的操作性编码 | CNN 自动发现了全部关系，除非另有独立输出与验证 |

## 6. “Cognitive map”与“relational topology”的谨慎表述

### 6.1 认知地图的原始语境

Tolman 的经典论文 *Cognitive Maps in Rats and Men* 基于迷宫、潜伏学习、空间定向等行为实验讨论动物如何形成不只是单一刺激—反应路径的空间组织。[Tolman, 1948](https://doi.org/10.1037/h0061626)

O’Keefe 与 Dostrovsky 随后在自由活动大鼠的海马单元活动中报告了与动物所处位置有关的初步神经证据；这项工作依赖颅内单元记录，而不是对图片运行 CNN。[O’Keefe & Dostrovsky, 1971](https://doi.org/10.1016/0006-8993(71)90358-1)

由此，本项目可以把观看《宫娥》时对人物、镜子、门和画外位置的组织称为“受 cognitive-map 概念启发的场景关系图”，但不能称其为对观看者海马或真实神经认知地图的测量。本项目没有神经记录，也没有建立从 VGG activation 到人脑空间表征的验证链。

### 6.2 本项目的操作性定义

本项目把三个层次明确分开：

- **Style**：颜色、纹理、局部笔触与多层 CNN feature correlations；
- **Geometry**：坐标、尺度、角度、透视方向、遮挡边界和相对深度；
- **Relational topology**：不依赖精确距离的定性关系，例如 `mirror reflects royal couple`、`door contains José Nieto`、`large canvas occludes left room`、`figure looks toward implied outside position`。

这里的 “relational topology” 是项目自定的图结构编码，不是对数学拓扑不变量的完整证明，也不是“神经拓扑”。它的价值在于让每次 transformation 都能回答一个可审计问题：节点是否仍存在、边是否被保留、哪一条关系被有意切断。

### 6.3 可用与不可用表述

可用：

> 本项目借用 cognitive map 的组织性思想，把画中与画外的关键元素编码为一个关系图；该图是分析工具，不是脑活动测量。[Tolman, 1948](https://doi.org/10.1037/h0061626)

> CNN 提供 feature maps、Gram matrices 与边界图；镜像、凝视和画外人物等高层关系由明确的人工规则标注并接受视觉核验。

不可用：

> VGG 的 feature space 就是人类的 cognitive map。

> HED 检出的边缘证明了观看者大脑如何理解《宫娥》。

> PCA/UMAP 的二维图展示了人脑真实的拓扑结构。

## 7. 最终报告的主张—来源对照表

| 报告中的主张 | 证据等级 | 紧邻引用 | 写作限制 |
| --- | --- | --- | --- |
| 《宫娥》为 Velázquez 1656 年布面油画，现藏 Prado | 馆藏第一方 | [Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f) | 使用馆方尺寸与馆藏号 |
| 镜中人物为 Philip IV 与 Mariana | 馆藏第一方解释 | [Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f) | 不进一步断言唯一光路 |
| 空间由科学透视、空气透视和多光源共同建立 | 馆藏第一方解释 | [Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f) | 不能把空间分析缩成单一消失点 |
| 镜子引导观看者思考观看、再现与自身角色 | 馆藏第一方解释 | [Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f) | 可支撑个人叙事，不等于精确坐标测量 |
| Picasso 1957 版本改变画幅、中心与空间处理，但保留原作角色 | 收藏机构第一方 | [Museu Picasso](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9) | “仍可认出”的人类判断若要量化，需用户实验 |
| VGG 可作为预训练多层 CNN feature extractor | 原始论文 + 官方实现 | [VGG paper](https://arxiv.org/abs/1409.1556)；[Torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19) | 明确 ImageNet 训练来源和预处理 |
| Gram matrix 可作 CNN-defined style representation | 原始论文 + 官方教程 | [Gatys et al.](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)；[PyTorch tutorial](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html) | 不等同完整艺术风格；不用于空间定位 |
| HED 用全卷积、多尺度、深监督进行边缘预测 | 原始论文 | [Xie & Tu](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html) | HED 输出边界，不直接输出艺术史解释 |
| Hough 可从边缘图提取候选直线 | 官方文档 | [OpenCV](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html) | 记录阈值，并用 Canny 与人工叠图复核 |
| 部分 ImageNet CNN 在 cue-conflict 条件下呈现 texture bias | 原始论文 | [Geirhos et al.](https://arxiv.org/abs/1811.12231) | 不泛化为所有 CNN 的先天属性 |
| cognitive map 源于空间学习与组织问题 | 原始论文 | [Tolman](https://doi.org/10.1037/h0061626) | 本项目只作启发性、操作性借用 |

## 8. 可直接放入最终报告的方法依据段落

### 8.1 为什么使用 CNN，而不是只用 Photoshop

> 本项目的 transformation 不只是视觉滤镜。冻结的 VGG-19 为每个版本生成多层 feature maps；保留空间坐标的 feature maps 用于比较构图变化，而 Gram matrices 用于比较模型定义的纹理与风格相关性。Gatys 等人的方法证明了 CNN 表征可以在优化过程中分离并重新组合 content 与 style，但这种“style”是模型中的特征相关性，不等同于完整的艺术史风格。[Gatys, Ecker, & Bethge, 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) [PyTorch neural style tutorial](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)

### 8.2 为什么边缘不等于几何理解

> HED 以全卷积、深监督网络产生多尺度边界预测；本项目再用 Hough 变换从边缘图中寻找候选直线。这个流程能把“CNN 提取视觉边界”和“几何算法估计直线”清楚分开，但它不能单独判断镜中人物是谁，也不能证明某一条线在艺术史上的意义。[Xie & Tu, 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html) [OpenCV Hough documentation](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html)

### 8.3 为什么比较 Picasso

> Picasso 的 1957 年大型版本不是简单复制：Museu Picasso 指出，他把 Velázquez 的竖幅改为横幅，放大画家、重组空间并平面化人物，同时保留原作角色和门中人物等关系。它因此成为测试“外观差异很大但关系仍可对应”这一条件的艺术史对照。[Museu Picasso Barcelona](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)

### 8.4 为什么称 CNN 为“不完整的观看者”

> Prado 对《宫娥》的解释同时强调透视、光线、镜像与观看者角色；CNN 可以输出 feature maps、风格统计和边界，但镜像如何连接画内与画外、观看者为何被卷入再现结构，并不会自动出现在这些数值中。因此，“不完整”不是模型运行失败的借口，而是本项目要通过受控 transformation 明确展示的表征边界。[Museo Nacional del Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

## 9. References

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *Proceedings of the 7th International Conference on Learning Representations (ICLR)*. [https://openreview.net/forum?id=Bygh9j09KX](https://openreview.net/forum?id=Bygh9j09KX)

Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. *arXiv preprint arXiv:1508.06576*. [https://arxiv.org/abs/1508.06576](https://arxiv.org/abs/1508.06576)

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2414–2423). [https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)

Museo Nacional del Prado. (n.d.). *The Family of Felipe IV, or Las Meninas*. Collection record, inventory P001174. Accessed July 17, 2026. [https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

Museu Picasso Barcelona. (n.d.). *Las Meninas*. Collection record, inventory MPB 70.433. Accessed July 17, 2026. [https://museupicassobcn.cat/en/collection/artwork/las-meninas-9](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)

O’Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map: Preliminary evidence from unit activity in the freely-moving rat. *Brain Research, 34*(1), 171–175. [https://doi.org/10.1016/0006-8993(71)90358-1](https://doi.org/10.1016/0006-8993(71)90358-1)

OpenCV. (n.d.). *Hough Line Transform*. OpenCV documentation. Accessed July 17, 2026. [https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html)

Portús Pérez, J. (2013). Diego Velázquez, *Las Meninas*. In *Velázquez y la familia de Felipe IV [1650–1680]* (pp. 126–129, no. 16). Museo Nacional del Prado. Collection essay available at [https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)

PyTorch. (n.d.). *Neural Transfer Using PyTorch*. PyTorch Tutorials. Accessed July 17, 2026. [https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*. [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

Tolman, E. C. (1948). Cognitive maps in rats and men. *Psychological Review, 55*(4), 189–208. [https://doi.org/10.1037/h0061626](https://doi.org/10.1037/h0061626)

Torchvision Maintainers and Contributors. (n.d.). *VGG19*. Torchvision model documentation. Accessed July 17, 2026. [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19)

Wikimedia Commons contributors. (n.d.). *File: Las Meninas, by Diego Velázquez, from Prado in Google Earth.jpg*. Accessed July 17, 2026. [https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg)

Xie, S., & Tu, Z. (2015). Holistically-nested edge detection. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 1395–1403). [https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html)

Xie, S. (2015). *Code for Holistically-Nested Edge Detection* [Source code and pretrained model]. GitHub. [https://github.com/s9xie/hed](https://github.com/s9xie/hed)
