# 项目生成视觉素材

本目录中的图片由项目作者为本项目编写提示词，并使用 Codex 内置图像生成工具完成。它们不替代 CNN 实验结果，而分别承担风格输入和装置概念展示的角色。生成日期为 2026-07-17；工具未返回可记录的模型版本或随机种子，因此提示词可复核创作意图，但不能保证逐像素重现。

提示词中的 “original” 是对生成内容的创作约束，不是独创性或版权状态的法律认证。这些图片不在代码的 MIT License 范围内；本仓库不通过 `LICENSE` 授予通用图片再利用许可，也不主张任何特定 AI 输出必然在所有司法辖区可受版权保护。详见 [`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md)。

## `style/cognitive-map-style-reference.png`

用途：公有仓库中使用的 neural style transfer 参考图。图像把透视线、嵌套画框、镜面焦点和节点关系组织成项目生成的抽象油画，不以模仿任何具名艺术家为提示目标。

最终提示词：

```text
Use case: stylized-concept
Asset type: original public-project style-reference image for CNN neural style transfer
Primary request: create a square abstract painterly texture that visualizes a cognitive map of a room being separated into style, geometry, and relational topology
Scene/backdrop: no literal room and no recognizable artwork; an abstract field built from receding perspective lines, nested frames, a small mirror-like luminous rectangle, and subtle node-and-edge relationships
Style/medium: original mixed-media oil paint, charcoal construction lines, scraped gesso, translucent geometric planes; sophisticated museum-grade contemporary abstraction; not in the style of any named artist
Composition/framing: square, edge-to-edge texture with balanced detail at every scale; one quiet luminous focal rectangle slightly above center; no horizon photograph and no identifiable human figure
Lighting/mood: contemplative, analytical, mysterious
Color palette: warm umber, bone white, lamp black, muted ultramarine, restrained oxidized red
Materials/textures: visible dry brush, impasto ridges, charcoal dust, thin glazing, canvas grain
Constraints: completely original; suitable as a style donor; no text, no letters, no logos, no watermark, no copyrighted characters, no direct reproduction of Las Meninas
Avoid: generic AI glow, neon cyberpunk, photorealism, ornate baroque figures, symmetrical mandala, infographic labels
```

SHA-256：`d8adc658b8a303847ab5a823f0c2c20852c7520bc4b9f8511314396d37708e4f`

## `installation/the-missing-viewer-mockup.png`

用途：ready-made *The Missing Viewer* 的 AI-generated installation mockup。它说明镜子应如何把现实观看者放到《宫娥》的画外观看轴线上，不声称装置已经真实搭建。

最终提示词：

```text
Use case: stylized-concept
Asset type: AI-generated installation mockup for an academic art project
Primary request: visualize a conceptual ready-made installation titled The Missing Viewer, where an ordinary unmodified rectangular wall mirror places a present-day museum visitor into the implied off-canvas viewing position of Velázquez's Las Meninas
Scene/backdrop: quiet contemporary gallery with charcoal walls; a large public-domain reproduction of Las Meninas is projected or printed on the wall; a plain freestanding mirror is precisely aligned in front of the painting so its reflection visually occupies the painting's small rear-mirror relationship
Subject: one anonymous visitor seen only from behind at a respectful distance, with their reflection subtle in the ordinary mirror
Style/medium: highly realistic architectural exhibition visualization, honest concept mockup rather than documentary photography
Composition/framing: wide 16:9 landscape, strong central perspective, painting on the left-center and physical mirror aligned along the viewer axis, generous dark negative space
Lighting/mood: restrained museum lighting, contemplative, intellectually precise
Color palette: charcoal, warm umber, bone white, small muted red accents
Materials/textures: matte gallery wall, real mirror glass, simple dark wood frame, polished concrete floor
Constraints: no labels, no title text, no logos, no watermark; mirror must look like an ordinary found object; installation must be physically plausible; clearly contemporary gallery context
Avoid: surreal floating objects, ornate fantasy frame, crowds, neon lighting, decorative UI, misleading news-photo aesthetics
```

SHA-256：`deb8b95d987f664a9b3b3d69de10c52ba295e00207d6aea47ce42a504501a261`
