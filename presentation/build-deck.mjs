import fs from "node:fs/promises";
import path from "node:path";
import { image, layers, shape, text, Presentation, PresentationFile } from "@oai/artifact-tool";

const SLIDE_WIDTH = 1280;
const SLIDE_HEIGHT = 720;

const COLORS = {
  ink: "#0D1011",
  inkSoft: "#171B1D",
  ivory: "#F4EFE5",
  ivorySoft: "#E8E0D2",
  warmGray: "#B8AEA0",
  gray: "#6E7377",
  red: "#B83A2D",
  redSoft: "#D86E5B",
  cyan: "#69D2E7",
  cyanSoft: "#A9E8F2",
  white: "#FFFFFF",
};

const FONTS = {
  sans: "Helvetica Neue",
  serif: "Georgia",
};

function mimeType(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".jpg" || extension === ".jpeg") return "image/jpeg";
  if (extension === ".webp") return "image/webp";
  return "image/png";
}

async function loadImage(projectRoot, relativePath) {
  const absolutePath = path.join(projectRoot, relativePath);
  return {
    bytes: new Uint8Array(await fs.readFile(absolutePath)),
    contentType: mimeType(absolutePath),
    relativePath,
  };
}

function rectangle(name, left, top, width, height, fill, options = {}) {
  return shape({
    name,
    geometry: options.geometry ?? "rect",
    fill,
    ...(options.line
      ? { line: options.line }
      : { line: { style: "solid", width: 0, fill } }),
    position: { left, top },
    width,
    height,
    ...(options.borderRadius ? { borderRadius: options.borderRadius } : {}),
  });
}

function line(name, left, top, width, height, fill, weight = 2) {
  return shape({
    name,
    geometry: "straightConnector1",
    fill: "none",
    line: { style: "solid", width: weight, fill },
    position: { left, top },
    width: Math.max(width, 0.03),
    height: Math.max(height, 0.03),
  });
}

function labelText(name, value, left, top, width, height, options = {}) {
  return text([value], {
    name,
    position: { left, top },
    width,
    height,
    style: {
      fontSize: `${options.fontSize ?? 20}px`,
      typeface: options.typeface ?? FONTS.sans,
      color: options.color ?? COLORS.ink,
      alignment: options.alignment ?? "left",
      verticalAlignment: options.verticalAlignment ?? "top",
      autoFit: options.autoFit ?? "shrinkText",
      wrap: "square",
      bold: options.bold ?? false,
      italic: options.italic ?? false,
      insets: options.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
    },
  });
}

function assetImage(name, asset, left, top, width, height, options = {}) {
  return image({
    name,
    blob: asset.bytes,
    contentType: asset.contentType,
    alt: options.alt ?? name,
    fit: options.fit ?? "cover",
    geometry: options.geometry ?? "rect",
    position: { left, top },
    width,
    height,
    ...(options.crop ? { crop: options.crop } : {}),
    ...(options.borderRadius ? { borderRadius: options.borderRadius } : {}),
  });
}

function footer(pageNumber, source, dark = true) {
  const color = dark ? COLORS.warmGray : COLORS.gray;
  return [
    line(`footer-rule-${pageNumber}`, 44, 680, 1192, 0, dark ? "#33383B" : "#CDC3B4", 1),
    labelText(`footer-source-${pageNumber}`, source, 44, 688, 1120, 16, {
      fontSize: 10,
      color,
      autoFit: "shrinkText",
    }),
    labelText(`footer-page-${pageNumber}`, String(pageNumber).padStart(2, "0"), 1178, 688, 58, 16, {
      fontSize: 10,
      color,
      alignment: "right",
      autoFit: "none",
    }),
  ];
}

function header(pageNumber, title, kicker, dark = true) {
  const titleColor = dark ? COLORS.ivory : COLORS.ink;
  const kickerColor = dark ? COLORS.cyan : COLORS.red;
  return [
    labelText(`kicker-${pageNumber}`, kicker.toUpperCase(), 44, 30, 530, 22, {
      fontSize: 13,
      color: kickerColor,
      bold: true,
      autoFit: "none",
    }),
    labelText(`slide-title-${pageNumber}`, title, 44, 58, 1192, 52, {
      fontSize: 38,
      color: titleColor,
      bold: true,
      autoFit: "shrinkText",
    }),
  ];
}

function createSlide(presentation, name, background, nodes) {
  const slide = presentation.slides.add();
  slide.background.fill = background;
  slide.compose(
    layers({ name, width: "fill", height: "fill" }, nodes),
    {
      frame: { left: 0, top: 0, width: SLIDE_WIDTH, height: SLIDE_HEIGHT },
      baseUnit: 1,
    },
  );
  return slide;
}

function panel(name, left, top, width, height, fill, lineColor = null) {
  return rectangle(name, left, top, width, height, fill, {
    line: lineColor
      ? { style: "solid", width: 1, fill: lineColor }
      : { style: "solid", width: 0, fill },
  });
}

async function buildPresentation(projectRoot) {
  const assetPaths = {
    original: "outputs/reference/las-meninas-reference.jpg",
    matrix: "outputs/figures/controlled-neural-matrix.png",
    activationLayers: "outputs/figures/vgg-activation-layers.png",
    trajectories: "outputs/metrics/transformation-trajectories.png",
    neuralStyleSequence: "outputs/figures/neural-style-sequence.png",
    geometrySequence: "outputs/figures/geometry-sequence.png",
    topologyGraph: "outputs/figures/mirror-deletion-difference.png",
    topologyControls: "outputs/figures/topology-controls.png",
    geometryMethod: "outputs/figures/geometry-method.png",
    mockup: "assets/installation/the-missing-viewer-mockup.png",
    finalArtwork: "outputs/artwork/entering-las-meninas-final.png",
    relationGraph: "outputs/figures/las-meninas-relations.png",
  };

  const loadedEntries = await Promise.all(
    Object.entries(assetPaths).map(async ([key, relativePath]) => [key, await loadImage(projectRoot, relativePath)]),
  );
  const assets = Object.fromEntries(loadedEntries);

  const presentation = Presentation.create({
    slideSize: { width: SLIDE_WIDTH, height: SLIDE_HEIGHT },
  });

  // 01 — Codex Grid slide-08 silhouette: a restrained text field beside one dominant image.
  createSlide(presentation, "01-cover-image-field", COLORS.ink, [
    assetImage("cover-painting", assets.original, 694, 0, 586, 720, {
      fit: "cover",
      crop: { left: 0.04, top: 0, right: 0.04, bottom: 0 },
      alt: "Public-domain reproduction of Diego Velázquez's Las Meninas",
    }),
    rectangle("cover-red-rail", 0, 0, 12, 720, COLORS.red),
    labelText("cover-course", "ART, GEOMETRY, AND COGNITION · FINAL PROJECT", 54, 52, 570, 24, {
      fontSize: 13,
      color: COLORS.cyan,
      bold: true,
      autoFit: "none",
    }),
    labelText("cover-title", "Entering\nLas Meninas", 54, 144, 590, 178, {
      fontSize: 62,
      typeface: FONTS.serif,
      color: COLORS.ivory,
      bold: true,
      autoFit: "shrinkText",
    }),
    line("cover-rule", 54, 348, 120, 0, COLORS.red, 4),
    labelText("cover-subtitle", "How a Painting Changed\nthe Way I Look at Art", 54, 380, 520, 100, {
      fontSize: 28,
      color: COLORS.ivorySoft,
      autoFit: "shrinkText",
    }),
    labelText("cover-method", "A controlled CNN study of style, geometry, and relational topology", 54, 540, 520, 58, {
      fontSize: 18,
      color: COLORS.warmGray,
      autoFit: "shrinkText",
    }),
    ...footer(
      1,
      "Image: Diego Velázquez, Las Meninas (1656), Prado; public-domain reproduction via Wikimedia Commons · AI use disclosed in repository",
      true,
    ),
  ]);

  // 02 — A single personal claim, grounded only in the work completed for this project.
  createSlide(presentation, "02-personal-shift", COLORS.ivory, [
    assetImage("shift-original", assets.original, 44, 126, 446, 508, {
      fit: "cover",
      crop: { left: 0.05, top: 0.12, right: 0.02, bottom: 0.03 },
      alt: "Las Meninas cropped to foreground figures, mirror, and rear doorway",
    }),
    panel("shift-quote-field", 536, 126, 700, 508, COLORS.ink),
    labelText("shift-quote-mark", "“", 576, 142, 80, 84, {
      fontSize: 82,
      typeface: FONTS.serif,
      color: COLORS.redSoft,
      autoFit: "none",
    }),
    labelText(
      "shift-quote",
      "I began with an image on a screen.\n\nThe project forced me to define the position from which the painting looks back.",
      586,
      218,
      592,
      250,
      { fontSize: 34, typeface: FONTS.serif, color: COLORS.ivory, autoFit: "shrinkText" },
    ),
    labelText("shift-detail", "The mirror, painter, canvas, Infanta, and doorway became relations—not a list of objects.", 586, 520, 580, 66, {
      fontSize: 19,
      color: COLORS.cyanSoft,
      autoFit: "shrinkText",
    }),
    ...header(2, "The painting changed when I stopped treating it as a surface", "A personal shift", false),
    ...footer(2, "Interpretation grounded in this project · Art-historical context: Museo Nacional del Prado", false),
  ]);

  // 03 — Sparse thesis slide, adapted from Codex Grid slide-01/26 hierarchy.
  createSlide(presentation, "03-question-thesis", COLORS.ink, [
    rectangle("thesis-cyan-rail", 44, 164, 6, 326, COLORS.cyan),
    labelText("thesis-question-label", "RESEARCH QUESTION", 76, 162, 300, 24, {
      fontSize: 13,
      color: COLORS.cyan,
      bold: true,
      autoFit: "none",
    }),
    labelText(
      "thesis-question",
      "How far can a CNN separate style, geometry, and relational topology in Las Meninas?",
      76,
      205,
      1080,
      132,
      { fontSize: 46, typeface: FONTS.serif, color: COLORS.ivory, bold: true, autoFit: "shrinkText" },
    ),
    labelText("thesis-label", "THESIS", 76, 392, 180, 24, {
      fontSize: 13,
      color: COLORS.redSoft,
      bold: true,
      autoFit: "none",
    }),
    labelText(
      "thesis-copy",
      "The CNN is useful precisely as an incomplete viewer: it measures visual change, then exposes the relations its distances do not explain.",
      76,
      430,
      1080,
      102,
      { fontSize: 30, color: COLORS.ivorySoft, autoFit: "shrinkText" },
    ),
    ...footer(3, "Project research question and thesis · No human-participant study was conducted", true),
  ]);

  // 04 — Codex Grid slide-06 three-column ownership, recolored for the museum system.
  createSlide(presentation, "04-three-layer-framework", COLORS.ivory, [
    ...header(4, "One painting can be tested at three different layers", "Operational framework", false),
    panel("framework-style", 44, 158, 374, 448, COLORS.ink),
    panel("framework-geometry", 453, 158, 374, 448, COLORS.red),
    panel("framework-topology", 862, 158, 374, 448, "#D8EEF1"),
    labelText("framework-style-index", "01", 72, 184, 80, 42, { fontSize: 32, color: COLORS.cyan, bold: true }),
    labelText("framework-style-title", "STYLE", 72, 246, 280, 44, { fontSize: 34, color: COLORS.ivory, bold: true }),
    labelText("framework-style-copy", "Color · texture · local mark\n\nVGG19 Gram matrices summarize channel co-activation while weakening exact position.", 72, 316, 300, 170, {
      fontSize: 20,
      color: COLORS.ivorySoft,
    }),
    labelText("framework-style-test", "Test: Does appearance change while composition remains recognizable?", 72, 520, 304, 56, {
      fontSize: 17,
      color: COLORS.cyanSoft,
      bold: true,
    }),
    labelText("framework-geometry-index", "02", 481, 184, 80, 42, { fontSize: 32, color: COLORS.ivory, bold: true }),
    labelText("framework-geometry-title", "GEOMETRY", 481, 246, 310, 44, { fontSize: 34, color: COLORS.ivory, bold: true }),
    labelText("framework-geometry-copy", "Coordinate · scale · angle\n\nSpatial VGG maps, HED boundaries, Hough segments, and RANSAC test layout and convergence.", 481, 316, 300, 170, {
      fontSize: 20,
      color: COLORS.ivory,
    }),
    labelText("framework-geometry-test", "Test: Does structure move when texture is comparatively stable?", 481, 520, 304, 56, {
      fontSize: 17,
      color: COLORS.ivory,
      bold: true,
    }),
    labelText("framework-topology-index", "03", 890, 184, 80, 42, { fontSize: 32, color: COLORS.red, bold: true }),
    labelText("framework-topology-title", "TOPOLOGY", 890, 246, 310, 44, { fontSize: 34, color: COLORS.ink, bold: true }),
    labelText("framework-topology-copy", "Reflects · contains · occludes\n\nFive qualitative relations are declared, edited, and manually audited—not inferred by VGG.", 890, 316, 300, 170, {
      fontSize: 20,
      color: COLORS.ink,
    }),
    labelText("framework-topology-test", "Test: Can meaning change while CNN distance stays small?", 890, 520, 304, 56, {
      fontSize: 17,
      color: COLORS.red,
      bold: true,
    }),
    ...footer(4, "Project definitions · “Relational topology” is an operational scene graph, not a brain measurement", false),
  ]);

  // 05 — Image-led 2×2 experiment, following Codex Grid slide-08 proportions.
  createSlide(presentation, "05-experiment-matrix", COLORS.ink, [
    ...header(5, "A 2 × 2 design separates style from geometry", "Controlled experiment", true),
    panel("matrix-frame", 44, 124, 748, 526, COLORS.ivory),
    assetImage("matrix-figure", assets.matrix, 62, 134, 712, 508, {
      fit: "contain",
      alt: "Controlled neural transformation matrix crossing geometry and style",
    }),
    labelText("matrix-g0s0", "G0S0", 850, 158, 120, 38, { fontSize: 28, color: COLORS.ivory, bold: true }),
    labelText("matrix-g0s0-copy", "Original working proxy", 970, 164, 230, 34, { fontSize: 18, color: COLORS.warmGray }),
    line("matrix-rule-1", 850, 210, 350, 0, "#343A3D", 1),
    labelText("matrix-g0s1", "G0S1", 850, 232, 120, 38, { fontSize: 28, color: COLORS.cyan, bold: true }),
    labelText("matrix-g0s1-copy", "Neural style on original geometry", 970, 238, 230, 46, { fontSize: 18, color: COLORS.ivorySoft }),
    line("matrix-rule-2", 850, 300, 350, 0, "#343A3D", 1),
    labelText("matrix-g1s0", "G1S0", 850, 322, 120, 38, { fontSize: 28, color: COLORS.redSoft, bold: true }),
    labelText("matrix-g1s0-copy", "Perspective geometry transformed", 970, 328, 230, 46, { fontSize: 18, color: COLORS.ivorySoft }),
    line("matrix-rule-3", 850, 390, 350, 0, "#343A3D", 1),
    labelText("matrix-g1s1", "G1S1", 850, 412, 120, 38, { fontSize: 28, color: COLORS.ivory, bold: true }),
    labelText("matrix-g1s1-copy", "Geometry + neural style", 970, 418, 230, 46, { fontSize: 18, color: COLORS.ivorySoft }),
    labelText("matrix-trajectories", "Five registered intensities—0, 25, 50, 75, 100%—turn each endpoint into a trajectory.", 850, 520, 350, 90, {
      fontSize: 19,
      color: COLORS.cyanSoft,
      bold: true,
    }),
    ...footer(5, "Project-generated matrix · Same 1,779 × 2,048 working proxy and frozen preprocessing across conditions", true),
  ]);

  // 06 — Method flow, adapted from Codex Grid slide-17 timeline.
  createSlide(presentation, "06-vgg-hed-method", COLORS.ivory, [
    ...header(6, "The neural networks are instruments inside a larger measurement chain", "Method", false),
    line("method-main-line", 88, 300, 1100, 0, COLORS.ink, 2),
    rectangle("method-node-1", 82, 293, 14, 14, COLORS.red, { geometry: "ellipse" }),
    rectangle("method-node-2", 420, 293, 14, 14, COLORS.cyan, { geometry: "ellipse" }),
    rectangle("method-node-3", 762, 293, 14, 14, COLORS.red, { geometry: "ellipse" }),
    rectangle("method-node-4", 1100, 293, 14, 14, COLORS.cyan, { geometry: "ellipse" }),
    labelText("method-label-1", "INPUT", 82, 242, 180, 28, { fontSize: 14, color: COLORS.red, bold: true }),
    labelText("method-label-2", "VGG19", 420, 242, 180, 28, { fontSize: 14, color: COLORS.cyan, bold: true }),
    labelText("method-label-3", "HED", 762, 242, 180, 28, { fontSize: 14, color: COLORS.red, bold: true }),
    labelText("method-label-4", "AUDIT", 1100, 242, 120, 28, { fontSize: 14, color: COLORS.cyan, bold: true }),
    labelText("method-title-1", "Fixed image", 82, 338, 220, 36, { fontSize: 25, color: COLORS.ink, bold: true }),
    labelText("method-copy-1", "Aspect-preserving\n512 × 512 letterbox\nImageNet normalization", 82, 388, 250, 104, { fontSize: 18, color: COLORS.gray }),
    labelText("method-title-2", "Two descriptors", 420, 338, 250, 36, { fontSize: 25, color: COLORS.ink, bold: true }),
    labelText("method-copy-2", "Gram correlations → style\nSpatial feature maps → layout\nFrozen ImageNet-1K weights", 420, 388, 274, 104, { fontSize: 18, color: COLORS.gray }),
    labelText("method-title-3", "Boundary evidence", 762, 338, 280, 36, { fontSize: 25, color: COLORS.ink, bold: true }),
    labelText("method-copy-3", "HED probability → threshold\nHough line segments\nRANSAC intersection candidate", 762, 388, 292, 104, { fontSize: 18, color: COLORS.gray }),
    labelText("method-title-4", "Human meaning", 1100, 338, 132, 64, { fontSize: 25, color: COLORS.ink, bold: true }),
    labelText("method-copy-4", "Five relations\nManual check\nNo observer sample", 1100, 418, 132, 92, { fontSize: 18, color: COLORS.gray }),
    panel("method-note", 82, 540, 1070, 70, "#DDEFF1", COLORS.cyan),
    labelText("method-note-copy", "CNN output is never labeled “understanding”: VGG measures representational distance; HED measures boundary probability.", 108, 561, 1018, 34, {
      fontSize: 20,
      color: COLORS.ink,
      bold: true,
    }),
    ...footer(6, "Methods: Simonyan & Zisserman (VGG19); Gatys et al. (Gram style); Xie & Tu (HED); OpenCV Hough", false),
  ]);

  // 07 — Full-width visual evidence.
  createSlide(presentation, "07-activation-layers", COLORS.ink, [
    ...header(7, "Deep layers trade crisp edges for coarse scene structure", "Inside VGG19", true),
    panel("activation-frame", 44, 140, 1192, 350, COLORS.ivory),
    assetImage("activation-figure", assets.activationLayers, 54, 158, 1172, 312, {
      fit: "contain",
      alt: "VGG19 activation energy from CNN input through relu5_1",
    }),
    labelText("activation-shallow", "SHALLOW", 70, 528, 160, 22, { fontSize: 13, color: COLORS.cyan, bold: true }),
    labelText("activation-shallow-copy", "Frames, seams, faces, and dress boundaries remain locally sharp.", 70, 558, 330, 56, {
      fontSize: 19,
      color: COLORS.ivorySoft,
    }),
    line("activation-arrow", 432, 570, 290, 0, COLORS.redSoft, 3),
    labelText("activation-deep", "DEEP", 770, 528, 160, 22, { fontSize: 13, color: COLORS.redSoft, bold: true }),
    labelText("activation-deep-copy", "Spatial detail pools into broad figure and doorway responses—but relations are still not explicit edges in a graph.", 770, 558, 410, 66, {
      fontSize: 19,
      color: COLORS.ivorySoft,
    }),
    ...footer(7, "Project visualization of frozen Torchvision VGG19 activations · Heatmaps show channel-mean activation energy", true),
  ]);

  // 08 — Chart-led evidence, based on Codex Grid slide-21 balance.
  createSlide(presentation, "08-deterministic-trajectories", COLORS.ivory, [
    ...header(8, "The controls separate direction in feature space—not perfectly, but clearly", "Deterministic trajectories", false),
    assetImage("trajectory-chart", assets.trajectories, 42, 128, 900, 500, {
      fit: "contain",
      alt: "Two charts comparing deterministic style and geometry transformations",
    }),
    panel("trajectory-stat-1", 972, 148, 264, 150, COLORS.ink),
    labelText("trajectory-stat-1-value", "0.303", 998, 170, 210, 60, { fontSize: 46, color: COLORS.cyan, bold: true }),
    labelText("trajectory-stat-1-copy", "style distance\nfor style-100", 998, 240, 210, 44, { fontSize: 17, color: COLORS.ivorySoft }),
    panel("trajectory-stat-2", 972, 318, 264, 150, COLORS.red),
    labelText("trajectory-stat-2-value", "0.919", 998, 340, 210, 60, { fontSize: 46, color: COLORS.ivory, bold: true }),
    labelText("trajectory-stat-2-copy", "spatial relative RMS\nfor geometry-100", 998, 410, 210, 44, { fontSize: 17, color: COLORS.ivory }),
    labelText("trajectory-meaning", "Style changes mostly along the Gram axis; geometry changes dominate the spatial axis.", 972, 510, 264, 110, {
      fontSize: 20,
      color: COLORS.ink,
      bold: true,
    }),
    ...footer(8, "Raw VGG19 aggregate distances from identical 0% baseline · Style and spatial metrics are not merged into one score", false),
  ]);

  // 09 — Neural-style sequence with concise metric rail.
  createSlide(presentation, "09-neural-style-series", COLORS.ink, [
    ...header(9, "Optimization changes CNN style as content loss rises", "Neural style transfer", true),
    panel("nst-frame", 44, 140, 1192, 302, COLORS.ivory),
    assetImage("nst-sequence", assets.neuralStyleSequence, 54, 152, 1172, 276, {
      fit: "contain",
      alt: "Las Meninas neural style transfer sequence at strengths 0.25, 0.50, and 1.00",
    }),
    labelText("nst-stat-a", "0.555", 70, 486, 200, 60, { fontSize: 46, color: COLORS.cyan, bold: true }),
    labelText("nst-stat-a-copy", "Gram style distance\nat strength 1.0", 70, 548, 210, 50, { fontSize: 17, color: COLORS.ivorySoft }),
    line("nst-divider-a", 316, 482, 0, 122, "#394044", 1),
    labelText("nst-stat-b", "0.130", 354, 486, 200, 60, { fontSize: 46, color: COLORS.redSoft, bold: true }),
    labelText("nst-stat-b-copy", "spatial cosine distance\nat strength 1.0", 354, 548, 210, 50, { fontSize: 17, color: COLORS.ivorySoft }),
    line("nst-divider-b", 600, 482, 0, 122, "#394044", 1),
    labelText("nst-stat-c", "500", 638, 486, 200, 60, { fontSize: 46, color: COLORS.ivory, bold: true }),
    labelText("nst-stat-c-copy", "Adam steps\nper registered strength", 638, 548, 210, 50, { fontSize: 17, color: COLORS.ivorySoft }),
    panel("nst-meaning", 910, 478, 326, 134, COLORS.red),
    labelText("nst-meaning-copy", "The donor is an original project-generated cognitive-map image—not a named artist's style.", 936, 504, 274, 86, {
      fontSize: 20,
      color: COLORS.ivory,
      bold: true,
    }),
    ...footer(9, "Frozen VGG19 Gatys-style optimization · Seed 139 · AI-generated style reference disclosed in assets/README.md", true),
  ]);

  // 10 — Geometry sequence and quantitative endpoint.
  createSlide(presentation, "10-geometry-series", COLORS.ivory, [
    ...header(10, "Perspective warp dominates spatial-feature change", "Geometry intervention", false),
    assetImage("geometry-sequence", assets.geometrySequence, 44, 128, 1192, 286, {
      fit: "contain",
      alt: "Las Meninas geometry transformation sequence from 0 to 100 percent",
    }),
    panel("geometry-explainer", 44, 452, 440, 176, COLORS.ink),
    labelText("geometry-explainer-title", "Declared intervention", 70, 478, 340, 36, { fontSize: 25, color: COLORS.ivory, bold: true }),
    labelText("geometry-explainer-copy", "Perspective homography with fixed pivot, convergence, safe-interior crop, and identical output dimensions.", 70, 530, 356, 72, {
      fontSize: 18,
      color: COLORS.ivorySoft,
    }),
    labelText("geometry-style-value", "0.127", 542, 470, 190, 62, { fontSize: 48, color: COLORS.red, bold: true }),
    labelText("geometry-style-copy", "Gram style distance\nat geometry-100", 542, 542, 210, 54, { fontSize: 18, color: COLORS.gray }),
    labelText("geometry-spatial-value", "0.353", 814, 470, 190, 62, { fontSize: 48, color: COLORS.cyan, bold: true }),
    labelText("geometry-spatial-copy", "spatial cosine distance\nat geometry-100", 814, 542, 210, 54, { fontSize: 18, color: COLORS.gray }),
    labelText("geometry-ratio-value", "10.6×", 1070, 470, 150, 62, { fontSize: 48, color: COLORS.ink, bold: true }),
    labelText("geometry-ratio-copy", "larger spatial change than deterministic style-100", 1070, 542, 150, 68, { fontSize: 17, color: COLORS.gray }),
    ...footer(10, "Project geometry trajectory · Endpoint ratio uses spatial cosine distance: 0.3525 ÷ 0.0333", false),
  ]);

  // 11 — Graphviz topology figure plus the decisive near-equal CNN metrics.
  createSlide(presentation, "11-mirror-topology-ablation", COLORS.ink, [
    ...header(11, "Deleting one tiny mirror relation barely moves VGG", "Topology ablation", true),
    panel("topology-graph-frame", 44, 126, 820, 520, COLORS.ivory),
    assetImage("topology-graph", assets.topologyGraph, 54, 138, 800, 496, {
      fit: "contain",
      alt: "Graphviz comparison of relational topology before and after mirror deletion",
    }),
    labelText("topology-control-label", "MATCHED CONTROL", 906, 146, 286, 24, { fontSize: 13, color: COLORS.cyan, bold: true }),
    labelText("topology-control-score", "5 / 5", 906, 184, 286, 62, { fontSize: 48, color: COLORS.ivory, bold: true }),
    labelText("topology-control-metrics", "Style 0.0249\nSpatial cosine 0.0015", 906, 252, 286, 66, { fontSize: 20, color: COLORS.ivorySoft }),
    line("topology-rule", 906, 342, 286, 0, "#3A4043", 1),
    labelText("topology-delete-label", "MIRROR DELETION", 906, 370, 286, 24, { fontSize: 13, color: COLORS.redSoft, bold: true }),
    labelText("topology-delete-score", "4 / 5", 906, 408, 286, 62, { fontSize: 48, color: COLORS.redSoft, bold: true }),
    labelText("topology-delete-metrics", "Style 0.0219\nSpatial cosine 0.0025", 906, 476, 286, 66, { fontSize: 20, color: COLORS.ivorySoft }),
    labelText("topology-conclusion", "The larger semantic loss does not receive the larger style distance.", 906, 568, 286, 66, {
      fontSize: 20,
      color: COLORS.cyanSoft,
      bold: true,
    }),
    ...footer(11, "Manual audit of five predefined relations (one author) + frozen VGG19 distances · Descriptive case study, not an inferential test", true),
  ]);

  // 12 — Honest method failure with exact frozen coordinates and ROI labels.
  createSlide(presentation, "12-geometry-roi-dependence", COLORS.ivory, [
    ...header(12, "HED finds edges; vanishing-point candidates follow the crop", "An instructive failure", false),
    assetImage("geometry-method-strip", assets.geometryMethod, 44, 122, 1192, 230, {
      fit: "contain",
      alt: "HED probability, Hough line segments, RANSAC candidate, and Canny control",
    }),
    panel("roi-card-1", 44, 394, 374, 208, COLORS.ink),
    panel("roi-card-2", 453, 394, 374, 208, COLORS.red),
    panel("roi-card-3", 862, 394, 374, 208, "#DDEFF1"),
    labelText("roi-card-1-label", "HED · NO ROI · AUTOMATIC", 70, 420, 320, 22, { fontSize: 13, color: COLORS.cyan, bold: true }),
    labelText("roi-card-1-point", "(0.720, 0.697)", 70, 458, 320, 46, { fontSize: 31, color: COLORS.ivory, bold: true }),
    labelText("roi-card-1-copy", "57 inliers · ratio 0.173\nmedium evidence grade", 70, 522, 300, 54, { fontSize: 18, color: COLORS.ivorySoft }),
    labelText("roi-card-2-label", "HED · ART-HISTORY ROI", 479, 420, 320, 22, { fontSize: 13, color: COLORS.ivory, bold: true }),
    labelText("roi-card-2-point", "(0.529, 0.439)", 479, 458, 320, 46, { fontSize: 31, color: COLORS.ivory, bold: true }),
    labelText("roi-card-2-copy", "19 inliers · ratio 0.055\nlow evidence grade", 479, 522, 300, 54, { fontSize: 18, color: COLORS.ivory }),
    labelText("roi-card-3-label", "HED · REAR-DOOR ROI", 888, 420, 320, 22, { fontSize: 13, color: COLORS.red, bold: true }),
    labelText("roi-card-3-point", "(0.626, 0.612)", 888, 458, 320, 46, { fontSize: 31, color: COLORS.ink, bold: true }),
    labelText("roi-card-3-copy", "30 inliers · ratio 0.106\nlow evidence grade", 888, 522, 300, 54, { fontSize: 18, color: COLORS.ink }),
    labelText("roi-bottom-line", "Changing the analyst's window changes the candidate: this pipeline does not discover one stable historical vanishing point.", 64, 622, 1128, 34, {
      fontSize: 20,
      color: COLORS.ink,
      bold: true,
      alignment: "center",
    }),
    ...footer(12, "HED → HoughLinesP → 5,000-iteration RANSAC · Coordinates normalized to each full working image · Human validation required", false),
  ]);

  // 13 — AI-generated ready-made concept, explicitly labeled as a mockup.
  createSlide(presentation, "13-ready-made", COLORS.ink, [
    assetImage("ready-made-mockup", assets.mockup, 0, 0, 1280, 720, {
      fit: "cover",
      alt: "AI-generated concept mockup of a mirror aligned with Las Meninas in a gallery",
    }),
    panel("ready-made-text-field", 0, 0, 510, 720, COLORS.ink),
    rectangle("ready-made-red-rail", 0, 0, 12, 720, COLORS.red),
    labelText("ready-made-kicker", "READY-MADE PROPOSAL", 48, 48, 370, 24, { fontSize: 13, color: COLORS.cyan, bold: true }),
    labelText("ready-made-title", "The Missing\nViewer", 48, 112, 410, 126, { fontSize: 50, typeface: FONTS.serif, color: COLORS.ivory, bold: true }),
    line("ready-made-rule", 48, 272, 96, 0, COLORS.red, 4),
    labelText("ready-made-copy", "An ordinary mirror stands on the painting's implied viewing axis. A present-day body occupies the off-canvas position that the mirror and figures construct.", 48, 310, 410, 176, {
      fontSize: 23,
      color: COLORS.ivorySoft,
    }),
    panel("ready-made-disclosure", 48, 520, 410, 82, COLORS.red),
    labelText("ready-made-disclosure-copy", "AI-GENERATED CONCEPT MOCKUP\nNot documentary evidence that the installation was built.", 68, 540, 370, 48, {
      fontSize: 15,
      color: COLORS.ivory,
      bold: true,
    }),
    ...footer(13, "Concept and title: project author · Visualization: AI-generated mockup; prompt and disclosure recorded in assets/README.md", true),
  ]);

  // 14 — Final artwork as the culminating visual, with an honest measurement/creation boundary.
  createSlide(presentation, "14-final-artwork", COLORS.ink, [
    assetImage("final-artwork", assets.finalArtwork, 446, 0, 716, 720, {
      fit: "contain",
      alt: "AI-assisted final artwork Entering Las Meninas",
    }),
    rectangle("artwork-cyan-rail", 0, 0, 12, 720, COLORS.cyan),
    labelText("artwork-kicker", "FINAL ARTWORK", 48, 48, 300, 24, { fontSize: 13, color: COLORS.redSoft, bold: true }),
    labelText("artwork-title", "Entering\nLas Meninas", 48, 110, 350, 126, { fontSize: 48, typeface: FONTS.serif, color: COLORS.ivory, bold: true }),
    labelText("artwork-statement", "The image preserves the five-relation graph while replacing the royal reflection with a contemporary viewer silhouette.", 48, 278, 340, 130, {
      fontSize: 23,
      color: COLORS.ivorySoft,
    }),
    labelText("artwork-relations-value", "5 / 5", 48, 458, 150, 56, { fontSize: 44, color: COLORS.cyan, bold: true }),
    labelText("artwork-relations-copy", "relations preserved\nmanual audit", 48, 520, 152, 48, { fontSize: 16, color: COLORS.warmGray }),
    line("artwork-stat-divider", 224, 452, 0, 124, "#394044", 1),
    labelText("artwork-distance-value", "0.461", 254, 458, 150, 56, { fontSize: 44, color: COLORS.redSoft, bold: true }),
    labelText("artwork-distance-copy", "VGG style distance\nspatial cosine 0.364", 254, 520, 160, 48, { fontSize: 16, color: COLORS.warmGray }),
    labelText("artwork-boundary", "Generative artwork informed by the experiments—not a direct quantitative pipeline output.", 48, 610, 352, 48, {
      fontSize: 15,
      color: COLORS.ivory,
      bold: true,
    }),
    ...footer(14, "AI-assisted artwork · Complete prompt, input description, and disclosure in outputs/artwork/README.md", true),
  ]);

  // 15 — Rights-safe text-only Picasso comparison.
  createSlide(presentation, "15-picasso-generalization", COLORS.ivory, [
    ...header(15, "Historical relation survives large VGG distance", "Picasso stress test · text only", false),
    labelText("picasso-year", "1957", 44, 150, 280, 94, { fontSize: 82, typeface: FONTS.serif, color: COLORS.red, bold: true }),
    labelText("picasso-metadata", "Pablo Picasso's Las Meninas\nMuseu Picasso Barcelona · MPB 70.433\nLandscape format; figures and space reorganized", 44, 270, 420, 110, {
      fontSize: 20,
      color: COLORS.ink,
    }),
    line("picasso-divider", 500, 152, 0, 386, COLORS.warmGray, 1),
    labelText("picasso-style-label", "VGG GRAM STYLE DISTANCE", 548, 164, 300, 22, { fontSize: 13, color: COLORS.red, bold: true }),
    labelText("picasso-style-value", "1.4196", 548, 205, 340, 92, { fontSize: 72, color: COLORS.ink, bold: true }),
    labelText("picasso-spatial-label", "VGG SPATIAL COSINE DISTANCE", 548, 338, 340, 22, { fontSize: 13, color: COLORS.cyan, bold: true }),
    labelText("picasso-spatial-value", "0.7139", 548, 379, 340, 92, { fontSize: 72, color: COLORS.ink, bold: true }),
    panel("picasso-meaning", 930, 160, 306, 350, COLORS.ink),
    labelText("picasso-meaning-title", "What the numbers can say", 958, 190, 250, 56, { fontSize: 25, color: COLORS.ivory, bold: true }),
    labelText("picasso-meaning-copy", "This local reproduction is the most distant comparison in the set.\n\nThe museum's historical interpretation—not VGG—establishes its relationship to Velázquez.", 958, 270, 250, 164, {
      fontSize: 20,
      color: COLORS.ivorySoft,
    }),
    labelText("picasso-rights", "NO IMAGE REDISTRIBUTED\nRights status is not treated as public domain.", 958, 450, 250, 48, { fontSize: 14, color: COLORS.redSoft, bold: true }),
    labelText("picasso-limit", "No human recognition judgments were collected; this is a numerical comparison, not proof of human–CNN disagreement.", 44, 574, 1120, 54, {
      fontSize: 20,
      color: COLORS.ink,
      bold: true,
    }),
    ...footer(15, "Artwork facts: Museu Picasso Barcelona · Local numeric comparison only; source pixels are excluded from the public repository and this deck", false),
  ]);

  // 16 — Closing synthesis, not a generic thank-you slide.
  createSlide(presentation, "16-conclusion-limits-sources", COLORS.ink, [
    rectangle("closing-red-rail", 0, 0, 12, 720, COLORS.red),
    labelText("closing-kicker", "CONCLUSION", 48, 42, 220, 22, { fontSize: 13, color: COLORS.cyan, bold: true }),
    labelText("closing-title", "The CNN helped me see more clearly\nby showing me what it could not see.", 48, 86, 1120, 126, {
      fontSize: 45,
      typeface: FONTS.serif,
      color: COLORS.ivory,
      bold: true,
    }),
    line("closing-rule", 48, 238, 116, 0, COLORS.red, 4),
    labelText("closing-finding-1", "01", 48, 282, 70, 36, { fontSize: 28, color: COLORS.cyan, bold: true }),
    labelText("closing-finding-1-copy", "Style and geometry followed different VGG trajectories.", 118, 286, 440, 42, { fontSize: 20, color: COLORS.ivorySoft, bold: true }),
    labelText("closing-finding-2", "02", 48, 352, 70, 36, { fontSize: 28, color: COLORS.redSoft, bold: true }),
    labelText("closing-finding-2-copy", "Mirror deletion broke a declared relation while CNN distances stayed tiny.", 118, 356, 440, 58, { fontSize: 20, color: COLORS.ivorySoft, bold: true }),
    labelText("closing-finding-3", "03", 48, 438, 70, 36, { fontSize: 28, color: COLORS.ivory, bold: true }),
    labelText("closing-finding-3-copy", "HED extracted edges, but a stable vanishing point did not survive ROI changes.", 118, 442, 440, 58, { fontSize: 20, color: COLORS.ivorySoft, bold: true }),
    panel("closing-limit-panel", 640, 272, 556, 246, COLORS.ivory),
    labelText("closing-limit-title", "Evidence boundary", 670, 298, 490, 38, { fontSize: 28, color: COLORS.ink, bold: true }),
    labelText("closing-limit-copy", "One painting · frozen pretrained models · five manually defined relations · one author · no human participants · no inferential statistics", 670, 354, 490, 108, {
      fontSize: 21,
      color: COLORS.ink,
    }),
    labelText("closing-source-title", "PRIMARY SOURCES", 48, 558, 260, 20, { fontSize: 12, color: COLORS.cyan, bold: true }),
    labelText(
      "closing-sources",
      "Museo Nacional del Prado · Wikimedia Commons · Simonyan & Zisserman (VGG, 2015) · Gatys et al. (CVPR 2016) · Xie & Tu (ICCV 2015) · OpenCV Hough documentation · Museu Picasso Barcelona",
      48,
      588,
      1148,
      52,
      { fontSize: 14, color: COLORS.warmGray },
    ),
    ...footer(16, "Full citations, methods, prompts, checksums, and reproducibility notes: README.md and report/final-report.md", true),
  ]);

  return presentation;
}

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const finalPptx = path.resolve(process.argv[2] ?? "outputs/presentation/entering-las-meninas.pptx");
  const qaDir = path.resolve(process.argv[3] ?? "outputs/presentation/rendered");
  const projectRoot = path.resolve(process.argv[4] ?? process.cwd());

  await fs.mkdir(path.dirname(finalPptx), { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const presentation = await buildPresentation(projectRoot);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(qaDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 2 }));
    const layoutBlob = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(qaDir, `${stem}.layout.json`), await layoutBlob.text());
  }

  await writeBlob(
    path.join(qaDir, "deck-montage.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );
  const inspection = await presentation.inspect({
    kind: "slide,textbox,shape,image,chart,table",
    maxChars: 100000,
  });
  await fs.writeFile(path.join(qaDir, "deck-inspection.ndjson"), inspection.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  process.stdout.write(`Created ${finalPptx} with ${presentation.slides.items.length} slides.\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
