#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

projectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pythonExecutable="${PYTHON_EXECUTABLE:-python3}"
sourceImage="${projectRoot}/data/raw/las-meninas-original.jpg"
styleImage="${projectRoot}/assets/style/cognitive-map-style-reference.png"
finalArtwork="${projectRoot}/outputs/artwork/entering-las-meninas-final.png"
outputRoot="${projectRoot}/outputs"
cacheRoot="${projectRoot}/.cache"
device="auto"
geometryBackend="auto"
styleSteps=500
styleLongSide=512
cnnImageSize=512
seed=139
dryRun=false
skipDownload=false
forceCnn=false

usage() {
  cat <<'EOF'
用法：./scripts/run-full-pipeline.sh [选项]

完整复现《宫娥》项目的受控变换、Gatys 神经风格迁移、VGG19 表征比较、
HED/Hough/RANSAC 几何分析、ROI 敏感性检查、定量汇总、图表和视差视频。

选项：
  --python PATH              Python 可执行文件（默认读取 PYTHON_EXECUTABLE 或 python3）
  --source PATH              公版《宫娥》原图路径
  --style-image PATH         本项目原创风格参考图路径
  --output-root PATH         输出根目录（默认 outputs）
  --device auto|cpu|mps      VGG19 与风格迁移设备（默认 auto）
  --geometry-backend NAME    auto|hed|vgg|canny（默认 auto）
  --style-steps N            每个神经风格强度的 Adam 步数（正式默认 500）
  --style-long-side N        神经风格工作图最长边（正式默认 512）
  --cnn-image-size N         VGG19 比较的方形输入尺寸（正式默认 512）
  --seed N                   全流程随机种子（默认 139）
  --skip-download            原图缺失时不自动下载
  --force-cnn                忽略 VGG19 描述符缓存并重新提取
  --dry-run                  只打印全部命令，不读取模型或生成结果
  -h, --help                 显示本帮助

环境变量：
  PYTHON_EXECUTABLE          与 --python 等价，命令行参数优先

示例：
  PYTHON_EXECUTABLE=/path/to/python ./scripts/run-full-pipeline.sh --device mps
  ./scripts/run-full-pipeline.sh --dry-run --skip-download
EOF
}

fail() {
  printf '错误：%s\n' "$1" >&2
  exit 2
}

requireValue() {
  local optionName="$1"
  local optionValue="${2:-}"
  [[ -n "${optionValue}" ]] || fail "${optionName} 需要一个值。"
}

requirePositiveInteger() {
  local optionName="$1"
  local optionValue="$2"
  [[ "${optionValue}" =~ ^[1-9][0-9]*$ ]] || fail "${optionName} 必须是正整数。"
}

printCommand() {
  local argument
  local escapedArgument
  printf '[执行]'
  for argument in "$@"; do
    # macOS 自带旧版 Bash 的 printf %q 会破坏 UTF-8；这里仅为日志做可读转义。
    escapedArgument="${argument//\\/\\\\}"
    escapedArgument="${escapedArgument//\"/\\\"}"
    escapedArgument="${escapedArgument//\$/\\\$}"
    escapedArgument="${escapedArgument//\`/\\\`}"
    printf ' "%s"' "${escapedArgument}"
  done
  printf '\n'
}

runCommand() {
  printCommand "$@"
  if [[ "${dryRun}" == false ]]; then
    "$@"
  fi
}

onError() {
  local exitCode=$?
  local lineNumber="${BASH_LINENO[0]:-unknown}"
  printf '流水线在第 %s 行失败，退出码 %s。\n' "${lineNumber}" "${exitCode}" >&2
  exit "${exitCode}"
}
trap onError ERR

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      requireValue "$1" "${2:-}"
      pythonExecutable="$2"
      shift 2
      ;;
    --source)
      requireValue "$1" "${2:-}"
      sourceImage="$2"
      shift 2
      ;;
    --style-image)
      requireValue "$1" "${2:-}"
      styleImage="$2"
      shift 2
      ;;
    --output-root)
      requireValue "$1" "${2:-}"
      outputRoot="$2"
      shift 2
      ;;
    --device)
      requireValue "$1" "${2:-}"
      device="$2"
      shift 2
      ;;
    --geometry-backend)
      requireValue "$1" "${2:-}"
      geometryBackend="$2"
      shift 2
      ;;
    --style-steps)
      requireValue "$1" "${2:-}"
      styleSteps="$2"
      shift 2
      ;;
    --style-long-side)
      requireValue "$1" "${2:-}"
      styleLongSide="$2"
      shift 2
      ;;
    --cnn-image-size)
      requireValue "$1" "${2:-}"
      cnnImageSize="$2"
      shift 2
      ;;
    --seed)
      requireValue "$1" "${2:-}"
      seed="$2"
      shift 2
      ;;
    --skip-download)
      skipDownload=true
      shift
      ;;
    --force-cnn)
      forceCnn=true
      shift
      ;;
    --dry-run)
      dryRun=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "未知选项：$1"
      ;;
  esac
done

case "${device}" in
  auto|cpu|mps) ;;
  *) fail "--device 必须是 auto、cpu 或 mps。" ;;
esac

case "${geometryBackend}" in
  auto|hed|vgg|canny) ;;
  *) fail "--geometry-backend 必须是 auto、hed、vgg 或 canny。" ;;
esac

requirePositiveInteger "--style-steps" "${styleSteps}"
requirePositiveInteger "--style-long-side" "${styleLongSide}"
requirePositiveInteger "--cnn-image-size" "${cnnImageSize}"
requirePositiveInteger "--seed" "${seed}"

defaultSourceImage="${projectRoot}/data/raw/las-meninas-original.jpg"
if [[ "${dryRun}" == false ]]; then
  command -v "${pythonExecutable}" >/dev/null 2>&1 || fail "找不到 Python：${pythonExecutable}"

  if [[ "${sourceImage}" == "${defaultSourceImage}" ]]; then
    if [[ ! -f "${sourceImage}" && "${skipDownload}" == true ]]; then
      fail "原图不存在且已指定 --skip-download：${sourceImage}"
    fi
    # 下载脚本对已有文件只做 SHA-256 校验，不会重复传输。
    runCommand "${projectRoot}/scripts/download-source-image.sh"
  elif [[ ! -f "${sourceImage}" ]]; then
    fail "自定义原图不存在：${sourceImage}"
  fi

  [[ -f "${styleImage}" ]] || fail "风格参考图不存在：${styleImage}"
  [[ -f "${finalArtwork}" ]] || fail "最终原创作品不存在：${finalArtwork}"
fi

runCommand "${pythonExecutable}" -c \
  'import cv2, matplotlib, numpy, PIL, torch, torchvision; print("依赖检查通过", torch.__version__, torchvision.__version__)'
runCommand mkdir -p "${outputRoot}" "${cacheRoot}"

transformationsRoot="${outputRoot}/transformations"
neuralStyleRoot="${outputRoot}/neural-style-transfer"
combinedNeuralStyleRoot="${outputRoot}/neural-style-transfer-combined"
cnnRoot="${outputRoot}/cnn-analysis"
geometryRoot="${outputRoot}/geometry-analysis"
tablesRoot="${outputRoot}/tables"
metricsRoot="${outputRoot}/metrics"
figuresRoot="${outputRoot}/figures"
animationRoot="${outputRoot}/animation"

# 第一阶段：生成固定尺度代理、裁剪、几何序列、非神经基线和镜像拓扑控制。
runCommand "${pythonExecutable}" "${projectRoot}/src/transformations.py" all \
  --input "${sourceImage}" \
  --output-dir "${transformationsRoot}" \
  --proxy-long-side 2048 \
  --levels 0,25,50,75,100 \
  --vp-x 0.56 \
  --vp-y 0.53

workingProxy="${transformationsRoot}/working-proxy.png"
geometryMaximum="${transformationsRoot}/geometry/geometry-100.png"

# 第二阶段：用同一 VGG19、同一初始状态分别优化三个 style strengths。
runCommand "${pythonExecutable}" "${projectRoot}/src/neural-style-transfer.py" "${workingProxy}" \
  --style-image "${styleImage}" \
  --output-dir "${neuralStyleRoot}" \
  --style-strengths 0.25,0.5,1.0 \
  --long-side "${styleLongSide}" \
  --steps "${styleSteps}" \
  --device "${device}" \
  --seed "${seed}" \
  --weights default \
  --progress-every 25 \
  --overwrite

# 组合条件从几何最大变换出发，只改变 style，形成真正的 2×2 神经实验矩阵。
runCommand "${pythonExecutable}" "${projectRoot}/src/neural-style-transfer.py" "${geometryMaximum}" \
  --style-image "${styleImage}" \
  --output-dir "${combinedNeuralStyleRoot}" \
  --style-strengths 1.0 \
  --long-side "${styleLongSide}" \
  --steps "${styleSteps}" \
  --device "${device}" \
  --seed "${seed}" \
  --weights default \
  --progress-every 25 \
  --overwrite

candidateImages=(
  "${transformationsRoot}/geometry/geometry-000.png"
  "${transformationsRoot}/geometry/geometry-025.png"
  "${transformationsRoot}/geometry/geometry-050.png"
  "${transformationsRoot}/geometry/geometry-075.png"
  "${transformationsRoot}/geometry/geometry-100.png"
  "${transformationsRoot}/style/style-baseline-000.png"
  "${transformationsRoot}/style/style-baseline-025.png"
  "${transformationsRoot}/style/style-baseline-050.png"
  "${transformationsRoot}/style/style-baseline-075.png"
  "${transformationsRoot}/style/style-baseline-100.png"
  "${transformationsRoot}/topology/topology-original.png"
  "${transformationsRoot}/topology/topology-mirror-deletion.png"
  "${transformationsRoot}/topology/topology-matched-control.png"
  "${transformationsRoot}/topology/topology-sham-warp.png"
  "${transformationsRoot}/combined-geometry-style-100.png"
  "${neuralStyleRoot}/neural-style-strength-0p25.png"
  "${neuralStyleRoot}/neural-style-strength-0p5.png"
  "${neuralStyleRoot}/neural-style-strength-1.png"
  "${combinedNeuralStyleRoot}/neural-style-strength-1.png"
  "${finalArtwork}"
)

# Picasso 复制图仅是可选的本地比较输入。公有仓库不含该文件时自动跳过，不影响主实验。
picassoComparison="${projectRoot}/pics/picasso.png"
if [[ -f "${picassoComparison}" ]]; then
  candidateImages+=("${picassoComparison}")
  printf '已加入本地 Picasso 比较输入；其 CNN 输入图与叠加图受 .gitignore 排除。\n'
else
  printf '未发现本地 Picasso 比较输入，按公有仓库模式跳过。\n'
fi

# 第三阶段：以原作代理为唯一参考，比较全部受控条件的 Gram 与空间特征距离。
cnnCommand=(
  "${pythonExecutable}"
  "${projectRoot}/src/cnn-analysis.py"
  --reference "${workingProxy}"
  --images
)
cnnCommand+=("${candidateImages[@]}")
cnnCommand+=(
  --output-dir "${cnnRoot}"
  --cache-dir "${cacheRoot}/cnn-analysis"
  --image-size "${cnnImageSize}"
  --spatial-size 32
  --device "${device}"
  --weights default
  --seed "${seed}"
)
if [[ "${forceCnn}" == true ]]; then
  cnnCommand+=(--force)
fi
runCommand "${cnnCommand[@]}"

# 第四阶段：在原作、几何强度和镜像控制上独立运行 CNN edge + Hough + RANSAC。
geometryInputs=(
  "original=${workingProxy}"
  "geometry-025=${transformationsRoot}/geometry/geometry-025.png"
  "geometry-050=${transformationsRoot}/geometry/geometry-050.png"
  "geometry-075=${transformationsRoot}/geometry/geometry-075.png"
  "geometry-100=${transformationsRoot}/geometry/geometry-100.png"
  "topology-original=${transformationsRoot}/topology/topology-original.png"
  "topology-mirror-deletion=${transformationsRoot}/topology/topology-mirror-deletion.png"
  "topology-matched-control=${transformationsRoot}/topology/topology-matched-control.png"
  "topology-sham-warp=${transformationsRoot}/topology/topology-sham-warp.png"
)

for geometrySpec in "${geometryInputs[@]}"; do
  geometryLabel="${geometrySpec%%=*}"
  geometryImage="${geometrySpec#*=}"
  runCommand "${pythonExecutable}" "${projectRoot}/src/geometry-analysis.py" \
    --image "${geometryImage}" \
    --output-dir "${geometryRoot}/${geometryLabel}" \
    --backend "${geometryBackend}" \
    --device "${device}" \
    --hed-cache-dir "${cacheRoot}/hed" \
    --max-side 1280 \
    --seed "${seed}"
done

# 两个 ROI 是报告中明确披露的分析者先验，用来测试候选消失点对观察窗口的敏感性。
# 它们不参与自动主结果，也不能被表述为模型自动发现的 ground truth。
runCommand "${pythonExecutable}" "${projectRoot}/src/geometry-analysis.py" \
  --image "${workingProxy}" \
  --output-dir "${geometryRoot}/art-history-roi" \
  --backend "${geometryBackend}" \
  --device "${device}" \
  --hed-cache-dir "${cacheRoot}/hed" \
  --max-side 1280 \
  --vanishing-roi 0.5,0.3,0.65,0.48 \
  --seed "${seed}"

runCommand "${pythonExecutable}" "${projectRoot}/src/geometry-analysis.py" \
  --image "${workingProxy}" \
  --output-dir "${geometryRoot}/door-roi" \
  --backend "${geometryBackend}" \
  --device "${device}" \
  --hed-cache-dir "${cacheRoot}/hed" \
  --max-side 1280 \
  --vanishing-roi 0.5,0.495,0.645,0.62 \
  --seed "${seed}"

# 第五阶段：把逐图 JSON 汇总为可画图的审计表，再统一出图。
runCommand "${pythonExecutable}" "${projectRoot}/scripts/build-analysis-tables.py" \
  --cnn-json "${cnnRoot}/cnn-analysis.json" \
  --geometry-root "${geometryRoot}" \
  --output-dir "${tablesRoot}"

runCommand "${pythonExecutable}" "${projectRoot}/src/summarize-results.py" \
  --cnn-json "${cnnRoot}/cnn-analysis.json" \
  --style-csv "${cnnRoot}/style-distances.csv" \
  --spatial-csv "${cnnRoot}/spatial-distances.csv" \
  --style-transfer-manifest "${neuralStyleRoot}/manifest.json" \
  --combined-style-transfer-manifest "${combinedNeuralStyleRoot}/manifest.json" \
  --topology-csv "${projectRoot}/data/topology-relations.csv" \
  --transform-manifest "${transformationsRoot}/transformations-manifest.json" \
  --output-dir "${metricsRoot}" \
  --dpi 300

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" matrix \
  --original "${workingProxy}" \
  --style-only "${neuralStyleRoot}/neural-style-strength-1.png" \
  --geometry-only "${geometryMaximum}" \
  --combined "${combinedNeuralStyleRoot}/neural-style-strength-1.png" \
  --output "${figuresRoot}/controlled-neural-matrix.png" \
  --title "Style × Geometry: Controlled Neural Transformation Matrix"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" compare \
  --image "Original=${workingProxy}" \
  --image "25%=${transformationsRoot}/geometry/geometry-025.png" \
  --image "50%=${transformationsRoot}/geometry/geometry-050.png" \
  --image "75%=${transformationsRoot}/geometry/geometry-075.png" \
  --image "100%=${transformationsRoot}/geometry/geometry-100.png" \
  --columns 5 \
  --output "${figuresRoot}/geometry-sequence.png" \
  --title "Controlled Perspective and Depth Transformation"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" compare \
  --image "Original=${workingProxy}" \
  --image "Strength 0.25=${neuralStyleRoot}/neural-style-strength-0p25.png" \
  --image "Strength 0.50=${neuralStyleRoot}/neural-style-strength-0p5.png" \
  --image "Strength 1.00=${neuralStyleRoot}/neural-style-strength-1.png" \
  --columns 4 \
  --output "${figuresRoot}/neural-style-sequence.png" \
  --title "Gatys Neural Style Transfer — Fixed VGG19, Increasing Strength"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" compare \
  --image "Original=${transformationsRoot}/topology/topology-original.png" \
  --image "Mirror removed=${transformationsRoot}/topology/topology-mirror-deletion.png" \
  --image "Matched patch=${transformationsRoot}/topology/topology-matched-control.png" \
  --image "Sham warp=${transformationsRoot}/topology/topology-sham-warp.png" \
  --columns 4 \
  --output "${figuresRoot}/topology-controls.png" \
  --title "Mirror Relation: Target Edit and Matched Controls"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" compare \
  --image "Input=${geometryRoot}/original/working-image.png" \
  --image "HED/VGG edge probability=${geometryRoot}/original/cnn-edge-probability.png" \
  --image "CNN lines and RANSAC point=${geometryRoot}/original/cnn-lines-vanishing-point.png" \
  --image "Canny control=${geometryRoot}/original/canny-lines-vanishing-point.png" \
  --columns 4 \
  --output "${figuresRoot}/geometry-method.png" \
  --title "CNN Boundary Evidence, Hough Lines, and RANSAC Estimate"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" feature-space \
  --csv "${tablesRoot}/cnn-feature-space.csv" \
  --output "${figuresRoot}/cnn-style-spatial-feature-space.png" \
  --x-column x \
  --y-column y \
  --label-column label \
  --group-column group \
  --title "VGG19 Style Distance vs Spatial-Content Distance"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" metrics \
  --csv "${tablesRoot}/geometry-cnn-metrics.csv" \
  --x-column intensity \
  --output "${figuresRoot}/geometry-cnn-trajectories.png" \
  --title "CNN Distances Across Geometry Strength"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" metrics \
  --csv "${tablesRoot}/neural-style-cnn-metrics.csv" \
  --x-column intensity \
  --output "${figuresRoot}/neural-style-cnn-trajectories.png" \
  --title "CNN Distances Across Neural Style Strength"

runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" metrics \
  --csv "${tablesRoot}/geometry-detection-metrics.csv" \
  --x-column intensity \
  --output "${figuresRoot}/geometry-detection-trajectories.png" \
  --title "Line and Vanishing-Point Evidence Across Geometry Strength"

# 该视频显式使用启发式纵向深度，只用于“进入画面”的展示，不冒充 CNN 深度预测。
runCommand "${pythonExecutable}" "${projectRoot}/src/visualization.py" parallax \
  --image "${finalArtwork}" \
  --heuristic-depth \
  --output "${animationRoot}/entering-las-meninas-heuristic-parallax.mp4" \
  --duration 5 \
  --fps 24 \
  --horizontal-amplitude 20 \
  --vertical-amplitude 3 \
  --max-long-side 960

# 最后阶段只接受具备真实模型签名、完整清单和非空提交图表的结果。
runCommand "${pythonExecutable}" "${projectRoot}/scripts/validate-artifacts.py" \
  --project-root "${projectRoot}" \
  --output-root "${outputRoot}" \
  --write-report "${outputRoot}/pipeline-validation.json"

printf '完整流水线结束。验证报告：%s\n' "${outputRoot}/pipeline-validation.json"
