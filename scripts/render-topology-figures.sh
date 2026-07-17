#!/usr/bin/env bash
# 重新生成关系图与镜子局部消融对比板；不读取 pics/ 中的任何辅助图片。
set -euo pipefail

projectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dotBinary="${DOT_BINARY:-$(command -v dot || true)}"

if [[ -z "${dotBinary}" && -x "/opt/homebrew/bin/dot" ]]; then
  dotBinary="/opt/homebrew/bin/dot"
fi

if [[ -z "${dotBinary}" ]]; then
  echo "Graphviz dot 未安装；DOT 源文件已保留，但无法渲染。" >&2
  exit 1
fi

mkdir -p "${projectRoot}/outputs/figures"

for graphName in las-meninas-relations mirror-deletion-difference; do
  sourcePath="${projectRoot}/assets/graphs/${graphName}.dot"
  "${dotBinary}" -Kdot -Tsvg "${sourcePath}" -o "${projectRoot}/outputs/figures/${graphName}.svg"
  "${dotBinary}" -Kdot -Tpng:cairo -Gdpi=300 "${sourcePath}" -o "${projectRoot}/outputs/figures/${graphName}.png"
done

if ! command -v magick >/dev/null 2>&1; then
  echo "ImageMagick magick 未安装；无法生成局部对比板。" >&2
  exit 1
fi

topologyRoot="${projectRoot}/outputs/transformations/topology"
workDirectory="$(mktemp -d)"
trap 'rm -rf "${workDirectory}"' EXIT

# 三幅图使用完全相同的像素窗口。窗口在消融框周围保留足够语境，便于确认它确是墙上镜子。
cropGeometry="360x500+570+900"
panelWidth=720
panelHeight=1000
regularFont="/System/Library/Fonts/Supplemental/Arial.ttf"
boldFont="/System/Library/Fonts/Supplemental/Arial Bold.ttf"

renderPanel() {
  local sourcePath="$1"
  local label="$2"
  local note="$3"
  local outputPath="$4"

  magick "${sourcePath}" \
    -crop "${cropGeometry}" +repage \
    -resize "${panelWidth}x${panelHeight}" \
    -background "#F6F0E4" -gravity north \
    -splice 0x150 \
    -fill "#152329" -font "${boldFont}" -pointsize 48 \
    -annotate +0+36 "${label}" \
    -fill "#536267" -font "${regularFont}" -pointsize 27 \
    -annotate +0+94 "${note}" \
    -bordercolor "#C9C1B4" -border 3 \
    "${outputPath}"
}

renderPanel \
  "${topologyRoot}/topology-original.png" \
  "ORIGINAL" \
  "reflection relation intact" \
  "${workDirectory}/original.png"

renderPanel \
  "${topologyRoot}/topology-matched-control.png" \
  "MATCHED CONTROL" \
  "equal-area perturbation; figures remain" \
  "${workDirectory}/matched-control.png"

renderPanel \
  "${topologyRoot}/topology-mirror-deletion.png" \
  "MIRROR REMOVED" \
  "local edit, relational endpoint lost" \
  "${workDirectory}/mirror-removed.png"

magick \
  "${workDirectory}/original.png" \
  "${workDirectory}/matched-control.png" \
  "${workDirectory}/mirror-removed.png" \
  +append +repage \
  -background "#F6F0E4" -gravity north \
  -splice 0x205 \
  -fill "#152329" -font "${boldFont}" -pointsize 52 \
  -annotate +0+38 "MIRROR ABLATION · IDENTICAL LOCAL WINDOW" \
  -fill "#536267" -font "${regularFont}" -pointsize 26 \
  -annotate +0+113 "Only the target relation changes; framing and scale remain fixed." \
  "${workDirectory}/mirror-local-ablation.png"

# 在第二次无重绘写出时添加 300 ppi 元数据，避免密度设置放大排版字体。
magick \
  "${workDirectory}/mirror-local-ablation.png" \
  -units PixelsPerInch -density 300 \
  "${projectRoot}/outputs/figures/mirror-local-ablation.png"

echo "拓扑图已写入 outputs/figures。"
