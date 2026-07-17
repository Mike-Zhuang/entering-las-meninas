#!/usr/bin/env bash
set -euo pipefail

projectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${projectRoot}"

requiredFiles=(
  "LICENSE"
  "THIRD-PARTY-NOTICES.md"
  "CITATION.cff"
  "data/README.md"
  "assets/README.md"
  "outputs/artwork/README.md"
)

for requiredFile in "${requiredFiles[@]}"; do
  if [[ ! -s "${requiredFile}" ]]; then
    printf '缺少公开发布所需文件或文件为空：%s\n' "${requiredFile}" >&2
    exit 1
  fi
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf '请在 Git 仓库初始化后运行此检查。\n' >&2
  exit 1
fi

trackedFiles="$(git ls-files)"
for forbiddenPattern in \
  '^pics/' \
  '^data/raw/' \
  '^models/' \
  '^cache/' \
  '^\.cache/' \
  '^outputs/cnn-analysis/descriptors/'
do
  if printf '%s\n' "${trackedFiles}" | grep -Eq "${forbiddenPattern}"; then
    printf '公开仓库包含禁止追踪的路径，匹配：%s\n' "${forbiddenPattern}" >&2
    printf '%s\n' "${trackedFiles}" | grep -E "${forbiddenPattern}" >&2
    exit 1
  fi
done

publicFiles=(
  "data/processed/las-meninas-2048.jpg"
  "outputs/reference/las-meninas-reference.jpg"
  "assets/style/cognitive-map-style-reference.png"
  "assets/installation/the-missing-viewer-mockup.png"
  "outputs/artwork/entering-las-meninas-final.png"
)

for publicFile in "${publicFiles[@]}"; do
  if git check-ignore --quiet --no-index "${publicFile}"; then
    printf '应公开的成果被 .gitignore 排除：%s\n' "${publicFile}" >&2
    exit 1
  fi
done

maximumBytes=$((99 * 1024 * 1024))
disallowedAssetHashes="070d97f487755826289867b6c15919ec631c2d3dd5c911989bbb20655844dacc
78752b3439a250e3bacff898f7f6d776e55a22f75e45cfa16b509a6e6b33388a
807ba7e96546762b6e59b343c41bb88a25769ae7500df9959b888f0950304eec
fd3932138def185693b78036659f5c01602b3ad492316dd56d0e326141805771"

while IFS= read -r trackedFile; do
  [[ -z "${trackedFile}" || ! -f "${trackedFile}" ]] && continue
  fileBytes="$(stat -f '%z' "${trackedFile}" 2>/dev/null || stat -c '%s' "${trackedFile}")"
  if (( fileBytes > maximumBytes )); then
    printf '文件超过 99 MiB 的公开发布上限：%s (%s bytes)\n' \
      "${trackedFile}" "${fileBytes}" >&2
    exit 1
  fi
  fileSha256="$(shasum -a 256 "${trackedFile}" | awk '{print $1}')"
  if printf '%s\n' "${disallowedAssetHashes}" | grep -Fqx "${fileSha256}"; then
    printf '公开仓库包含本地未授权或来源未核验素材的字节副本：%s\n' \
      "${trackedFile}" >&2
    exit 1
  fi
done <<< "${trackedFiles}"

if ! grep -Fq 'Picasso / VEGAP' THIRD-PARTY-NOTICES.md; then
  printf '第三方声明缺少 Picasso 复制图权利边界。\n' >&2
  exit 1
fi

if ! grep -Fq 'dd0cab7a6bebcee8c492f3181b324b91df8a8f23f1794dcfae45e454efa3fda0' \
  THIRD-PARTY-NOTICES.md; then
  printf '第三方声明缺少 Velázquez 原始输入的 SHA-256。\n' >&2
  exit 1
fi

printf '公开发布检查通过：禁止素材未追踪，许可与来源文件齐全。\n'
