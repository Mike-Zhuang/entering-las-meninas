#!/usr/bin/env bash
set -euo pipefail

projectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outputDirectory="${projectRoot}/data/raw"
outputPath="${outputDirectory}/las-meninas-original.jpg"
sourceUrl="https://upload.wikimedia.org/wikipedia/commons/9/99/Las_Meninas%2C_by_Diego_Vel%C3%A1zquez%2C_from_Prado_in_Google_Earth.jpg"
expectedSha256="dd0cab7a6bebcee8c492f3181b324b91df8a8f23f1794dcfae45e454efa3fda0"

mkdir -p "${outputDirectory}"

if [[ ! -f "${outputPath}" ]]; then
  curl --fail --location --progress-bar "${sourceUrl}" --output "${outputPath}"
fi

actualSha256="$(shasum -a 256 "${outputPath}" | awk '{print $1}')"
if [[ "${actualSha256}" != "${expectedSha256}" ]]; then
  printf 'Checksum mismatch for %s\nExpected: %s\nActual:   %s\n' \
    "${outputPath}" "${expectedSha256}" "${actualSha256}" >&2
  exit 1
fi

printf 'Verified source image: %s\n' "${outputPath}"
