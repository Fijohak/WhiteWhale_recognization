#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: $0 /absolute/output-directory" >&2
  exit 2
fi
output="$1"
[[ "${output}" == /* && ! -e "${output}" ]] || {
  echo "output must be an absolute path that does not exist" >&2
  exit 2
}
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${output}"/{docker-images,python-wheels,release}
python3 -m pip download --requirement "${repo_root}/requirements-platform.txt" \
  --dest "${output}/python-wheels"
WHITEWHALE_RELEASE_TAG=offline docker compose -f "${repo_root}/compose.yaml" build api web
docker save whitewhale-api:offline | gzip -1 > "${output}/docker-images/whitewhale-api.tar.gz"
docker save whitewhale-web:offline postgres:16-alpine | gzip -1 \
  > "${output}/docker-images/whitewhale-web-postgres.tar.gz"
git -C "${repo_root}" archive HEAD | tar -x -C "${output}/release"
(cd "${output}" && find . -type f ! -name checksums.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > checksums.sha256)
echo "offline bundle created: ${output}"
