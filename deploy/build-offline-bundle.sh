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
mkdir -p "${output}"/{docker-images,python-wheels,frontend-build,database-migrations,models,model-manifests,configs,scripts,docs,release}
python3 -m pip download --requirement "${repo_root}/requirements-platform.txt" \
  --dest "${output}/python-wheels"
WHITEWHALE_RELEASE_TAG=offline docker compose -f "${repo_root}/compose.yaml" build api web
docker save whitewhale-api:offline | gzip -1 > "${output}/docker-images/whitewhale-api.tar.gz"
docker save whitewhale-web:offline postgres:16-alpine | gzip -1 \
  > "${output}/docker-images/whitewhale-web-postgres.tar.gz"
git -C "${repo_root}" archive HEAD | tar -x -C "${output}/release"
frontend_container="$(docker create whitewhale-web:offline)"
cleanup_container() { docker rm -f "${frontend_container}" >/dev/null 2>&1 || true; }
trap cleanup_container EXIT
docker cp "${frontend_container}:/srv/." "${output}/frontend-build"
cp -a "${repo_root}/migrations/." "${output}/database-migrations/"
cp -a "${repo_root}/scripts/." "${output}/scripts/"
cp -a "${repo_root}/docs/." "${output}/docs/"
if [[ -d "${repo_root}/configs" ]]; then
  cp -a "${repo_root}/configs/." "${output}/configs/"
fi
model_source="${WHITEWHALE_OFFLINE_MODELS_DIR:-${WHITEWHALE_DATA_ROOT:-/srv/whitewhale/data}/models}"
if [[ -d "${model_source}" ]]; then
  cp -a "${model_source}/." "${output}/models/"
fi
(cd "${output}/models" && find . -type f -print0 | sort -z \
  | xargs -0 -r sha256sum > "${output}/model-manifests/files.sha256")
python3 - "${output}/model-manifests/inventory.json" "${model_source}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
source = Path(sys.argv[2])
output.write_text(json.dumps({
    "source": str(source),
    "calibration_note": (
        "Model lifecycle metadata remains authoritative in PostgreSQL; "
        "verify files.sha256 before offline restore."
    ),
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
(cd "${output}" && find . -type f ! -name checksums.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > checksums.sha256)
echo "offline bundle created: ${output}"
