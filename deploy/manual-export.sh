#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: $0 /absolute/path/whitewhale-export.tar" >&2
  exit 2
fi
destination="$1"
[[ "${destination}" == /* && "${destination}" == *.tar ]] || {
  echo "destination must be an absolute .tar path" >&2
  exit 2
}
[[ ! -e "${destination}" && ! -e "${destination}.tmp" ]] || {
  echo "destination or temporary output already exists" >&2
  exit 2
}
: "${WHITEWHALE_CURRENT:=/srv/whitewhale/current}"
: "${WHITEWHALE_SHARED_ENV:=/etc/whitewhale/platform.env}"
: "${WHITEWHALE_DATA_ROOT:=/srv/whitewhale/data}"

parent="$(dirname "${destination}")"
mkdir -p "${parent}"
workspace="$(mktemp -d "${parent}/.whitewhale-export-XXXXXX")"
api_stopped=0
cleanup() {
  if ((api_stopped)); then
    docker compose --env-file "${WHITEWHALE_SHARED_ENV}" \
      -f "${WHITEWHALE_CURRENT}/compose.yaml" start api >/dev/null || true
  fi
  rm -rf -- "${workspace}"
}
trap cleanup EXIT

compose=(docker compose --env-file "${WHITEWHALE_SHARED_ENV}" -f "${WHITEWHALE_CURRENT}/compose.yaml")
"${compose[@]}" stop api
api_stopped=1
"${compose[@]}" exec -T postgres sh -c \
  'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom' \
  > "${workspace}/database.dump"
tar -C "${WHITEWHALE_DATA_ROOT}" -cf "${workspace}/data.tar" .
"${compose[@]}" exec -T postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT json_build_object('"'"'active_catalog_id'"'"', (SELECT catalog_id FROM active_catalog_pointer WHERE singleton_id=1), '"'"'production_models'"'"', COALESCE((SELECT json_object_agg(model_family, model_version_id) FROM production_model_pointer), '"'"'{}'"'"'::json))"' \
  > "${workspace}/active-pointers.json"
printf '{"created_at":"%s","release_commit":"%s"}\n' \
  "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
  "$(<"${WHITEWHALE_CURRENT}/.release-commit")" \
  > "${workspace}/export-manifest.json"
(cd "${workspace}" && sha256sum database.dump data.tar active-pointers.json export-manifest.json > checksums.sha256)
temporary="${destination}.tmp"
tar -C "${workspace}" -cf "${temporary}" \
  database.dump data.tar active-pointers.json export-manifest.json checksums.sha256
mv "${temporary}" "${destination}"
"${compose[@]}" start api
api_stopped=0
echo "export created: ${destination}"
