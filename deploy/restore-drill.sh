#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
  echo "usage: $0 export.tar /absolute/empty/restore-root" >&2
  exit 2
fi
bundle="$1"
restore_root="$2"
[[ "${restore_root}" == /* ]] || { echo "restore root must be absolute" >&2; exit 2; }
[[ ! -e "${restore_root}" ]] || { echo "restore root must not exist" >&2; exit 2; }
: "${WHITEWHALE_CURRENT:=/srv/whitewhale/current}"
: "${WHITEWHALE_SHARED_ENV:=/etc/whitewhale/platform.env}"

python3 "${WHITEWHALE_CURRENT}/scripts/verify_export.py" "${bundle}" >/dev/null
workspace="$(mktemp -d)"
cleanup() { rm -rf -- "${workspace}"; }
trap cleanup EXIT
tar -C "${workspace}" -xf "${bundle}"
mkdir -p "${restore_root}"
tar -C "${restore_root}" -xf "${workspace}/data.tar"

compose=(docker compose --env-file "${WHITEWHALE_SHARED_ENV}" -f "${WHITEWHALE_CURRENT}/compose.yaml")
database="whitewhale_restore_check"
"${compose[@]}" exec -T postgres sh -c \
  'dropdb -U "$POSTGRES_USER" --if-exists whitewhale_restore_check && createdb -U "$POSTGRES_USER" whitewhale_restore_check'
"${compose[@]}" exec -T postgres sh -c \
  'pg_restore -U "$POSTGRES_USER" -d whitewhale_restore_check --exit-on-error' \
  < "${workspace}/database.dump"
"${compose[@]}" exec -T postgres sh -c \
  'psql -U "$POSTGRES_USER" -d whitewhale_restore_check -Atc "SELECT version_num FROM alembic_version"'
echo "restore drill passed; files retained at ${restore_root}; check database: ${database}"
