#!/usr/bin/env bash
set -euo pipefail

: "${DEPLOY_BRANCH:?DEPLOY_BRANCH is required}"
: "${WHITEWHALE_REPO:?WHITEWHALE_REPO is required}"
: "${WHITEWHALE_RELEASES:?WHITEWHALE_RELEASES is required}"
: "${WHITEWHALE_CURRENT:?WHITEWHALE_CURRENT is required}"
: "${WHITEWHALE_SHARED_ENV:?WHITEWHALE_SHARED_ENV is required}"

deploy_remote="${DEPLOY_REMOTE:-origin}"
keep_releases="${WHITEWHALE_KEEP_RELEASES:-5}"
ready_url="${WHITEWHALE_READY_URL:-https://127.0.0.1/ready}"
lock_file="/var/lock/whitewhale-deploy.lock"
status_file="${WHITEWHALE_DEPLOY_STATUS_FILE:-/srv/whitewhale/data/working/deploy-status.json}"
status_writer="${WHITEWHALE_REPO}/scripts/write_deploy_status.py"
target_commit="unknown"
deployed_at="$(date --utc +%Y-%m-%dT%H:%M:%SZ)"

write_deploy_status() {
  local state="$1"
  local reason="${2:-}"
  local -a status_args=(
    "${status_file}" --status "${state}" --branch "${DEPLOY_BRANCH}"
    --commit "${target_commit}" --deployed-at "${deployed_at}"
  )
  if [[ -n "${reason}" ]]; then
    status_args+=(--failure-reason "${reason}")
  fi
  python3 "${status_writer}" "${status_args[@]}"
}

on_error() {
  local exit_code="$?"
  local line="$1"
  local command="$2"
  trap - ERR
  write_deploy_status failed "line ${line}: ${command}" || true
  exit "${exit_code}"
}

trap 'on_error "${LINENO}" "${BASH_COMMAND}"' ERR

exec 9>"${lock_file}"
flock -n 9 || exit 0

git -C "${WHITEWHALE_REPO}" fetch --prune "${deploy_remote}" "${DEPLOY_BRANCH}"
target_commit="$(git -C "${WHITEWHALE_REPO}" rev-parse "FETCH_HEAD^{commit}")"
if [[ ! "${target_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "remote commit is invalid" >&2
  exit 1
fi

current_release=""
current_commit=""
if [[ -L "${WHITEWHALE_CURRENT}" ]]; then
  current_release="$(readlink -f "${WHITEWHALE_CURRENT}")"
  if [[ -f "${current_release}/.release-commit" ]]; then
    current_commit="$(<"${current_release}/.release-commit")"
  fi
fi
if [[ "${current_commit}" == "${target_commit}" ]]; then
  exit 0
fi

mkdir -p "${WHITEWHALE_RELEASES}"
build_root="$(mktemp -d "${WHITEWHALE_RELEASES}/.build-${target_commit:0:12}-XXXXXX")"
cleanup() { rm -rf -- "${build_root}"; }
trap cleanup EXIT
git -C "${WHITEWHALE_REPO}" archive "${target_commit}" | tar -x -C "${build_root}"
printf '%s\n' "${target_commit}" > "${build_root}/.release-commit"

if [[ -n "${current_commit}" ]]; then
  mapfile -t migration_files < <(git -C "${WHITEWHALE_REPO}" diff --name-only \
    "${current_commit}" "${target_commit}" -- 'migrations/versions/*.py')
  if ((${#migration_files[@]})); then
    migration_paths=()
    for file in "${migration_files[@]}"; do
      [[ -f "${build_root}/${file}" ]] && migration_paths+=("${build_root}/${file}")
    done
    if ((${#migration_paths[@]})); then
      python3 "${build_root}/scripts/check_expand_only_migrations.py" \
        "${migration_paths[@]}"
    fi
  fi
fi

release_path="${WHITEWHALE_RELEASES}/${target_commit}"
if [[ ! -d "${release_path}" ]]; then
  mv "${build_root}" "${release_path}"
  build_root="$(mktemp -d "${WHITEWHALE_RELEASES}/.cleanup-XXXXXX")"
fi

compose=(docker compose --env-file "${WHITEWHALE_SHARED_ENV}" -f "${release_path}/compose.yaml")
WHITEWHALE_RELEASE_TAG="${target_commit}" \
WHITEWHALE_DEPLOYED_AT="${deployed_at}" \
DEPLOY_BRANCH="${DEPLOY_BRANCH}" "${compose[@]}" build api web
WHITEWHALE_RELEASE_TAG="${target_commit}" \
WHITEWHALE_DEPLOYED_AT="${deployed_at}" \
DEPLOY_BRANCH="${DEPLOY_BRANCH}" "${compose[@]}" run --rm --no-deps api \
  python /app/scripts/release_smoke.py
WHITEWHALE_RELEASE_TAG="${target_commit}" \
WHITEWHALE_DEPLOYED_AT="${deployed_at}" \
DEPLOY_BRANCH="${DEPLOY_BRANCH}" "${compose[@]}" run --rm api \
  python -m alembic upgrade head

next_link="${WHITEWHALE_CURRENT}.next"
ln -sfn "${release_path}" "${next_link}"
mv -Tf "${next_link}" "${WHITEWHALE_CURRENT}"

if ! WHITEWHALE_RELEASE_TAG="${target_commit}" \
    WHITEWHALE_DEPLOYED_AT="${deployed_at}" DEPLOY_BRANCH="${DEPLOY_BRANCH}" \
    "${compose[@]}" up -d --no-build; then
  deploy_failed=1
else
  deploy_failed=0
  for _ in {1..30}; do
    if curl --fail --silent --insecure --max-time 5 "${ready_url}" >/dev/null; then
      deploy_failed=0
      break
    fi
    deploy_failed=1
    sleep 2
  done
fi

if ((deploy_failed)); then
  if [[ -n "${current_release}" && -d "${current_release}" ]]; then
    previous_tag="$(<"${current_release}/.release-commit")"
    ln -sfn "${current_release}" "${next_link}"
    mv -Tf "${next_link}" "${WHITEWHALE_CURRENT}"
    WHITEWHALE_RELEASE_TAG="${previous_tag}" \
      DEPLOY_BRANCH="${DEPLOY_BRANCH}" \
      docker compose --env-file "${WHITEWHALE_SHARED_ENV}" \
      -f "${current_release}/compose.yaml" up -d --no-build
  else
    unlink "${WHITEWHALE_CURRENT}"
  fi
  write_deploy_status failed \
    "new release failed readiness; previous release restored"
  echo "new release failed readiness; previous release restored" >&2
  exit 1
fi

write_deploy_status deployed

mapfile -t old_releases < <(find "${WHITEWHALE_RELEASES}" -mindepth 1 -maxdepth 1 \
  -type d -name '[0-9a-f]*' -printf '%T@ %p\n' | sort -nr | tail -n +$((keep_releases + 1)) | cut -d' ' -f2-)
for old_release in "${old_releases[@]}"; do
  [[ "${old_release}" == "${release_path}" || "${old_release}" == "${current_release}" ]] \
    || rm -rf -- "${old_release}"
done
