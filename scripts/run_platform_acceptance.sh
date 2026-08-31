#!/usr/bin/env bash
set -euo pipefail

: "${WHITEWHALE_TEST_DATABASE_URL:?set WHITEWHALE_TEST_DATABASE_URL}"
python_bin="${WHITEWHALE_PYTHON:-python3}"
"${python_bin}" -m pytest -q \
  tests/test_platform_postgres.py \
  tests/test_platform_worker_api.py \
  tests/test_platform_worker_auth.py \
  tests/test_platform_artifacts.py \
  tests/test_platform_uploads.py \
  tests/test_platform_archival_dispatch.py \
  tests/test_platform_archival_workflow.py \
  tests/test_platform_catalogs.py \
  tests/test_platform_cooccurrence.py \
  tests/test_platform_identity_changes.py \
  tests/test_platform_training.py \
  tests/test_platform_auth.py \
  tests/test_m5_delivery.py
