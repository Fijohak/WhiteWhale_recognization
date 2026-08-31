"""add legacy artifact registry

Revision ID: d20f4ea783b2
Revises: c19e75b4d2a1
Create Date: 2026-08-31 21:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d20f4ea783b2"
down_revision: Union[str, Sequence[str], None] = "c19e75b4d2a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "legacy_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_kind", sa.String(length=64), nullable=False),
        sa.Column("source_name", sa.String(length=512), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("calibration_status", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "calibration_status IN ('not_applicable', "
            "'provisional_unvalidated', 'calibrated')",
            name=op.f("ck_legacy_artifacts_valid_calibration_status")),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name=op.f("ck_legacy_artifacts_nonnegative_size")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_legacy_artifacts")),
        sa.UniqueConstraint(
            "artifact_kind", "sha256",
            name=op.f("uq_legacy_artifacts_artifact_kind")),
    )


def downgrade() -> None:
    op.drop_table("legacy_artifacts")
