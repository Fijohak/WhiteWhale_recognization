"""add catalog lifecycle

Revision ID: d3a4f2b71c90
Revises: 9d6e7b4ff6ee
Create Date: 2026-08-31 08:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d3a4f2b71c90"
down_revision: Union[str, Sequence[str], None] = "9d6e7b4ff6ee"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create immutable catalog snapshots and their active pointer."""
    op.create_table(
        "catalog_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("calibration_status", sa.String(length=64), nullable=False),
        sa.Column("feature_dim", sa.Integer(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("membership_digest", sa.String(length=64), nullable=False),
        sa.Column("index_path", sa.Text(), nullable=False),
        sa.Column("index_sha256", sa.String(length=64), nullable=False),
        sa.Column("parent_catalog_id", sa.Uuid(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "feature_dim > 0",
            name=op.f("ck_catalog_versions_positive_feature_dim"),
        ),
        sa.CheckConstraint(
            "row_count > 0",
            name=op.f("ck_catalog_versions_positive_row_count"),
        ),
        sa.CheckConstraint(
            "status IN ('staged', 'active', 'retired')",
            name=op.f("ck_catalog_versions_valid_status"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_catalog_id"],
            ["catalog_versions.id"],
            name=op.f(
                "fk_catalog_versions_parent_catalog_id_catalog_versions"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_versions")),
    )
    op.create_table(
        "active_catalog_pointer",
        sa.Column("singleton_id", sa.Integer(), nullable=False),
        sa.Column("catalog_id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "singleton_id = 1",
            name=op.f("ck_active_catalog_pointer_singleton"),
        ),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalog_versions.id"],
            name=op.f(
                "fk_active_catalog_pointer_catalog_id_catalog_versions"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "singleton_id", name=op.f("pk_active_catalog_pointer")
        ),
        sa.UniqueConstraint(
            "catalog_id", name=op.f("uq_active_catalog_pointer_catalog_id")
        ),
    )
    op.create_table(
        "catalog_memberships",
        sa.Column("catalog_id", sa.Uuid(), nullable=False),
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("individual_id", sa.Uuid(), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("embedding_sha256", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "row_index >= 0",
            name=op.f("ck_catalog_memberships_nonnegative_row_index"),
        ),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalog_versions.id"],
            name=op.f(
                "fk_catalog_memberships_catalog_id_catalog_versions"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["individual_id"],
            ["confirmed_individuals.id"],
            name=op.f(
                "fk_catalog_memberships_individual_id_confirmed_individuals"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["observation_id"],
            ["observations.id"],
            name=op.f(
                "fk_catalog_memberships_observation_id_observations"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "catalog_id",
            "observation_id",
            name=op.f("pk_catalog_memberships"),
        ),
        sa.UniqueConstraint(
            "catalog_id",
            "row_index",
            name=op.f("uq_catalog_memberships_catalog_id"),
        ),
    )
    op.create_table(
        "catalog_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("catalog_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("previous_catalog_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalog_versions.id"],
            name=op.f("fk_catalog_events_catalog_id_catalog_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["previous_catalog_id"],
            ["catalog_versions.id"],
            name=op.f(
                "fk_catalog_events_previous_catalog_id_catalog_versions"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_events")),
    )


def downgrade() -> None:
    """Remove catalog lifecycle tables in dependency order."""
    op.drop_table("catalog_events")
    op.drop_table("catalog_memberships")
    op.drop_table("active_catalog_pointer")
    op.drop_table("catalog_versions")
