"""add archival embeddings and matches

Revision ID: e5b7c91a42fd
Revises: d3a4f2b71c90
Create Date: 2026-08-31 08:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e5b7c91a42fd"
down_revision: Union[str, Sequence[str], None] = "d3a4f2b71c90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add immutable embedding row bindings and historical match candidates."""
    op.add_column(
        "catalog_versions",
        sa.Column("source_batch_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_catalog_versions_source_batch_id_batches"),
        "catalog_versions",
        "batches",
        ["source_batch_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_table(
        "crop_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("crop_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("feature_dim", sa.Integer(), nullable=False),
        sa.Column("embedding_sha256", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("preprocess_id", sa.String(length=128), nullable=False),
        sa.Column(
            "pipeline_config_digest", sa.String(length=64), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "feature_dim > 0",
            name=op.f("ck_crop_embeddings_positive_feature_dim"),
        ),
        sa.CheckConstraint(
            "row_index >= 0",
            name=op.f("ck_crop_embeddings_nonnegative_row_index"),
        ),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifacts.id"],
            name=op.f("fk_crop_embeddings_artifact_id_artifacts"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["crop_id"],
            ["crops.id"],
            name=op.f("fk_crop_embeddings_crop_id_crops"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_crop_embeddings")),
        sa.UniqueConstraint(
            "artifact_id",
            "row_index",
            name="crop_embeddings_artifact_row",
        ),
        sa.UniqueConstraint(
            "crop_id",
            "model_version",
            "preprocess_id",
            name="crop_embeddings_crop_model_preprocess",
        ),
    )
    op.create_table(
        "match_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("cluster_id", sa.Uuid(), nullable=False),
        sa.Column("catalog_id", sa.Uuid(), nullable=False),
        sa.Column("individual_id", sa.Uuid(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("support_frames", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rank > 0", name=op.f("ck_match_candidates_positive_rank")
        ),
        sa.CheckConstraint(
            "support_frames > 0",
            name=op.f("ck_match_candidates_positive_support_frames"),
        ),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalog_versions.id"],
            name=op.f(
                "fk_match_candidates_catalog_id_catalog_versions"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["cluster_id"],
            ["candidate_clusters.id"],
            name=op.f(
                "fk_match_candidates_cluster_id_candidate_clusters"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["individual_id"],
            ["confirmed_individuals.id"],
            name=op.f(
                "fk_match_candidates_individual_id_confirmed_individuals"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_match_candidates")),
        sa.UniqueConstraint(
            "cluster_id",
            "individual_id",
            name="match_candidates_cluster_individual",
        ),
        sa.UniqueConstraint(
            "cluster_id", "rank", name="match_candidates_cluster_rank"
        ),
    )


def downgrade() -> None:
    """Remove archival projection tables and catalog source link."""
    op.drop_table("match_candidates")
    op.drop_table("crop_embeddings")
    op.drop_constraint(
        op.f("fk_catalog_versions_source_batch_id_batches"),
        "catalog_versions",
        type_="foreignkey",
    )
    op.drop_column("catalog_versions", "source_batch_id")
