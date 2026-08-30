"""add cooccurrence and relationships

Revision ID: f7c2a8d14b63
Revises: e5b7c91a42fd
Create Date: 2026-08-31 10:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f7c2a8d14b63"
down_revision: Union[str, Sequence[str], None] = "e5b7c91a42fd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("system_key", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_collections")),
        sa.UniqueConstraint("system_key", name=op.f("uq_collections_system_key")),
    )
    op.create_table(
        "cooccurrence_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("provenance_artifact_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('review_pending', 'confirmed', 'rejected', 'disputed')",
            name=op.f("ck_cooccurrence_events_valid_status")),
        sa.ForeignKeyConstraint(
            ["image_id"], ["images.id"],
            name=op.f("fk_cooccurrence_events_image_id_images"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["provenance_artifact_id"], ["artifacts.id"],
            name=op.f(
                "fk_cooccurrence_events_provenance_artifact_id_artifacts"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cooccurrence_events")),
        sa.UniqueConstraint(
            "image_id", name=op.f("uq_cooccurrence_events_image_id")),
    )
    op.create_table(
        "relationship_hypotheses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("individual_low_id", sa.Uuid(), nullable=False),
        sa.Column("individual_high_id", sa.Uuid(), nullable=False),
        sa.Column("relationship_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "individual_low_id <> individual_high_id",
            name=op.f("ck_relationship_hypotheses_distinct_pair")),
        sa.CheckConstraint(
            "status IN ('suspected', 'evidence_insufficient', 'disputed', "
            "'rejected')",
            name=op.f("ck_relationship_hypotheses_valid_status")),
        sa.CheckConstraint(
            "relationship_type IN ('co_occurrence', 'repeated_association', "
            "'suspected_kinship')",
            name=op.f("ck_relationship_hypotheses_valid_type")),
        sa.ForeignKeyConstraint(
            ["individual_high_id"], ["confirmed_individuals.id"],
            name=op.f(
                "fk_relationship_hypotheses_individual_high_id_confirmed_individuals"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["individual_low_id"], ["confirmed_individuals.id"],
            name=op.f(
                "fk_relationship_hypotheses_individual_low_id_confirmed_individuals"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_relationship_hypotheses")),
        sa.UniqueConstraint(
            "individual_low_id", "individual_high_id", "relationship_type",
            name="relationship_hypotheses_pair_type"),
    )
    op.create_table(
        "collection_memberships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=True),
        sa.Column("crop_id", sa.Uuid(), nullable=True),
        sa.Column("assignment_source", sa.String(length=64), nullable=False),
        sa.Column("membership_status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "(image_id IS NULL) <> (crop_id IS NULL)",
            name=op.f("ck_collection_memberships_exactly_one_subject")),
        sa.CheckConstraint(
            "membership_status IN ('candidate', 'review_pending', "
            "'confirmed_member', 'rejected')",
            name=op.f("ck_collection_memberships_valid_status")),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["collections.id"],
            name=op.f(
                "fk_collection_memberships_collection_id_collections"),
            ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["crop_id"], ["crops.id"],
            name=op.f("fk_collection_memberships_crop_id_crops"),
            ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["image_id"], ["images.id"],
            name=op.f("fk_collection_memberships_image_id_images"),
            ondelete="CASCADE"),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_collection_memberships")),
        sa.UniqueConstraint(
            "collection_id", "crop_id",
            name="collection_memberships_collection_crop"),
        sa.UniqueConstraint(
            "collection_id", "image_id",
            name="collection_memberships_collection_image"),
    )
    op.create_table(
        "cooccurrence_members",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("crop_id", sa.Uuid(), nullable=False),
        sa.Column("individual_id", sa.Uuid(), nullable=True),
        sa.Column("membership_status", sa.String(length=32), nullable=False),
        sa.Column("source_review_task_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "membership_status IN ('candidate', 'confirmed_member', 'rejected')",
            name=op.f("ck_cooccurrence_members_valid_status")),
        sa.ForeignKeyConstraint(
            ["crop_id"], ["crops.id"],
            name=op.f("fk_cooccurrence_members_crop_id_crops"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["event_id"], ["cooccurrence_events.id"],
            name=op.f(
                "fk_cooccurrence_members_event_id_cooccurrence_events"),
            ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["individual_id"], ["confirmed_individuals.id"],
            name=op.f(
                "fk_cooccurrence_members_individual_id_confirmed_individuals"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["source_review_task_id"], ["review_tasks.id"],
            name=op.f(
                "fk_cooccurrence_members_source_review_task_id_review_tasks"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint(
            "event_id", "crop_id", name=op.f("pk_cooccurrence_members")),
    )
    op.create_table(
        "relationship_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("hypothesis_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["actor_user_id"], ["users.id"],
            name=op.f("fk_relationship_events_actor_user_id_users"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["hypothesis_id"], ["relationship_hypotheses.id"],
            name=op.f(
                "fk_relationship_events_hypothesis_id_relationship_hypotheses"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_relationship_events")),
    )
    op.create_table(
        "relationship_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("hypothesis_id", sa.Uuid(), nullable=False),
        sa.Column("cooccurrence_event_id", sa.Uuid(), nullable=False),
        sa.Column("evidence_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["cooccurrence_event_id"], ["cooccurrence_events.id"],
            name=op.f(
                "fk_relationship_evidence_cooccurrence_event_id_cooccurrence_events"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["hypothesis_id"], ["relationship_hypotheses.id"],
            name=op.f(
                "fk_relationship_evidence_hypothesis_id_relationship_hypotheses"),
            ondelete="CASCADE"),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_relationship_evidence")),
        sa.UniqueConstraint(
            "hypothesis_id", "cooccurrence_event_id",
            name=op.f("uq_relationship_evidence_hypothesis_id")),
    )


def downgrade() -> None:
    op.drop_table("relationship_evidence")
    op.drop_table("relationship_events")
    op.drop_table("cooccurrence_members")
    op.drop_table("collection_memberships")
    op.drop_table("relationship_hypotheses")
    op.drop_table("cooccurrence_events")
    op.drop_table("collections")
