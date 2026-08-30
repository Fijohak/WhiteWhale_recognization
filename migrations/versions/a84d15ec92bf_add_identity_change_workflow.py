"""add identity change workflow

Revision ID: a84d15ec92bf
Revises: f7c2a8d14b63
Create Date: 2026-08-31 14:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a84d15ec92bf"
down_revision: Union[str, Sequence[str], None] = "f7c2a8d14b63"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "observations",
        sa.Column(
            "state", sa.String(length=32), nullable=False,
            server_default="active",
        ),
    )
    op.create_check_constraint(
        op.f("ck_observations_valid_state"),
        "observations",
        "state IN ('active', 'withdrawn')",
    )
    op.alter_column("observations", "state", server_default=None)
    op.create_table(
        "identity_change_proposals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("change_type", sa.String(length=32), nullable=False),
        sa.Column("plan", sa.JSON(), nullable=False),
        sa.Column("plan_digest", sa.String(length=64), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.CheckConstraint(
            "change_type IN ('merge', 'split', 'withdrawal')",
            name=op.f("ck_identity_change_proposals_valid_change_type"),
        ),
        sa.CheckConstraint(
            "status IN ('review_pending', 'applied', 'rejected', 'disputed')",
            name=op.f("ck_identity_change_proposals_valid_status"),
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"],
            name=op.f(
                "fk_identity_change_proposals_created_by_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_identity_change_proposals")),
    )
    op.create_table(
        "identity_change_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("proposal_id", sa.Uuid(), nullable=False),
        sa.Column("review_task_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"], ["users.id"],
            name=op.f("fk_identity_change_events_actor_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["proposal_id"], ["identity_change_proposals.id"],
            name=op.f(
                "fk_identity_change_events_proposal_id_identity_change_proposals"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["review_task_id"], ["review_tasks.id"],
            name=op.f(
                "fk_identity_change_events_review_task_id_review_tasks"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_identity_change_events")),
        sa.UniqueConstraint(
            "proposal_id",
            name=op.f("uq_identity_change_events_proposal_id")),
        sa.UniqueConstraint(
            "review_task_id",
            name=op.f("uq_identity_change_events_review_task_id")),
    )


def downgrade() -> None:
    op.drop_table("identity_change_events")
    op.drop_table("identity_change_proposals")
    op.drop_constraint(
        op.f("ck_observations_valid_state"),
        "observations", type_="check",
    )
    op.drop_column("observations", "state")
