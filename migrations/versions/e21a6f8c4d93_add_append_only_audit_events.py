"""add append-only audit events

Revision ID: e21a6f8c4d93
Revises: d20f4ea783b2
Create Date: 2026-08-31 22:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e21a6f8c4d93"
down_revision: Union[str, Sequence[str], None] = "d20f4ea783b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("actor_type", sa.String(length=32), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("actor_worker_id", sa.Uuid(), nullable=True),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=True),
        sa.Column("target_id", sa.String(length=256), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "actor_type IN ('user', 'worker', 'system', 'anonymous')",
            name=op.f("ck_audit_events_valid_actor_type")),
        sa.CheckConstraint(
            "actor_user_id IS NULL OR actor_worker_id IS NULL",
            name=op.f("ck_audit_events_at_most_one_actor")),
        sa.ForeignKeyConstraint(
            ["actor_user_id"], ["users.id"], ondelete="RESTRICT",
            name=op.f("fk_audit_events_actor_user_id_users")),
        sa.ForeignKeyConstraint(
            ["actor_worker_id"], ["worker_devices.id"], ondelete="RESTRICT",
            name=op.f("fk_audit_events_actor_worker_id_worker_devices")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_index(
        "ix_audit_events_occurred_at", "audit_events", ["occurred_at"])
    op.create_index(
        "ix_audit_events_event_type", "audit_events", ["event_type"])
    op.execute("""
        CREATE FUNCTION whitewhale_reject_audit_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'audit_events is append-only';
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER audit_events_append_only
        BEFORE UPDATE OR DELETE ON audit_events
        FOR EACH ROW EXECUTE FUNCTION whitewhale_reject_audit_mutation()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER audit_events_append_only ON audit_events")
    op.drop_index("ix_audit_events_event_type", table_name="audit_events")
    op.drop_index("ix_audit_events_occurred_at", table_name="audit_events")
    op.drop_table("audit_events")
    op.execute("DROP FUNCTION whitewhale_reject_audit_mutation()")
