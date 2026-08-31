"""add query requests

Revision ID: f22b7a9d5e04
Revises: e21a6f8c4d93
Create Date: 2026-09-01 00:15:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f22b7a9d5e04"
down_revision: Union[str, Sequence[str], None] = "e21a6f8c4d93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "query_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("upload_session_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("catalog_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("detector_version", sa.String(length=128), nullable=False),
        sa.Column("preprocess_id", sa.String(length=128), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("input_digest", sa.String(length=64), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "state IN ('queued', 'running', 'succeeded', 'failed')",
            name=op.f("ck_query_requests_valid_state")),
        sa.CheckConstraint(
            "top_k > 0 AND top_k <= 100",
            name=op.f("ck_query_requests_valid_top_k")),
        sa.ForeignKeyConstraint(["catalog_id"], ["catalog_versions.id"],
                                ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"],
                                ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"],
                                ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["upload_session_id"], ["upload_sessions.id"],
                                ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
        sa.UniqueConstraint("upload_session_id"),
    )
    op.create_table(
        "query_images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("query_request_id", sa.Uuid(), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("original_relative_path", sa.Text(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("row_index >= 0",
                           name=op.f("ck_query_images_nonnegative_row_index")),
        sa.CheckConstraint("size_bytes >= 0",
                           name=op.f("ck_query_images_nonnegative_size_bytes")),
        sa.ForeignKeyConstraint(["query_request_id"], ["query_requests.id"],
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "query_request_id", "original_relative_path",
            name="query_images_request_path"),
        sa.UniqueConstraint(
            "query_request_id", "row_index",
            name="query_images_request_row"),
    )


def downgrade() -> None:
    op.drop_table("query_images")
    op.drop_table("query_requests")
