"""add training and model lifecycle

Revision ID: c19e75b4d2a1
Revises: a84d15ec92bf
Create Date: 2026-08-31 18:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c19e75b4d2a1"
down_revision: Union[str, Sequence[str], None] = "a84d15ec92bf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("protocol", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("catalog_id", sa.Uuid(), nullable=True),
        sa.Column("membership_digest", sa.String(length=64), nullable=False),
        sa.Column("rights_snapshot", sa.JSON(), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("frozen_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "protocol IN ('known_identity_update', 'open_set_unknown')",
            name=op.f("ck_dataset_versions_valid_protocol")),
        sa.CheckConstraint(
            "status IN ('frozen', 'retired')",
            name=op.f("ck_dataset_versions_valid_status")),
        sa.ForeignKeyConstraint(
            ["catalog_id"], ["catalog_versions.id"],
            name=op.f("fk_dataset_versions_catalog_id_catalog_versions"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"],
            name=op.f("fk_dataset_versions_created_by_user_id_users"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_versions")),
        sa.UniqueConstraint(
            "membership_digest",
            name=op.f("uq_dataset_versions_membership_digest")),
    )
    op.create_table(
        "dataset_memberships",
        sa.Column("dataset_version_id", sa.Uuid(), nullable=False),
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("crop_id", sa.Uuid(), nullable=False),
        sa.Column("individual_id", sa.Uuid(), nullable=False),
        sa.Column("label_source", sa.String(length=64), nullable=False),
        sa.Column("sequence_key", sa.String(length=256), nullable=False),
        sa.Column("encounter_key", sa.String(length=256), nullable=False),
        sa.Column("duplicate_group", sa.String(length=256), nullable=False),
        sa.Column("data_license", sa.String(length=128), nullable=False),
        sa.CheckConstraint(
            "label_source IN ('provider_confirmed', 'project_verified', "
            "'high_trust_pseudo_label')",
            name=op.f("ck_dataset_memberships_valid_label_source")),
        sa.ForeignKeyConstraint(
            ["crop_id"], ["crops.id"],
            name=op.f("fk_dataset_memberships_crop_id_crops"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"],
            name=op.f(
                "fk_dataset_memberships_dataset_version_id_dataset_versions"),
            ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["image_id"], ["images.id"],
            name=op.f("fk_dataset_memberships_image_id_images"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["individual_id"], ["confirmed_individuals.id"],
            name=op.f(
                "fk_dataset_memberships_individual_id_confirmed_individuals"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["observation_id"], ["observations.id"],
            name=op.f("fk_dataset_memberships_observation_id_observations"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint(
            "dataset_version_id", "observation_id",
            name=op.f("pk_dataset_memberships")),
        sa.UniqueConstraint(
            "dataset_version_id", "crop_id",
            name="dataset_memberships_dataset_crop"),
    )
    op.create_table(
        "dataset_splits",
        sa.Column("dataset_version_id", sa.Uuid(), nullable=False),
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "split IN ('train', 'val', 'calibration', 'test')",
            name=op.f("ck_dataset_splits_valid_split")),
        sa.ForeignKeyConstraint(
            ["dataset_version_id", "observation_id"],
            ["dataset_memberships.dataset_version_id",
             "dataset_memberships.observation_id"],
            name=op.f(
                "fk_dataset_splits_dataset_version_id_dataset_memberships"),
            ondelete="CASCADE"),
        sa.PrimaryKeyConstraint(
            "dataset_version_id", "observation_id",
            name=op.f("pk_dataset_splits")),
    )
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version_id", sa.Uuid(), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column("model_family", sa.String(length=128), nullable=False),
        sa.Column("base_model_version_id", sa.Uuid(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("required_vram_mb", sa.Integer(), nullable=False),
        sa.Column("max_runtime_seconds", sa.Integer(), nullable=False),
        sa.Column("checkpoint_interval_steps", sa.Integer(), nullable=False),
        sa.Column("resume_checkpoint_id", sa.Uuid(), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "required_vram_mb > 0 AND max_runtime_seconds > 0 "
            "AND checkpoint_interval_steps > 0",
            name=op.f("ck_training_runs_positive_limits")),
        sa.CheckConstraint(
            "state IN ('queued', 'running', 'completed', 'failed')",
            name=op.f("ck_training_runs_valid_state")),
        sa.CheckConstraint(
            "task_type IN ('detector_training', 'reid_training')",
            name=op.f("ck_training_runs_valid_task_type")),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"],
            name=op.f("fk_training_runs_dataset_version_id_dataset_versions"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"],
            name=op.f("fk_training_runs_job_id_jobs"), ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_runs")),
        sa.UniqueConstraint("job_id", name=op.f("uq_training_runs_job_id")),
    )
    op.create_table(
        "training_checkpoints",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("training_run_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.Integer(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "stage >= 0 AND epoch >= 0 AND step >= 0",
            name=op.f("ck_training_checkpoints_nonnegative_step")),
        sa.ForeignKeyConstraint(
            ["artifact_id"], ["artifacts.id"],
            name=op.f("fk_training_checkpoints_artifact_id_artifacts"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["training_run_id"], ["training_runs.id"],
            name=op.f("fk_training_checkpoints_training_run_id_training_runs"),
            ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_checkpoints")),
        sa.UniqueConstraint(
            "artifact_id", name=op.f("uq_training_checkpoints_artifact_id")),
        sa.UniqueConstraint(
            "training_run_id", "stage", "epoch", "step",
            name="training_checkpoints_run_stage_epoch_step"),
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("training_run_id", sa.Uuid(), nullable=False),
        sa.Column("weight_artifact_id", sa.Uuid(), nullable=False),
        sa.Column("model_family", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("weight_path", sa.Text(), nullable=False),
        sa.Column("feature_dim", sa.Integer(), nullable=True),
        sa.Column("preprocess_id", sa.String(length=128), nullable=False),
        sa.Column("checkpoint_source", sa.String(length=256), nullable=False),
        sa.Column("license", sa.String(length=256), nullable=False),
        sa.Column("compatible_detector_version", sa.String(length=128),
                  nullable=True),
        sa.Column("compatible_crop_config", sa.String(length=128),
                  nullable=False),
        sa.Column("compatible_index_schema", sa.Integer(), nullable=False),
        sa.Column("calibrated_thresholds", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "compatible_index_schema > 0",
            name=op.f("ck_model_versions_positive_index_schema")),
        sa.CheckConstraint(
            "feature_dim IS NULL OR feature_dim > 0",
            name=op.f("ck_model_versions_positive_dim")),
        sa.CheckConstraint(
            "status IN ('candidate', 'promotion_pending', 'production', "
            "'retired', 'rejected')",
            name=op.f("ck_model_versions_valid_status")),
        sa.ForeignKeyConstraint(
            ["training_run_id"], ["training_runs.id"],
            name=op.f("fk_model_versions_training_run_id_training_runs"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["weight_artifact_id"], ["artifacts.id"],
            name=op.f("fk_model_versions_weight_artifact_id_artifacts"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint(
            "training_run_id",
            name=op.f("uq_model_versions_training_run_id")),
        sa.UniqueConstraint(
            "version", name=op.f("uq_model_versions_version")),
        sa.UniqueConstraint(
            "weight_artifact_id",
            name=op.f("uq_model_versions_weight_artifact_id")),
    )
    op.create_foreign_key(
        "fk_training_runs_base_model_version_id_model_versions",
        "training_runs", "model_versions",
        ["base_model_version_id"], ["id"], ondelete="RESTRICT")
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version_id", sa.Uuid(), nullable=False),
        sa.Column("protocol", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("report_artifact_id", sa.Uuid(), nullable=True),
        sa.Column("comparison", sa.JSON(), nullable=False),
        sa.Column("calibrated_thresholds", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')",
            name=op.f("ck_evaluation_runs_valid_status")),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"],
            name=op.f("fk_evaluation_runs_dataset_version_id_dataset_versions"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"],
            name=op.f("fk_evaluation_runs_job_id_jobs"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["model_version_id"], ["model_versions.id"],
            name=op.f("fk_evaluation_runs_model_version_id_model_versions"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["report_artifact_id"], ["artifacts.id"],
            name=op.f("fk_evaluation_runs_report_artifact_id_artifacts"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_runs")),
        sa.UniqueConstraint("job_id", name=op.f("uq_evaluation_runs_job_id")),
    )
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("evaluation_run_id", sa.Uuid(), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["evaluation_run_id"], ["evaluation_runs.id"],
            name=op.f("fk_evaluation_results_evaluation_run_id_evaluation_runs"),
            ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_results")),
        sa.UniqueConstraint(
            "evaluation_run_id", "metric_name", "split",
            name="evaluation_results_run_metric_split"),
    )
    op.create_table(
        "model_promotion_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=False),
        sa.Column("catalog_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["actor_user_id"], ["users.id"],
            name=op.f("fk_model_promotion_events_actor_user_id_users"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["catalog_id"], ["catalog_versions.id"],
            name=op.f("fk_model_promotion_events_catalog_id_catalog_versions"),
            ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["model_version_id"], ["model_versions.id"],
            name=op.f(
                "fk_model_promotion_events_model_version_id_model_versions"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_model_promotion_events")),
    )
    op.create_table(
        "production_model_pointer",
        sa.Column("model_family", sa.String(length=128), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"], ["model_versions.id"],
            name=op.f(
                "fk_production_model_pointer_model_version_id_model_versions"),
            ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint(
            "model_family", name=op.f("pk_production_model_pointer")),
        sa.UniqueConstraint(
            "model_version_id",
            name=op.f("uq_production_model_pointer_model_version_id")),
    )


def downgrade() -> None:
    op.drop_table("production_model_pointer")
    op.drop_table("model_promotion_events")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
    op.drop_constraint(
        "fk_training_runs_base_model_version_id_model_versions",
        "training_runs", type_="foreignkey")
    op.drop_table("model_versions")
    op.drop_table("training_checkpoints")
    op.drop_table("training_runs")
    op.drop_table("dataset_splits")
    op.drop_table("dataset_memberships")
    op.drop_table("dataset_versions")
