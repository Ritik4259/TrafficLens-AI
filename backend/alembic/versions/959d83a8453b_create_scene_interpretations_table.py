"""create scene_interpretations table

Revision ID: 959d83a8453b
Revises: ccecc71bbc1f
Create Date: 2026-11-23 17:45:54.893591

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import backend.app.models.base


# revision identifiers, used by Alembic.
revision: str = '959d83a8453b'
down_revision: Union[str, None] = 'ccecc71bbc1f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'scene_interpretations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('job_id', sa.String(length=36), nullable=False),
        sa.Column('status', sa.Enum('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', name='scene_interpretation_status_enum', native_enum=False), nullable=False),
        sa.Column('created_at', backend.app.models.base.UTCDateTime(timezone=True), nullable=False),
        sa.Column('started_at', backend.app.models.base.UTCDateTime(timezone=True), nullable=True),
        sa.Column('completed_at', backend.app.models.base.UTCDateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('provider_type', sa.String(length=50), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('environmental_context', sa.JSON(), nullable=True),
        sa.Column('scene_description', sa.JSON(), nullable=True),
        sa.Column('event_interpretations', sa.JSON(), nullable=True),
        sa.Column('epistemic_claims', sa.JSON(), nullable=True),
        sa.Column('unverified_aspects', sa.JSON(), nullable=True),
        sa.Column('evidence_package_id', sa.String(length=100), nullable=True),
        sa.Column('error_code', sa.String(length=100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['processing_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('scene_interpretations', schema=None) as batch_op:
        batch_op.create_index('ix_scene_interpretations_job_created', ['job_id', 'created_at'], unique=False)
        batch_op.create_index('ix_scene_interpretations_status_created', ['status', 'created_at'], unique=False)
        batch_op.create_index(
            'uix_active_interpretation_per_job',
            ['job_id'],
            unique=True,
            sqlite_where=sa.text("status IN ('QUEUED', 'PROCESSING')"),
            postgresql_where=sa.text("status IN ('QUEUED', 'PROCESSING')"),
        )


def downgrade() -> None:
    with op.batch_alter_table('scene_interpretations', schema=None) as batch_op:
        batch_op.drop_index('uix_active_interpretation_per_job')
        batch_op.drop_index('ix_scene_interpretations_status_created')
        batch_op.drop_index('ix_scene_interpretations_job_created')

    op.drop_table('scene_interpretations')
