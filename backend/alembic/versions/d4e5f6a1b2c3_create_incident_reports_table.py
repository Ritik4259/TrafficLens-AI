"""create incident_reports table

Revision ID: d4e5f6a1b2c3
Revises: 959d83a8453b
Create Date: 2026-11-23 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import backend.app.models.base


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a1b2c3'
down_revision: Union[str, None] = '959d83a8453b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'incident_reports',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('job_id', sa.String(length=36), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('created_at', backend.app.models.base.UTCDateTime(timezone=True), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('report_data', sa.JSON(), nullable=False),
        sa.Column('validation_passed', sa.Boolean(), nullable=False, server_default=sa.text('1')),
        sa.Column('validation_summary', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['processing_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('incident_reports', schema=None) as batch_op:
        batch_op.create_index('ix_incident_reports_job_created', ['job_id', 'created_at'], unique=False)
        batch_op.create_index('ix_incident_reports_id', ['id'], unique=False)
        batch_op.create_index('ix_incident_reports_job_id', ['job_id'], unique=False)
        batch_op.create_index('ix_incident_reports_status', ['status'], unique=False)
        batch_op.create_index('ix_incident_reports_created_at', ['created_at'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('incident_reports', schema=None) as batch_op:
        batch_op.drop_index('ix_incident_reports_created_at')
        batch_op.drop_index('ix_incident_reports_status')
        batch_op.drop_index('ix_incident_reports_job_id')
        batch_op.drop_index('ix_incident_reports_id')
        batch_op.drop_index('ix_incident_reports_job_created')

    op.drop_table('incident_reports')
