"""Add is_sample column to models table

Revision ID: add_is_sample_column
Revises: 
Create Date: 2023-04-03

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_is_sample_column'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add is_sample column with default value False
    op.add_column('models', sa.Column('is_sample', sa.Boolean(), nullable=True))
    
    # Update all existing rows to have is_sample=False
    op.execute("UPDATE models SET is_sample = 0 WHERE is_sample IS NULL")
    
    # Make the column non-nullable after setting values
    op.alter_column('models', 'is_sample', nullable=False, server_default=sa.text('0'))


def downgrade():
    # Drop the is_sample column
    op.drop_column('models', 'is_sample') 