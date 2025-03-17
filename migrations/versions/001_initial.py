"""initial

Revision ID: 001
Revises: 
Create Date: 2024-03-17 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create users table if it doesn't exist
    op.execute('CREATE TABLE IF NOT EXISTS users ('
               'id SERIAL PRIMARY KEY, '
               'name VARCHAR(100) NOT NULL, '
               'face_encoding TEXT NOT NULL, '
               'is_active BOOLEAN DEFAULT TRUE, '
               'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, '
               'last_access TIMESTAMP NULL'
               ')')

    # Create access_logs table if it doesn't exist
    op.execute('CREATE TABLE IF NOT EXISTS access_logs ('
               'id SERIAL PRIMARY KEY, '
               'user_id INTEGER NOT NULL, '
               'access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, '
               'access_type VARCHAR(20) NOT NULL, '
               'confidence FLOAT NOT NULL, '
               'status VARCHAR(20) NOT NULL, '
               'FOREIGN KEY (user_id) REFERENCES users(id)'
               ')')

def downgrade() -> None:
    op.drop_table('access_logs')
    op.drop_table('users') 