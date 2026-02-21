"""initial: users + backtest_runs

Revision ID: 001
Revises:
Create Date: 2025-02-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("anonymous_id", sa.String(length=64), nullable=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_backtest_runs_anonymous_id"), "backtest_runs", ["anonymous_id"], unique=False)
    op.create_index("ix_backtest_runs_anon_created", "backtest_runs", ["anonymous_id", "created_at"], unique=False)
    op.create_index(op.f("ix_backtest_runs_user_id"), "backtest_runs", ["user_id"], unique=False)
    op.create_index("ix_backtest_runs_user_created", "backtest_runs", ["user_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_backtest_runs_user_created", table_name="backtest_runs")
    op.drop_index(op.f("ix_backtest_runs_user_id"), table_name="backtest_runs")
    op.drop_index("ix_backtest_runs_anon_created", table_name="backtest_runs")
    op.drop_index(op.f("ix_backtest_runs_anonymous_id"), table_name="backtest_runs")
    op.drop_table("backtest_runs")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
