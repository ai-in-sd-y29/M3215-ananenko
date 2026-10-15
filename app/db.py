import asyncpg

from app.config import Settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_settings (
    chat_id BIGINT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'study',
    temperature DOUBLE PRECISION NOT NULL DEFAULT 0.7,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dialog_messages (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS dialog_messages_chat_created_idx
    ON dialog_messages (chat_id, created_at);

ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS timezone TEXT;

CREATE TABLE IF NOT EXISTS pending_actions (
    id TEXT PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    source_key TEXT NOT NULL UNIQUE,
    tool_name TEXT NOT NULL,
    arguments JSONB NOT NULL,
    timezone TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'executing', 'done', 'cancelled', 'expired', 'failed')
    ),
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS pending_actions_chat_status_idx
    ON pending_actions (chat_id, status);

CREATE TABLE IF NOT EXISTS reminders (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    text TEXT NOT NULL CHECK (char_length(text) BETWEEN 1 AND 500),
    remind_at TIMESTAMPTZ NOT NULL,
    timezone TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS reminders_chat_remind_idx ON reminders (chat_id, remind_at);

CREATE TABLE IF NOT EXISTS agent_audit (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    request_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action TEXT NOT NULL,
    tool_name TEXT,
    args_summary TEXT,
    validation_status TEXT NOT NULL,
    execution_status TEXT NOT NULL,
    reason TEXT,
    mcp_calls INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS agent_audit_chat_created_idx ON agent_audit (chat_id, created_at DESC);
"""


async def create_pool(settings: Settings) -> asyncpg.Pool:
    # Небольшого пула достаточно для учебной ВМ. Таблицы добавляют студенты.
    pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        min_size=1,
        max_size=5,
        timeout=5,
        command_timeout=5,
    )
    try:
        await pool.fetchval("SELECT 1", timeout=3)
    except BaseException:
        await pool.close()
        raise
    return pool


async def init_schema(pool: asyncpg.Pool) -> None:
    await pool.execute(SCHEMA_SQL)
