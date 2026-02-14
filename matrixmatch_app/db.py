from contextlib import contextmanager
import logging
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from matrixmatch_app.config import get_db_config

logger = logging.getLogger(__name__)
_pool = None
_pooled_conn_ids = set()
_pool_init_failed = False


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r. Falling back to %d.", name, raw, default)
        return default


def _get_pool():
    global _pool, _pool_init_failed
    if _pool_init_failed:
        return None

    if _pool is None:
        minconn = max(_get_int_env("DB_POOL_MIN", 1), 1)
        maxconn = max(_get_int_env("DB_POOL_MAX", 10), minconn)
        try:
            _pool = ThreadedConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                cursor_factory=RealDictCursor,
                **get_db_config(),
            )
        except Exception:
            # Keep app usable even if pool creation fails on first request.
            _pool_init_failed = True
            logger.exception("Failed to initialize DB pool. Falling back to direct connections.")
            return None
    return _pool


def get_db_connection():
    """Get one pooled DB connection.

    Callers are responsible for closing via close_db_connection().
    """
    pool = _get_pool()
    if pool is None:
        return psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor)

    conn = pool.getconn()
    _pooled_conn_ids.add(id(conn))
    return conn


def close_db_connection(conn):
    if conn is not None:
        pool = _get_pool()
        if pool is not None and id(conn) in _pooled_conn_ids:
            _pooled_conn_ids.discard(id(conn))
            pool.putconn(conn)
            return
        conn.close()


@contextmanager
def db_cursor(commit=False):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        close_db_connection(conn)
