from contextlib import contextmanager
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from matrixmatch_app.config import get_db_config

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        minconn = int(os.getenv("DB_POOL_MIN", "1"))
        maxconn = int(os.getenv("DB_POOL_MAX", "10"))
        _pool = ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            cursor_factory=RealDictCursor,
            **get_db_config(),
        )
    return _pool


def get_db_connection():
    """Get one pooled DB connection.

    Callers are responsible for closing via close_db_connection().
    """
    return _get_pool().getconn()


def close_db_connection(conn):
    if conn is not None:
        _get_pool().putconn(conn)


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
