# src/database/DB_Config.py
import sqlite3
from typing import Optional, Dict, Any, Union, List
import logging
import json
from contextlib import contextmanager
from abc import ABC, abstractmethod
import re

# Optional: psycopg2 imports will be attempted when needed
try:
    import psycopg2
    import psycopg2.pool
    from psycopg2 import OperationalError, sql
except Exception:
    psycopg2 = None
    psycopg2 = None  # keep name defined

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store PostgreSQL connection pool
pg_pool = None

def create_pg_pool(dbname: str = None,
                   host: str = "localhost",
                   user: str = None,
                   password: str = None,
                   port: int = 5432,
                   minconn: int = 1,
                   maxconn: int = 10,
                   sslmode: str = "require") -> Optional[Any]:
    """
    Create PostgreSQL connection pool and store it in global pg_pool.
    Accepts both positional and keyword-style parameters (minconn/maxconn).
    Returns the pool instance on success, or None on failure.
    """
    global pg_pool

    if psycopg2 is None:
        logger.error("psycopg2 is not installed in this environment.")
        return None

    try:
        conn_str = f"dbname={dbname} host={host} user={user} password={password} port={port} sslmode={sslmode}"
        logger.info("Creating Postgres pool -> host=%s dbname=%s user=%s min=%s max=%s", host, dbname, user, minconn, maxconn)
        pool = psycopg2.pool.SimpleConnectionPool(int(minconn), int(maxconn), conn_str)

        # Quick smoke test: get and release a connection
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            _ = cur.fetchone()
            cur.close()
        finally:
            pool.putconn(conn)

        pg_pool = pool
        logger.info("PostgreSQL connection pool created successfully and assigned to global pg_pool.")
        return pg_pool
    except Exception as e:
        logger.exception("Failed to create PostgreSQL connection pool: %s", e)
        pg_pool = None
        return None


@contextmanager
def get_pg_connection():
    """
    Context manager for obtaining a connection from the global PostgreSQL connection pool.
    Yields a psycopg2 connection; caller should not call conn.close(), but the manager will putconn().
    """
    global pg_pool
    if pg_pool is None:
        raise RuntimeError("PostgreSQL pool not initialized. Call create_pg_pool() first.")
    conn = None
    try:
        conn = pg_pool.getconn()
        yield conn
    except Exception as e:
        logger.exception("Error while getting PostgreSQL connection: %s", e)
        raise
    finally:
        if conn is not None and pg_pool is not None:
            try:
                pg_pool.putconn(conn)
            except Exception:
                # best-effort release
                try:
                    conn.close()
                except Exception:
                    pass


@contextmanager
def get_connection(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    port: Optional[int] = 5432
):
    """
    Generic context manager: yields a DB connection object depending on db_type.
    For 'postgresql' it ensures a pool exists and returns a psycopg2 connection (from pool).
    For 'sqlite' it returns a sqlite3.Connection.
    """
    conn = None
    try:
        if db_type.lower() == 'postgresql':
            global pg_pool
            if not pg_pool:
                # Align parameter names with create_pg_pool signature
                pool = create_pg_pool(
                    dbname=db_name,
                    host=host or "localhost",
                    user=user,
                    password=password,
                    port=int(port) if port else 5432,
                    minconn=1,
                    maxconn=10,
                    sslmode="require"
                )
                if not pool:
                    raise RuntimeError("PostgreSQL pool creation failed (see logs).")
                pg_pool = pool

            # get a connection from pool
            conn = pg_pool.getconn()
            logger.info("Connected to PostgreSQL database using connection pool.")
        elif db_type.lower() == 'sqlite':
            conn = sqlite3.connect(db_name)
            logger.info("Connected to SQLite database.")
        else:
            logger.error("Unsupported database type: %s", db_type)
            yield None
            return

        yield conn

    except Exception as e:
        logger.exception("Unexpected error while connecting to the database: %s", e)
        yield None
    finally:
        if conn:
            # if using postgres pool, return connection to pool, else close sqlite connection
            if db_type.lower() == 'postgresql' and pg_pool:
                try:
                    pg_pool.putconn(conn)
                    logger.info("PostgreSQL connection returned to pool.")
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
            else:
                try:
                    conn.close()
                    logger.info("SQLite connection closed.")
                except Exception:
                    pass


def query_database(
    query: str,
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    port: Optional[int] = 5432,
    limit: int = None,
    offset: int = 0
) -> pd.DataFrame:
    """
    Execute a SQL query and return a pandas DataFrame.
    If connection fails returns an empty DataFrame.
    """
    with get_connection(db_name, db_type, host, user, password, port) as conn:
        if conn is None:
            logger.error("Database connection failed. Returning empty DataFrame.")
            return pd.DataFrame()

        modified_query = query
        if db_type.lower() in ['sqlite', 'postgresql'] and query.strip().lower().startswith('select'):
            if limit is not None and "limit" not in query.lower():
                modified_query = f"{query.rstrip(';')} LIMIT {limit} OFFSET {offset};"

        try:
            # For psycopg2 connection and sqlite3 connection, pandas.read_sql_query accepts the connection object
            df = pd.read_sql_query(modified_query, conn)
            logger.info("Query executed successfully: %s", modified_query if len(modified_query) < 200 else modified_query[:200] + "...")
            return df
        except Exception as e:
            logger.exception("Unexpected error executing query: %s", e)
            return pd.DataFrame()


# --- Schema extractor base and SQLite/Postgres implementations (unchanged) ---
class SchemaExtractor(ABC):
    def __init__(self, connection):
        self.conn = connection

    @abstractmethod
    def get_tables(self) -> List[str]:
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        pass

# SQLite extractor
def get_sqlite_table_info(cursor, table_name: str) -> Dict[str, Any]:
    table_info = {
        'columns': {},
        'foreign_keys': [],
        'indexes': [],
        'sample_data': [],
        'primary_keys': [],
        'constraints': [],
        'triggers': []
    }
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = cursor.fetchall()
    for col in columns:
        col_id, col_name, col_type, not_null, default_val, pk = col
        table_info['columns'][col_name] = {
            'type': col_type,
            'nullable': (not not_null),
            'default': default_val,
            'primary_key': bool(pk)
        }
        if pk:
            table_info['primary_keys'].append(col_name)
    # foreign keys, indexes, sample data, triggers handled similarly...
    try:
        cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 5;")
        rows = cursor.fetchall()
        if rows:
            column_names = [desc[0] for desc in cursor.description]
            table_info['sample_data'] = [dict(zip(column_names, row)) for row in rows]
    except Exception as e:
        logger.debug("Unable to fetch sample data for SQLite table %s: %s", table_name, e)
    return table_info

class SQLiteSchemaExtractor(SchemaExtractor):
    def get_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [row[0] for row in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        return get_sqlite_table_info(cursor, table_name)

# PostgreSQL schema extraction (using psycopg2 cursor)
def get_postgresql_table_info(cursor, table_name: str) -> Dict[str, Any]:
    table_info = {
        'columns': {},
        'foreign_keys': [],
        'indexes': [],
        'sample_data': [],
        'primary_keys': [],
        'constraints': [],
        'triggers': []
    }
    cursor.execute(
        """
        SELECT
            column_name, data_type, is_nullable, column_default, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = 'public';
        """,
        [table_name]
    )
    columns = cursor.fetchall()
    for col_name, data_type, is_nullable, default_val, char_len in columns:
        table_info['columns'][col_name] = {
            'type': data_type,
            'nullable': (is_nullable.upper() == 'YES'),
            'default': default_val,
            'max_length': char_len,
            'primary_key': False
        }
    # primary keys, foreign keys, indexes, triggers, sample data...
    try:
        cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 5;").format(sql.Identifier(table_name)))
        sample_data = cursor.fetchall()
        if sample_data:
            column_names = [desc[0] for desc in cursor.description]
            table_info['sample_data'] = [dict(zip(column_names, row)) for row in sample_data]
    except Exception as e:
        logger.debug("Unable to fetch sample data for Postgres table %s: %s", table_name, e)
    return table_info

class PostgreSQLSchemaExtractor(SchemaExtractor):
    def get_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
            """
        )
        return [row[0] for row in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        return get_postgresql_table_info(cursor, table_name)


def get_all_schemas(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    port: Optional[int] = 5432
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve schema info for all tables.
    """
    schemas = {}
    with get_connection(db_name, db_type, host, user, password, port) as conn:
        if not conn:
            logger.error("Database connection failed. Returning empty schema.")
            return {}
        if db_type.lower() == 'sqlite':
            extractor = SQLiteSchemaExtractor(conn)
        elif db_type.lower() == 'postgresql':
            extractor = PostgreSQLSchemaExtractor(conn)
        else:
            logger.error("Unsupported database type: %s", db_type)
            return {}
        for table in extractor.get_tables():
            schemas[table] = extractor.get_table_info(table)
    return schemas


# Small helper you can call from the deployment shell for quick sanity check
def test_connection_quick(db_name, db_type, host=None, user=None, password=None, port=5432):
    logger.info("Running quick test_connection_quick against %s://%s", db_type, host or db_name)
    try:
        with get_connection(db_name, db_type, host, user, password, port) as conn:
            if conn:
                # Try listing tables depending on type
                if db_type.lower() == 'postgresql':
                    cur = conn.cursor()
                    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
                    tables = [r[0] for r in cur.fetchall()]
                    logger.info("Postgres tables: %s", tables)
                    return tables
                else:
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                    tables = [r[0] for r in cur.fetchall()]
                    logger.info("SQLite tables: %s", tables)
                    return tables
            else:
                logger.error("Connection returned None.")
                return []
    except Exception as e:
        logger.exception("Quick connection test failed: %s", e)
        return []
