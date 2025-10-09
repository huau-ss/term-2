import sqlite3
from typing import Optional, Dict, Any, Union, List
import logging
import json
from contextlib import contextmanager
from abc import ABC, abstractmethod
import re

# Optional psycopg2 imports; keep names even if import fails so code referencing them fails gracefully
try:
    import psycopg2
    from psycopg2 import OperationalError, sql
    import psycopg2.pool
except Exception:
    psycopg2 = None
    OperationalError = Exception
    sql = None

import pandas as pd

# Configure logging for improved debug and info messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store PostgreSQL connection pool (one per process)
pg_pool = None


def create_pg_pool(minconn: int,
                   maxconn: int,
                   dbname: str,
                   host: str,
                   user: str,
                   password: str,
                   port: int = 5432,
                   sslmode: str = "require") -> Optional[Any]:
    """
    Create PostgreSQL connection pool with SSL support (Neon requires sslmode=require).
    Stores the created pool into module-level `pg_pool` and returns it.
    Returns None on failure.

    Arguments:
      minconn, maxconn: pool size
      dbname, host, user, password, port, sslmode: connection info
    """
    global pg_pool

    if psycopg2 is None:
        logger.error("psycopg2 not installed; cannot create Postgres pool.")
        return None

    try:
        conn_str = f"dbname={dbname} host={host} user={user} password={password} port={port} sslmode={sslmode}"
        logger.info("Creating Postgres pool -> host=%s dbname=%s user=%s min=%s max=%s", host, dbname, user, minconn, maxconn)

        pool = psycopg2.pool.SimpleConnectionPool(int(minconn), int(maxconn), conn_str)

        # Smoke test: get and release a connection
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            _ = cur.fetchone()
            cur.close()
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

        pg_pool = pool
        logger.info("PostgreSQL connection pool created successfully with SSL and saved to global pg_pool (id=%s).", id(pg_pool))
        return pg_pool
    except Exception as e:
        logger.exception("Failed to create PostgreSQL connection pool: %s", e)
        pg_pool = None
        return None


@contextmanager
def get_pg_connection():
    """
    Context manager for obtaining a connection from the global PostgreSQL connection pool.
    Yields a psycopg2 connection. Caller should not close it directly; the manager will return it to the pool.
    """
    global pg_pool
    if pg_pool is None:
        raise RuntimeError("PostgreSQL pool is not initialized. Call create_pg_pool() or use get_connection() which will attempt to create it.")
    conn = None
    try:
        conn = pg_pool.getconn()
        yield conn
    except Exception as e:
        logger.exception("Error while getting PostgreSQL connection: %s", e)
        raise
    finally:
        if conn and pg_pool:
            try:
                pg_pool.putconn(conn)
            except Exception:
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
    Context manager for creating and managing database connections.

    For postgresql: ensures a pool exists (lazy-create) and yields a connection from it.
    For sqlite: yields a sqlite3.Connection.

    Returns None on failure (caller should check).
    """
    conn = None
    global pg_pool
    try:
        if db_type.lower() == 'postgresql':
            # If pg_pool not initialized, try to create it now
            if not pg_pool:
                logger.info("pg_pool is None — attempting to create a new pool from get_connection().")
                created = create_pg_pool(
                    minconn=1,
                    maxconn=10,
                    dbname=db_name,
                    host=host or "localhost",
                    user=user or "",
                    password=password or "",
                    port=int(port) if port else 5432,
                    sslmode="require"
                )
                if not created:
                    logger.error("create_pg_pool() returned None. PostgreSQL pool creation failed.")
                    raise RuntimeError("PostgreSQL pool creation failed (see logs).")

            if pg_pool is None:
                logger.error("pg_pool still None after attempted creation.")
                raise RuntimeError("pg_pool is None after create attempt.")

            logger.info("Using pg_pool (id=%s) to obtain connection.", id(pg_pool))
            conn = pg_pool.getconn()
            logger.info("Obtained connection from pg_pool.")
        elif db_type.lower() == 'sqlite':
            conn = sqlite3.connect(db_name)
            logger.info("Connected to SQLite database.")
        else:
            logger.error(f"Unsupported database type: {db_type}")
            yield None
            return

        yield conn

    except OperationalError as e:
        logger.exception("Operational error while connecting to the database: %s", e)
        yield None
    except Exception as e:
        logger.exception("Unexpected error while connecting to the database: %s", e)
        yield None
    finally:
        if conn:
            try:
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
            except Exception:
                logger.debug("Exception while releasing/closing connection.", exc_info=True)


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
    Executes an SQL query on the specified database and returns the results as a Pandas DataFrame.
    If a limit is provided and the query is a SELECT without a LIMIT clause,
    automatically appends the LIMIT and OFFSET.
    """
    with get_connection(db_name, db_type, host, user, password, port) as conn:
        if conn is None:
            logger.error("Database connection failed. Returning empty DataFrame.")
            return pd.DataFrame()

        modified_query = query
        if db_type.lower() in ['sqlite', 'postgresql'] and query.strip().lower().startswith('select'):
            if limit is not None and "limit" not in query.lower():
                modified_query = f"{query.rstrip(';')} LIMIT {limit} OFFSET {offset};"
                logger.warning("Query truncated with LIMIT for performance. Use pagination for full results.")
        try:
            df = pd.read_sql_query(modified_query, conn)
            logger.info("Query executed successfully.")
            return df
        except Exception as e:
            logger.exception(f"Unexpected error executing query: {e}")
            return pd.DataFrame()


# --- Abstract Schema Extractor Class ---
class SchemaExtractor(ABC):
    def __init__(self, connection):
        self.conn = connection

    @abstractmethod
    def get_tables(self) -> List[str]:
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        pass


# --- SQLite Schema Extraction ---
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

    cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
    fkeys = cursor.fetchall()
    for fk in fkeys:
        _, _, ref_table, from_col, to_col, on_update, on_delete, _ = fk
        table_info['foreign_keys'].append({
            'from_column': from_col,
            'to_table': ref_table,
            'to_column': to_col,
            'on_update': on_update,
            'on_delete': on_delete
        })

    cursor.execute(f"PRAGMA index_list('{table_name}');")
    indexes = cursor.fetchall()
    for idx in indexes:
        idx_id, idx_name, unique_flag = idx[0], idx[1], idx[2]
        cursor.execute(f"PRAGMA index_info('{idx_name}');")
        index_columns = cursor.fetchall()
        table_info['indexes'].append({
            'name': idx_name,
            'unique': bool(unique_flag),
            'columns': [col[2] for col in index_columns]
        })

    try:
        cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 5;")
        rows = cursor.fetchall()
        if rows:
            column_names = [desc[0] for desc in cursor.description]
            table_info['sample_data'] = [dict(zip(column_names, row)) for row in rows]
    except Exception as e:
        logger.warning(f"Unable to retrieve sample data for table {table_name}: {e}")

    cursor.execute(f"SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name='{table_name}';")
    trigger_rows = cursor.fetchall()
    for tr in trigger_rows:
        tr_name, tr_sql = tr
        table_info['triggers'].append({
            'name': tr_name,
            'definition': tr_sql
        })

    return table_info


class SQLiteSchemaExtractor(SchemaExtractor):
    def get_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [row[0] for row in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        return get_sqlite_table_info(cursor, table_name)


# --- PostgreSQL Schema Extraction ---
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
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
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

    cursor.execute(
        """
        SELECT kcu.column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = %s
          AND tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = 'public';
        """,
        [table_name]
    )
    pk_columns = [row[0] for row in cursor.fetchall()]
    for pk_col in pk_columns:
        if pk_col in table_info['columns']:
            table_info['columns'][pk_col]['primary_key'] = True
            table_info['primary_keys'].append(pk_col)

    cursor.execute(
        """
        SELECT
            kcu.column_name AS from_column,
            ccu.table_name AS to_table,
            ccu.column_name AS to_column,
            rc.update_rule AS on_update,
            rc.delete_rule AS on_delete
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON tc.constraint_name = ccu.constraint_name
        JOIN information_schema.referential_constraints AS rc
          ON tc.constraint_name = rc.constraint_name
        WHERE tc.table_name = %s
          AND tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public';
        """,
        [table_name]
    )
    fkeys = cursor.fetchall()
    for from_col, to_table, to_col, on_update, on_delete in fkeys:
        table_info['foreign_keys'].append({
            'from_column': from_col,
            'to_table': to_table,
            'to_column': to_col,
            'on_update': on_update,
            'on_delete': on_delete
        })

    cursor.execute(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s;
        """,
        [table_name]
    )
    indexes = cursor.fetchall()
    for idx_name, idx_def in indexes:
        idx_columns = []
        try:
            start = idx_def.index('(')
            end = idx_def.rindex(')')
            cols_part = idx_def[start + 1:end]
            idx_columns = [c.strip() for c in cols_part.split(',')]
        except ValueError:
            pass
        is_unique = 'UNIQUE' in idx_def.upper()
        table_info['indexes'].append({
            'name': idx_name,
            'unique': is_unique,
            'columns': idx_columns
        })

    cursor.execute(
        """
        SELECT tgname, pg_get_triggerdef(t.oid)
        FROM pg_trigger t
        JOIN pg_class c ON t.tgrelid = c.oid
        WHERE c.relname = %s
          AND NOT t.tgisinternal;
        """,
        [table_name]
    )
    triggers = cursor.fetchall()
    for tr_name, tr_def in triggers:
        table_info['triggers'].append({
            'name': tr_name,
            'definition': tr_def
        })

    try:
        cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 5;").format(sql.Identifier(table_name)))
        sample_data = cursor.fetchall()
        if sample_data:
            column_names = [desc[0] for desc in cursor.description]
            table_info['sample_data'] = [dict(zip(column_names, row)) for row in sample_data]
    except Exception as e:
        logger.warning(f"Unable to retrieve sample data for table {table_name}: {e}")

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


# --- Utility for generating a structured JSON string (optional) ---
def generate_json_schema(table_name: str, table_info: Dict[str, Any]) -> str:
    schema = {
        "object": table_name,
        "columns": table_info.get('columns', {}),
        "primary_keys": table_info.get('primary_keys', []),
        "foreign_keys": table_info.get('foreign_keys', []),
        "indexes": table_info.get('indexes', []),
        "triggers": table_info.get('triggers', []),
        "constraints": table_info.get('constraints', []),
        "sample_data": table_info.get('sample_data', [])
    }
    return json.dumps(schema, indent=2)


def get_all_schemas(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    schemas = {}

    with get_connection(db_name, db_type, host, user, password) as conn:
        if not conn:
            logger.error("Database connection failed. Returning empty schema.")
            return {}

        if db_type.lower() == 'sqlite':
            extractor = SQLiteSchemaExtractor(conn)
        elif db_type.lower() == 'postgresql':
            extractor = PostgreSQLSchemaExtractor(conn)
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return {}

        for table in extractor.get_tables():
            schemas[table] = extractor.get_table_info(table)

    return schemas


# --- Quick utility to test connection and list tables ---
def test_connection_quick(db_name, db_type, host=None, user=None, password=None, port=5432):
    """
    Quick check: attempt to connect and list tables. Returns list of table names or [].
    Useful to run in deployment shell or Streamlit button to verify connectivity.
    """
    logger.info("Running quick connection test for %s@%s (db=%s)", user, host, db_name)
    try:
        tables = []
        with get_connection(db_name, db_type, host, user, password, port) as conn:
            if conn is None:
                logger.error("Quick test: connection returned None.")
                return []
            if db_type.lower() == 'postgresql':
                cur = conn.cursor()
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
                tables = [r[0] for r in cur.fetchall()]
                cur.close()
                logger.info("Quick test found Postgres tables: %s", tables)
            else:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                tables = [r[0] for r in cur.fetchall()]
                cur.close()
                logger.info("Quick test found SQLite tables: %s", tables)
        return tables
    except Exception as e:
        logger.exception("Quick connection test failed: %s", e)
        return []
