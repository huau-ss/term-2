import pytest
import pandas as pd
import tempfile
import os


@pytest.fixture
def sample_dataframe():
    """提供测试用的DataFrame"""
    return pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': [i % 50 + 18 for i in range(100)],
        'salary': [i * 1000 for i in range(100)],
        'department': ['IT' if i % 2 == 0 else 'HR' for i in range(100)]
    })


@pytest.fixture
def temp_database():
    """创建临时测试数据库"""
    import sqlite3

    # 创建临时文件
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()

    # 创建测试表和数据
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE users
                   (
                       id    INTEGER PRIMARY KEY,
                       name  TEXT NOT NULL,
                       email TEXT,
                       age   INTEGER
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE orders
                   (
                       order_id   INTEGER PRIMARY KEY,
                       user_id    INTEGER,
                       amount     REAL,
                       order_date TEXT
                   )
                   ''')

    # 插入测试数据
    for i in range(10):
        cursor.execute(
            "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
            (f'User_{i}', f'user_{i}@test.com', 20 + i)
        )

    conn.commit()
    conn.close()

    yield temp_db.name

    # 清理
    os.unlink(temp_db.name)