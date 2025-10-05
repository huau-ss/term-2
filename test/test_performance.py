import pytest
import time
from app.NLP2SQL import generate_sql_query


def test_query_generation_performance():
    """测试查询生成性能"""
    test_schemas = {
        "test_table": {
            "columns": ["col1", "col2", "col3"],
            "types": ["TEXT", "INTEGER", "REAL"]
        }
    }

    start_time = time.time()
    response = generate_sql_query("显示所有数据", test_schemas)
    end_time = time.time()

    execution_time = end_time - start_time
    # 查询生成应该在合理时间内完成
    assert execution_time < 30.0  # 30秒超时


def test_memory_usage():
    """测试内存使用"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 执行一些操作
    test_schemas = {
        "large_table": {
            "columns": [f"col_{i}" for i in range(50)],
            "types": ["TEXT"] * 50
        }
    }

    for i in range(10):
        generate_sql_query(f"测试查询 {i}", test_schemas)

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory

    # 内存增长应该在合理范围内
    assert memory_increase < 500  # MB