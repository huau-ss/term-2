import pytest
from app.NLP2SQL import validate_sql_query


def test_sql_injection_prevention():
    """测试SQL注入防护"""
    malicious_queries = [
        "SELECT * FROM users; DROP TABLE users",
        "SELECT * FROM users WHERE name = 'test'; INSERT INTO users VALUES ('hacker', 'password')",
        "SELECT * FROM users UNION SELECT * FROM passwords",
        "'; EXEC sp_msforeachtable 'DROP TABLE ? --"
    ]

    for query in malicious_queries:
        assert validate_sql_query(query) == False


def test_input_validation():
    """测试输入验证"""
    from app.NLP2SQL import validate_query_tables

    test_schemas = {
        "valid_table": {"columns": ["col1"], "types": ["TEXT"]}
    }

    # 测试有效查询
    valid_query = "SELECT * FROM valid_table"
    assert validate_query_tables(valid_query, test_schemas) == True

    # 测试无效查询（引用不存在的表）
    invalid_query = "SELECT * FROM non_existent_table"
    assert validate_query_tables(invalid_query, test_schemas) == False