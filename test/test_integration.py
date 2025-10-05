import pytest
import streamlit as st
from app.NLP2SQL import generate_sql_query, handle_query_response
import json


class TestIntegration:
    """集成测试类"""

    def test_nl_to_sql_workflow(self):
        """测试自然语言到SQL的完整工作流"""
        # 模拟数据库schema
        test_schemas = {
            "users": {
                "columns": ["id", "name", "email", "age"],
                "types": ["INTEGER", "TEXT", "TEXT", "INTEGER"]
            },
            "orders": {
                "columns": ["order_id", "user_id", "amount", "date"],
                "types": ["INTEGER", "INTEGER", "REAL", "TEXT"]
            }
        }

        # 测试查询生成
        user_message = "显示所有用户的姓名和年龄"
        response = generate_sql_query(user_message, test_schemas)

        assert 'query' in response or 'error' in response
        assert 'decision_log' in response

        # 验证决策日志结构
        if 'decision_log' in response:
            decision_log = response['decision_log']
            required_keys = [
                'query_input_details', 'preprocessing_steps',
                'path_identification', 'final_summary'
            ]

            for key in required_keys:
                assert key in decision_log

    def test_markdown_generation(self):
        """测试Markdown决策日志生成"""
        sample_decision_log = {
            "query_input_details": ["用户查询: 显示用户数据"],
            "preprocessing_steps": ["步骤1: 解析查询", "步骤2: 识别实体"],
            "path_identification": [
                {
                    "description": "路径1描述",
                    "tables": ["users"],
                    "columns": [["name", "age"]],
                    "score": 85
                }
            ],
            "final_summary": "测试总结"
        }

        markdown = build_markdown_decision_log(sample_decision_log)
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "### Query Input Analysis" in markdown