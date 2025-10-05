import pytest
import pandas as pd
import numpy as np
from app.NLP2SQL import (
    create_chart,
    analyze_dataframe_for_visualization,
    analyze_query_performance,
    build_markdown_decision_log
)


def test_chart_creation():
    """测试图表生成"""
    # 创建测试数据
    df = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'values': [10, 20, 15, 25]
    })

    # 测试各种图表类型
    chart_types = ['Bar Chart', 'Line Chart', 'Scatter Plot']

    for chart_type in chart_types:
        fig = create_chart(df, chart_type, 'category', 'values')
        assert fig is not None


def test_dataframe_analysis():
    """测试DataFrame分析"""
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'date_col': pd.date_range('2023-01-01', periods=5)
    })

    suggestions = analyze_dataframe_for_visualization(df)
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0


def test_query_performance_analysis():
    """测试查询性能分析"""
    metrics = analyze_query_performance("SELECT * FROM table", 0.5, 1000)

    assert 'execution_time' in metrics
    assert 'rows_returned' in metrics
    assert 'performance_class' in metrics
    assert 'suggestions' in metrics