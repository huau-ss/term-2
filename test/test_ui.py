import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app():
    """测试Streamlit应用界面"""
    at = AppTest.from_file("app/NLP2SQL.py")
    at.run(timeout=30)

    # 检查基本组件是否存在
    assert len(at.sidebar) > 0
    assert "Select Database Type" in at.sidebar[0].label

    # 检查主题应用
    assert "custom_css" in at.markdown[0].body


def test_database_selection():
    """测试数据库选择功能"""
    at = AppTest.from_file("app/NLP2SQL.py")
    at.run(timeout=30)

    # 模拟选择SQLite
    at.sidebar.selectbox[0].select("SQLite")
    at.run(timeout=10)

    # 检查文件上传器是否存在
    assert "Upload SQLite Database" in at.sidebar[0].label