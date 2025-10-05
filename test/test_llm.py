import pytest
from src.api.LLM_Config import get_completion_from_messages
from src.prompts.Base_Prompt import SYSTEM_MESSAGE


def test_llm_config():
    """测试LLM配置"""
    # 测试系统消息加载
    assert SYSTEM_MESSAGE is not None
    assert len(SYSTEM_MESSAGE) > 0

    # 测试消息格式
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]

    # 注意：在实际测试中可能需要mock API调用
    # response = get_completion_from_messages(test_messages)
    # assert response is not None