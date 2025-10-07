import os
import logging
import re
from dotenv import load_dotenv

# 只导入必要的库
import dashscope
from dashscope import Generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 .env
load_dotenv()

# 强制使用 QWEN
LLM_PROVIDER = "QWEN"
MODEL_NAME = os.getenv("MODEL_NAME") or "qwen-plus"

# 检查必需的环境变量
REQUIRED_ENV_VARS = ["DASHSCOPE_API_KEY"]
missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]

if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# 配置 DashScope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def _clean_markdown_code_fences(text: str) -> str:
    """去掉返回中常见的 ```json ... ``` 或 ```sql ... ``` 包裹。"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```", "", text)
    return text.strip()

def get_completion_from_qwen(
    system_message: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 1.0
) -> str:
    """
    使用 阿里通义千问（DashScope） 生成回复。
    """
    model_name = MODEL_NAME

    logger.info("=== INPUT (QWEN) ===")
    logger.info(f"Model: {model_name}, Temperature: {temperature}, MaxTokens: {max_tokens}, TopP: {top_p}")
    logger.info(f"System:\n{system_message}")
    logger.info(f"User:\n{user_message}")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        resp = Generation.call(
            model=model_name,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens else None,
            top_p=float(top_p),
            result_format='message',
            stream=False
        )

        logger.info("=== RAW OUTPUT (QWEN) ===")
        logger.info(f"{resp}")

        out = getattr(resp, "output", None)
        if out and getattr(out, "choices", None):
            content = out.choices[0].message.get("content", "")
            return _clean_markdown_code_fences(content)

        if hasattr(resp, "message"):
            logger.error(f"[QWEN] Error message: {resp.message}")
        raise RuntimeError(f"QWEN response unexpected: {resp}")
    except Exception as e:
        logger.exception("Error generating response from QWEN (DashScope)")
        raise

def get_completion_from_messages(
    system_message: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    n: int = 1,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0
) -> str:
    """
    使用 QWEN 生成回复。
    """
    logger.info(f"Using provider: {LLM_PROVIDER}")
    
    global MODEL_NAME
    if model:
        MODEL_NAME = model
        
    return get_completion_from_qwen(
        system_message=system_message,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
