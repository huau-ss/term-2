import os
import logging
import re
from dotenv import load_dotenv

# 可按需安装：pip install dashscope
# 可按需安装：pip install google-generativeai openai
# import google.generativeai as genai
import openai
import dashscope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 .env
load_dotenv()

# 读取并统一化 Provider
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "GEMINI").upper()
MODEL_NAME = os.getenv("MODEL_NAME")  # 各 Provider 通用的模型名占位
logger.info(f"LLM Provider: {LLM_PROVIDER}")

# 按 Provider 校验所需环境变量
if LLM_PROVIDER == "GEMINI":
    REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]
elif LLM_PROVIDER == "AZURE":
    REQUIRED_ENV_VARS = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
elif LLM_PROVIDER == "QWEN":  # 通义千问（DashScope）
    REQUIRED_ENV_VARS = ["DASHSCOPE_API_KEY"]
else:
    logger.error(f"Unsupported LLM provider: {LLM_PROVIDER}. Supported: GEMINI, AZURE, QWEN")
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}. Supported: GEMINI, AZURE, QWEN")

missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Provider 级别初始化
if LLM_PROVIDER == "GEMINI":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
elif LLM_PROVIDER == "AZURE":
    # 兼容你现有的用法
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_type = "azure"
    # 如需更新，请与 Azure 文档同步
    openai.api_version = "2023-07-01-preview"
elif LLM_PROVIDER == "QWEN":
    # DashScope 在具体调用处设置 api_key
    pass


def _clean_markdown_code_fences(text: str) -> str:
    """去掉返回中常见的 ```json ... ``` 或 ```sql ... ``` 包裹。"""
    if not isinstance(text, str):
        text = str(text)
    # 去掉三引号代码块围栏
    text = re.sub(r"```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```", "", text)
    return text.strip()


# def get_completion_from_gemini(
#     system_message: str,
#     user_message: str,
#     temperature: float = 0.0
# ) -> str:
#     """
#     使用 Google Gemini 生成回复。
#     """
#     try:
#         combined_message = f"{system_message}\n\nUser Query: {user_message}"
#         logger.info("=== INPUT (Gemini) ===")
#         logger.info(f"Temperature: {temperature}")
#         logger.info(f"Combined:\n{combined_message}")
#
#         # 你原本使用的是 'gemini-2.0-flash'，保留默认
#         model_name = MODEL_NAME or "gemini-2.0-flash"
#         model_instance = genai.GenerativeModel(model_name)
#         response = model_instance.generate_content(
#             contents=combined_message,
#             generation_config={"temperature": float(temperature)}
#         )
#
#         logger.info("=== RAW OUTPUT (Gemini) ===")
#         logger.info(f"{response}")
#
#         text = getattr(response, "text", "")
#         return _clean_markdown_code_fences(text)
#     except Exception as e:
#         logger.exception("Error generating response from Gemini")
#         raise


def get_completion_from_azure(
    system_message: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0
) -> str:
    """
    使用 Azure OpenAI 生成回复。
    """
    try:
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        logger.info("=== INPUT (Azure) ===")
        logger.info(f"Temperature: {temperature}, MaxTokens: {max_tokens}, TopP: {top_p}")
        logger.info(f"Deployment: {deployment_name}")
        logger.info(f"System:\n{system_message}")
        logger.info(f"User:\n{user_message}")

        # 你现有代码使用老的 ChatCompletion 接口，这里保持一致
        response = openai.ChatCompletion.create(
            deployment_id=deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p)
        )

        logger.info("=== RAW OUTPUT (Azure) ===")
        logger.info(f"{response}")

        text = response["choices"][0]["message"]["content"]
        return _clean_markdown_code_fences(text)
    except Exception as e:
        logger.exception("Error generating response from Azure OpenAI")
        raise


def get_completion_from_qwen(
    system_message: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 1.0
) -> str:
    """
    使用 阿里通义千问（DashScope） 生成回复。
    需要环境变量：
      - DASHSCOPE_API_KEY
      - MODEL_NAME（可选，默认 qwen-plus）
    """
    try:
        import dashscope
        from dashscope import Generation
    except Exception as e:
        raise ImportError(
            "dashscope is not installed. Please `pip install dashscope`."
        ) from e

    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    model_name = MODEL_NAME or "qwen-plus"

    logger.info("=== INPUT (QWEN) ===")
    logger.info(f"Model: {model_name}, Temperature: {temperature}, MaxTokens: {max_tokens}, TopP: {top_p}")
    logger.info(f"System:\n{system_message}")
    logger.info(f"User:\n{user_message}")

    # 与 OpenAI 风格对齐的 messages 结构
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        # result_format='message' 让返回结构更接近 OpenAI 风格
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

        # 正常情况下：
        # resp.output.choices[0].message = {"role": "assistant", "content": "..."}
        out = getattr(resp, "output", None)
        if out and getattr(out, "choices", None):
            content = out.choices[0].message.get("content", "")
            return _clean_markdown_code_fences(content)

        # 兜底：有些错误信息会直接放在 resp.message 里
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
    根据 .env 里的 LLM_PROVIDER 路由到相应实现。
    返回值为纯文本字符串。
    """
    logger.info(f"Using provider: {LLM_PROVIDER}")

    # 如果配置为 GEMINI 但无法使用，自动降级到 QWEN
    effective_provider = LLM_PROVIDER
    if LLM_PROVIDER == "GEMINI":
        return get_completion_from_gemini(
            system_message=system_message,
            user_message=user_message,
            temperature=temperature
        )

    if LLM_PROVIDER == "AZURE":
        return get_completion_from_azure(
            system_message=system_message,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

    if LLM_PROVIDER == "QWEN":
        # 优先使用 .env 的 MODEL_NAME；如果函数参数传了 model，也可覆盖
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

    # 不应该走到这里（前面已做校验）
    # raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    raise ValueError(f"Unsupported LLM provider: {effective_provider}")