"""
日志配置 - 使用 Python 标准 logging 替代 loguru
"""
import logging
import sys

def setup_logging(level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

# 默认启用 logging
setup_logging()
