"""
日志工具模块
"""

import logging
import logging.config
from pathlib import Path

from config import LOGGING_CONFIG, LOG_DIR


def setup_logging():
    """
    配置应用程序日志系统
    """
    # 确保日志目录存在
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # 应用日志配置
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # 获取根日志器
    logger = logging.getLogger()
    logger.info("日志系统初始化完成")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称，通常使用模块名 __name__
    
    Returns:
        配置好的 Logger 实例
    """
    if name:
        return logging.getLogger(name)
    return logging.getLogger()
