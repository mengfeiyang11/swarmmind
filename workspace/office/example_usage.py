"""
日志配置使用示例
"""

import logging
from logger import setup_logging, get_logger


# 初始化日志系统
setup_logging()

# 获取日志器
logger = get_logger(__name__)


def main():
    """示例函数，展示不同级别的日志使用"""
    logger.debug("这是一条调试信息")
    logger.info("程序开始运行")
    logger.warning("这是一条警告信息")
    
    try:
        # 模拟一个错误
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error(f"发生除零错误: {e}", exc_info=True)
    
    logger.info("程序运行结束")


if __name__ == "__main__":
    main()
