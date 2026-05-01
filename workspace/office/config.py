# Python 项目配置文件
"""
项目配置和日志设置
"""

import os
from datetime import datetime

# 项目基础配置
PROJECT_NAME = "SwarmMind"
VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "simple": {
            "format": "%(levelname)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "error_file": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/error.log",
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": True
        },
        "swarmmind": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# 日志目录配置
LOG_DIR = "logs"
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 应用配置
APP_CONFIG = {
    "host": os.getenv("APP_HOST", "127.0.0.1"),
    "port": int(os.getenv("APP_PORT", 8000)),
    "workers": int(os.getenv("APP_WORKERS", 1)),
    "timeout": int(os.getenv("APP_TIMEOUT", 30)),
    "reload": DEBUG
}
