import logging
import sys
from typing import Optional
# from pathlib import Path

class MyLogger:
    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MyLogger, cls).__new__(cls)
        return cls._instance

    def setup_logger(self, 
                    name: str, 
                    log_level: str = 'INFO',
                    log_file: Optional[str] = None) -> logging.Logger:
        """设置日志记录器"""
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 如果指定了日志文件，添加文件处理器
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self._loggers[name] = logger
        return logger

def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    return MyLogger().setup_logger(name)