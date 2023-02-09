"""全局的单输出文件日志管理器.

Todo:
    - [ ] (Jiang Yuzhen) 目前这个模块的功能只是对log的输出格式和文件处理
        进行了整合，并且和qiankunding的log模块有冲突，未来可以使用统一的
        方案来代替。
"""
import logging


class LoggerManager:
    """全局的单输出文件日志管理器.

    Example:

        设置全局logger及其输出日志文件路径
        >>> LoggerManager.set_logger("test", "./tmp.log")

        在任意模块中可以通过logger名称获取logger
        >>> logger = LoggerManager.get_logger("test")
        >>> logger.info("hello world!")
    """

    @staticmethod
    def set_logger(logger_name: str, log_file_path: str):
        """设置logger与对应的输出文件.

        Args:
            logger_name (str): 为logger指定唯一名称以全局访问
            log_file_path (str): 输出log文件路径
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s - %(filename)s[line:%(lineno)d] ' +
                '- %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s - %(filename)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    @staticmethod
    def get_logger(logger_name: str) -> logging.Logger:
        """全局获取logger.

        Args:
            logger_name (str): 获取的logger名

        Returns:
            logging.Logger: 目标logger对象
        """
        return logging.getLogger(logger_name)


if __name__ == "__main__":
    LoggerManager.set_logger("test", "./tmp.log")

    # 在任意模块中
    logger = LoggerManager.get_logger("test")
    logger.info("hello world!")
