#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : humeng
# @Time    : 2020/12/29

import logging
from datetime import datetime
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class Logger(object):

    def __init__(self, file_name, log_level, logger_name, log_to_file=True):
        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)
        # self.base_dir = '/export/logs'
        self.base_dir = os.path.join(ROOT_DIR, 'logs')

        # 指定日志的最低输出级别，默认为WARN级别
        self.__logger.setLevel(log_level)
        # 设置格式
        date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        formatter = logging.Formatter(
            '[%(asctime)s] - [logger name :%(name)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')

        # 创建一个handler用于写入日志文件
        if log_to_file:
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
        log_path = os.path.join(self.base_dir, file_name + '_' + date + '.log')
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        # 给logger添加handler
        self.__logger.addHandler(file_handler)

        # log写入控制台
        stream_handler = logging.StreamHandler()  # 往屏幕上输出
        stream_handler.setFormatter(formatter)  # 设置屏幕上显示的格式
        # 给logger添加handler
        self.__logger.addHandler(stream_handler)

    def get_log(self):
        return self.__logger


if __name__ == '__main__':
    logger = Logger(log_level=logging.INFO, file_name='Logger', logger_name="test").get_log()
    logger.debug('testing ... ')
    logger.info('testing ... ')
