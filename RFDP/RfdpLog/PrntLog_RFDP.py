#
import os
import logging
#
from datetime import datetime
from logging import handlers
from MODEL.Enum_RFDP.LogTypeEnum import LogTypeEnum


class PrntLog(object):
    """ Log for RFDP
    2020-10-7
    """

    def __init__(self, log_type: LogTypeEnum = LogTypeEnum.WARNING):
        """
        :param log_type:
        2020-10-9
        """
        #
        txt_str = "RFDP_log_" + datetime.strftime(datetime.now(), "%Y-%m-%d") + ".txt"
        #
        self.__log = logging.getLogger()
        self.__log.setLevel(log_type.value)
        #
        fmt_str = logging.Formatter('%(asctime)s - %(pathname)s - [line:%(lineno)d] - %(levelname)s > %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt_str)
        #
        file_folder = os.path.abspath(os.path.join(os.path.dirname(__file__))) + os.sep + "logs"
        self.make_dir(file_folder)
        #
        file_str = file_folder + os.sep + txt_str
        #
        th = handlers.TimedRotatingFileHandler(filename=file_str, when='H', encoding='utf-8')
        th.setFormatter(fmt_str)
        self.__log.addHandler(sh)
        self.__log.addHandler(th)

    def make_dir(self, dir_pth):
        """
        :param dir_pth:
        :return:
        2020-10-9
        """
        tmp_pth = dir_pth.strip()
        if not os.path.exists(tmp_pth):
            os.makedirs(tmp_pth)

    def error(self, err_str: str):
        self.__log.error(err_str)
        raise RuntimeError(err_str)

    def critical(self, crit_str: str):
        self.__log.error(crit_str)
        raise RuntimeError(crit_str)


# Singleton
prnt_log = PrntLog()
