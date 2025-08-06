#
from RfdpLog.PrntLog_RFDP import prnt_log
from MODEL.Iput_RFDP.MdlIput import MdlIput


class RefRslt(object):
    """
    2020-12-23
    """

    def __init__(self, rslt_ids: list):
        """
        :param rslt_ids:
        2020-12-23
        """
        self.__rslt_ids = rslt_ids

    def chk(self, iput_idx: int) -> bool:
        """
        :param iput_idx:
        :return:
        """
        return iput_idx in self.__rslt_ids

    @property
    def rslt_ids(self) -> list:
        return self.__rslt_ids

    @staticmethod
    def easy_cstr(qry_lst: list, cls_num: int, ele_num: int) -> list:
        #
        ref_rslts = list()
        #
        for cls_idx in range(cls_num):
            ref_rslts.append(RefRslt([idx for idx in range(cls_idx * ele_num, (cls_idx + 1) * ele_num)]))
        #
        oput_lst = list()
        #
        for idx in range(len(qry_lst)):
            #
            if len(qry_lst[idx]) > 1:
                prnt_log('The number of elements in the ' + str(idx) + 'th Qry is greater than 1!')
            #
            oput_lst.append(ref_rslts[qry_lst[idx][0] // ele_num])
        #
        return oput_lst



