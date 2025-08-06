#
import copy
import operator
from typing import Tuple
from RfdpLog.PrntLog_RFDP import prnt_log


class ProcAsst(object):

    # Check the length of each query
    @staticmethod
    def meas_qry(qry_lst: list, rerank_num: int) -> bool:
        """
        :param qry_lst:
        :param rerank_num:
        :return:
        2021-7-11
        """
        #
        for qry in qry_lst:
            if len(qry) >= rerank_num:
                prnt_log.error("There exists incorrect query!")
                return False
        #
        return True

    # <editor-fold desc="Checking whether there exist repeated or out of bound samples in each query">meas_qry
    @staticmethod
    def spcf_qry(qry_lst: list, smpl_num: int) -> bool:
        """
        Check input queries
        :param qry_lst:
        :param smpl_num:
        :return:
        2020-10-9
        """
        #
        for idx in range(len(qry_lst)):
            #
            tmp_qry = qry_lst[idx]
            # Examine elements in each query
            if not ProcAsst.__rept_otbd(tmp_qry if isinstance(tmp_qry, list) else list([tmp_qry]),
                                        0, smpl_num - 1):
                prnt_log.error("Input queries contains repeated or incorrect elements!")
                return False
            # #
            # for tmp_idx in range(idx):
            #     if operator.eq(tmp_qry, qry_lst[tmp_idx]):
            #         prnt_log.error("There are two identical queries!")
            #         return False
        #
        return True

    @staticmethod
    def __rept_otbd(qry: list, min_val: int, max_val: int) -> bool:
        """
        Check whether there are repeated elements or out of bound elements
        :param qry:
        :param min_val:
        :param max_val:
        :return:
        2020-10-9
        """
        #
        tmp_lst = list(set(map(int, qry)))
        # Check whether there repeated samples
        if len(tmp_lst) != len(qry):
            prnt_log.error("One of the queries contains repeated elements!")
            return False
        # Check the bound of the elements
        if min(tmp_lst) < min_val or max(tmp_lst) > max_val:
            prnt_log.error("One of the queries contains incorrect elements!")
            return False
        #
        return True
    # </editor-fold>

    @staticmethod
    def find_idcl(iput_qries: list, ids_lst: list = None) -> Tuple[list, list]:
        """ Find identical queries
        :param iput_qries: Input list with queries
        :param ids_lst: Index list
        :return: (1) Output list of queies
                 (2) List of positions
        2021-8-23
        """
        #
        qry_len = len(iput_qries)
        #
        qry_lst = list()
        pos_lst = list()
        #
        if ids_lst is None:
            ids_lst = range(qry_len)
        #
        for idx in range(qry_len):
            #
            qry = set(iput_qries[idx])
            #
            idcl_flg = False
            #
            for tmp in range(len(qry_lst)):
                if operator.eq(qry, set(qry_lst[tmp])):
                    idcl_flg = True
                    if isinstance(ids_lst[idx], int):
                        pos_lst[tmp].append(ids_lst[idx])
                    elif isinstance(ids_lst[idx], list):
                        pos_lst[tmp].extend(ids_lst[idx])
                    else:
                        prnt_log.error("Input list is incorrect!")
                    break
            #
            if not idcl_flg:
                qry_lst.append(iput_qries[idx])
                if isinstance(ids_lst[idx], int):
                    pos_lst.append([ids_lst[idx]])
                elif isinstance(ids_lst[idx], list):
                    pos_lst.append(copy.deepcopy(ids_lst[idx]))
                else:
                    prnt_log.error("Input list is incorrect!")
        #
        return qry_lst, pos_lst

    @staticmethod
    def cvrtbk_idcl(rslt_lst: list, init_qries: list, pstn_lst: list) -> list:
        """ Convert back to original list of results, e.g. [list1, list2, ...]
        :param rslt_lst:
        :param init_qries:
        :param pstn_lst:
        :return:
        2021-8-23
        """
        #
        oput_lst = [None] * len(init_qries)
        #
        for idx in range(len(pstn_lst)):
            tmp_rslt = rslt_lst[idx]
            for tmp_pstn in pstn_lst[idx]:
                tmp = copy.deepcopy(init_qries[tmp_pstn])
                tmp.extend(tmp_rslt[len(tmp):])
                oput_lst[tmp_pstn] = tmp
        #
        return oput_lst

    @staticmethod
    def cvrtbk_mult_idcl(mult_rslts: list, init_qries: list, pstn_lst: list) -> list:
        """ Convert back to original list of multiple results, e.g. [[list1A, list1B], [list2A, list2B],  ...]
        :param mult_rslts:
        :param init_qries:
        :param pstn_lst:
        :return:
        2021-8-23
        """
        #
        oput_lst = [None] * len(init_qries)
        #
        for idx in range(len(pstn_lst)):
            tmp_mult = mult_rslts[idx]
            for tmp_pstn in pstn_lst[idx]:
                tmp_lst = list()
                for mult_idx in range(len(tmp_mult)):
                    tmp = copy.deepcopy(init_qries[tmp_pstn])
                    tmp.extend(tmp_mult[mult_idx][len(tmp):])
                    tmp_lst.append(tmp)
                #
                oput_lst[tmp_pstn] = tmp_lst
        #
        return oput_lst

    @staticmethod
    def repl_idcl(rslt: list, oput_lst: list, pstn_lst: list) -> list:
        """ Replace specific result
        :param rslt:
        :param oput_lst:
        :param pstn_lst:
        :return:
        2021-8-26
        """
        #
        for temp_pstn in pstn_lst:
            tmp = copy.deepcopy(oput_lst[temp_pstn])
            tmp.extend(rslt[len(tmp):])
            oput_lst[temp_pstn] = tmp
        #
        return oput_lst
