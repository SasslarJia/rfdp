#
from RfdpLog.PrntLog_RFDP import prnt_log


class Qry2Blk(object):
    """
    A class with the indexes of queries, and the indexes of related subsets,
    e.g. connected components and edges in list
    2020-10-23
    """

    def __init__(self, rela_ids: list, qry_idx: int):
        """
        :param rela_ids:
        :param qry_idx:
        2020-12-8
        """
        self._rela_ids = rela_ids
        self._qids_lst = list([qry_idx])

    def append(self, qry_idx: int):
        if qry_idx in self._qids_lst:
            prnt_log("Query already exists!")
        else:
            self._qids_lst.append(qry_idx)

    def exst_qry(self, qry_idx: int) -> bool:
        return qry_idx in self._qids_lst

    # <editor-fold desc="Properties">
    @property
    def rela_ids(self) -> list:
        return self._rela_ids

    @property
    def qids_lst(self) -> list:
        return self._qids_lst
    # </editor-fold>
