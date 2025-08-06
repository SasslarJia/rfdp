#
from typing import Tuple
from abc import ABCMeta
from abc import abstractmethod as ABCFctn

from MODEL.Asst_RFDP.ProcAsst import ProcAsst
from MODEL.Enum_RFDP.DimEnum import DimEnum
from MODEL.Aff_RFDP.CnctCpnt import CnctCpnt
from MODEL.Iput_RFDP.Qry2Blk import Qry2Blk
from MODEL.Asst_RFDP.OperAsst import OperAsst
from RfdpLog.PrntLog_RFDP import prnt_log


class MdlIput(metaclass=ABCMeta):
    """
    Abstract class for operating the relative inputs of Rfdp
    2020-11-9
    """

    def __init__(self, smth_w, fit_w, full_w, asst_rela: OperAsst):
        """
        :param smth_w: Matrix for smoothness constraint, smth_w∈𝓡^(n×n)
        :param fit_w: Matrix for fitting constraint, fit_w∈𝓡^(n×n)
        :param full_w: Affinity matrix, full_w∈𝓡^(n×n)
        :param asst_rela: Omit
        2020-11-9
        """
        if fit_w is None:
            if not MdlIput.__chk2dim(smth_w):
                prnt_log("Input matrix 'smth_w' is not 2D")
            self._fit_w = smth_w
        else:
            # Check two input matrices
            if not MdlIput.__chk2mat(smth_w, fit_w, asst_rela):
                prnt_log("The shapes of 'smth_w' and 'fit_w' are not same")
            self._fit_w = fit_w
        # Initialize parameters
        self._smth_w, self._full_w, self._asst_rela = smth_w, full_w, asst_rela
        self._smpl_num, self._dim = self._smth_w.shape[0], DimEnum.Dim2
        # Get the list with edges
        self._edges_lst = MdlIput.__get_edges(smth_w, True, asst_rela)
        # With smth_w, label each element with connected component index
        self._cnct_ids = CnctCpnt(self._asst_rela)(self._edges_lst, self._smpl_num)
        # Initialize outputs
        self._qry_lst, self._q2b_lst = list(), list()

    def __call__(self, iput_qries: list, incl_qry: bool) -> list:
        """
        :param iput_qries: The input list with queries
        :param incl_qry: Whether self-similarity/self-distance is learned
        :return: q2b_lst: the list of Qry2Blk
        2020-11-9
        """
        # Cleanse input queries
        if not ProcAsst.spcf_qry(iput_qries, self._smpl_num):
            prnt_log("There is an error in input queries!")
        # Initialize parameters
        self._incl_qry = incl_qry
        # Get Qry2Blk list and elements indexes for each block
        q2b_lst = self._stl_qries(iput_qries)
        #
        return q2b_lst

    # <editor-fold desc="Check the queries, and construct the list of block elements">
    @ABCFctn
    def _stl_qries(self, qry_lst: list) -> list:
        """
        Get the the list of Qry2Blk
        :param qry_lst:
        :return: List of Qry2Blk
        2020-10-9
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Get indexes of elements in the set of connected components">
    @ABCFctn
    def get_eles(self, rela_ids: list) -> list:
        """
        Get element indexes according to the block indexes
        :param rela_ids: Block indexes
        :return: List of element indexes
        2020-10-9
        """
        pass

    # </editor-fold>

    @staticmethod
    def _exst_lst(q2b_lst: list, iput_lst: list) -> Qry2Blk:
        """
        :param q2b_lst: List with lists
        :param iput_lst: Input list to be checked
        :return:
        """
        #
        for q2b in q2b_lst:
            if q2b.rela_ids == iput_lst:
                return q2b
        #
        return None

    # <editor-fold desc="Get the list with edges">
    @staticmethod
    def __get_edges(iput_mat, incl_self: bool, asst_rela: OperAsst) -> list:
        """ Construct graph based on the input matrix
        :param iput_mat: Input weighted matrix
        :param incl_self:
        :return: List of edges in graph
        """
        # Get adjacent matrix from input matrix
        cnct_mat = asst_rela.get_ajcn(iput_mat)
        # Get max nearest neighborhoods of weighted matrix
        max_nn = int(cnct_mat.sum(0).max())
        # Get neighbors of each element, filling with zero elements in the end
        _, top_ids = asst_rela.topk_2d(cnct_mat, max_nn, dim=1, rvrs=True)
        # Get list for tops without index itself
        edges_lst = list()
        #
        for idx in range(top_ids.shape[0]):
            tmp_flg = cnct_mat[idx, top_ids[idx, :]] > 0
            tmp_lst = list(set(asst_rela.cvrt2lst(top_ids[idx, tmp_flg])))
            #
            if not incl_self:
                tmp_lst.remove(idx)
            #
            edges_lst.append(tmp_lst)
        #
        return edges_lst

    # </editor-fold>

    # <editor-fold desc="Check input matrix">
    @staticmethod
    def __chk2mat(fir_mat, sec_mat, asst_rela: OperAsst) -> bool:
        """
        Check whether the shapes of given matrices are the same
        :param fir_mat: First matrix
        :param sec_mat: Second matrix
        :param asst_rela: Omit
        :return:
        """
        # Check whether fir_mat is a 2D matrix
        if not MdlIput.__chk2dim(fir_mat):
            return False
        # Check whether the shapes of wgt_mat and full_mat are the same
        if not asst_rela.chksmsq(fir_mat, sec_mat):
            prnt_log("Input matrices are not the same!")
            return False
        #
        return True

    @staticmethod
    def __chk2dim(iput_mat) -> bool:
        # Check whether wgt_mat is a 2D matrix
        if not DimEnum.Dim2.chk_dim(iput_mat):
            prnt_log("The input 'wgt_mat' is not a 2D matrix!")
            return False
        #
        return True

    # </editor-fold>

    # <editor-fold desc="Properties">
    @property
    def smth_w(self):
        return self._smth_w

    @property
    def fit_w(self):
        return self._fit_w

    @property
    def asst_rela(self):
        return self._asst_rela

    @property
    def smpl_num(self):
        return self._smpl_num

    @property
    def dim(self):
        return self._dim

    @property
    def ful_w(self):
        return self._full_w

    @property
    def incl_qry(self):
        return self._incl_qry

    # @property
    # def qry_lst(self):
    #     return self._qry_lst
    #
    # @property
    # def q2b_lst(self):
    #     return self._q2b_lst
    # </editor-fold>
