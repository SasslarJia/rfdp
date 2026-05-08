#
import numpy as np
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
        MdlIput.__warn_dense_graph(smth_w, asst_rela)
        # Defer explicit edge-list materialization until a subclass really needs it.
        self._edges_lst = None
        # Connected components are computed directly from the adjacency matrix.
        self._cnct_ids = MdlIput.__get_cnct_ids(smth_w, asst_rela)
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

    def _ensure_edges(self):
        if self._edges_lst is None:
            self._edges_lst = MdlIput.__get_edges(self._smth_w, True, self._asst_rela)

    # <editor-fold desc="Get the list with edges">
    @staticmethod
    def __get_edges(iput_mat, incl_self: bool, asst_rela: OperAsst) -> list:
        """ Construct graph based on the input matrix
        :param iput_mat: Input weighted matrix
        :param incl_self:
        :return: List of edges in graph
        """
        cnct_mat = MdlIput.__to_numpy(asst_rela.get_ajcn(iput_mat)) > 0
        row_ids, col_ids = np.nonzero(cnct_mat)
        if not incl_self:
            keep_flg = row_ids != col_ids
            row_ids = row_ids[keep_flg]
            col_ids = col_ids[keep_flg]
        edges_lst = list()
        for _ in range(cnct_mat.shape[0]):
            edges_lst.append(list())
        if len(row_ids) == 0:
            return edges_lst
        sort_ids = np.argsort(row_ids, kind='mergesort')
        row_ids = row_ids[sort_ids]
        col_ids = col_ids[sort_ids]
        unq_rows, strt_ids = np.unique(row_ids, return_index=True)
        for idx in range(len(unq_rows)):
            strt = strt_ids[idx]
            end = strt_ids[idx + 1] if idx + 1 < len(strt_ids) else len(row_ids)
            edges_lst[int(unq_rows[idx])] = col_ids[strt:end].tolist()
        return edges_lst

    @staticmethod
    def __get_cnct_ids(iput_mat, asst_rela: OperAsst) -> list:
        cnct_mat = MdlIput.__to_numpy(asst_rela.get_ajcn(iput_mat)) > 0
        if cnct_mat.shape[0] == 0:
            return list()
        np.fill_diagonal(cnct_mat, True)
        deg_vec = np.asarray(cnct_mat.sum(axis=1)).reshape(-1)
        if int(deg_vec.min()) == cnct_mat.shape[0]:
            return [0] * cnct_mat.shape[0]
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            _, cc_ids = connected_components(csr_matrix(cnct_mat), directed=False,
                                             return_labels=True)
            return cc_ids.tolist()
        except ModuleNotFoundError:
            edges_lst = MdlIput.__get_edges(cnct_mat, True, asst_rela)
            return CnctCpnt(asst_rela)(edges_lst, cnct_mat.shape[0])

    @staticmethod
    def __to_numpy(iput_mat):
        if isinstance(iput_mat, np.ndarray):
            return iput_mat
        if hasattr(iput_mat, 'detach'):
            return iput_mat.detach().cpu().numpy()
        if hasattr(iput_mat, 'cpu') and hasattr(iput_mat, 'numpy'):
            return iput_mat.cpu().numpy()
        return np.asarray(iput_mat)

    @staticmethod
    def __warn_dense_graph(iput_mat, asst_rela: OperAsst):
        smth_np = MdlIput.__to_numpy(iput_mat)
        if smth_np.ndim != 2 or smth_np.shape[0] == 0:
            return
        cnct_mat = MdlIput.__to_numpy(asst_rela.get_ajcn(iput_mat)) > 0
        np.fill_diagonal(cnct_mat, False)
        edge_ratio = float(cnct_mat.mean())
        if smth_np.shape[0] >= 4096 and edge_ratio > 0.05:
            prnt_log._PrntLog__log.warning(
                "Dense smoothness graph detected (edge ratio %.4f). "
                "This usually means a distance matrix was passed as 'smth_w'. "
                "Build a sparse KNN affinity graph first." % edge_ratio
            )

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
