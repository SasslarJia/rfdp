#
import numpy as np
#
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Enum_RFDP.AjcnEnum import AjcnEnum
from RfdpLog.PrntLog_RFDP import prnt_log


class OperAsstNumpy(OperAsst):

    # <editor-fold desc="Method group of 'chg_fmt'">
    def _chg_fmt(self, iput_mat) -> np.ndarray:
        """ Change input matrix into the expected format (Protected Method)
        :param iput_mat: Input matrix
        :return: Output matrix
        2020-10-5
        """
        if type(iput_mat) != np.ndarray:
            return np.array(iput_mat)
        else:
            return iput_mat

    # </editor-fold>

    # <editor-fold desc="Method group of 'expand_strt'">
    def _expand_strt(self, iput_mat: np.ndarray, num: int):
        """ Expand input-matrix with a extra dim0
        :param iput_mat: p×q
        :param num: n
        :return: n×p×q
        2020-12-18
        """
        return np.expand_dims(iput_mat, axis=0).repeat(num, axis=0)
    # </editor-fold>

    # <editor-fold desc="Method group of 'expand_dim'">
    def _expand_dim(self, iput_mat, idx: int):
        """ Expand input-matrix in axis idx
        :param iput_mat: p×q
        :param idx: insert index = 0, 1 or 2
        :return: 1×p×q, p×1×q or p×q×1
        2020-12-18
        """
        return np.expand_dims(iput_mat, axis=idx)
    # </editor-fold>

    # <editor-fold desc="Method group of 'repeat_2d'">
    def _repeat_2d(self, iput_vec: np.ndarray, rept: int, dim: int = 0) -> np.ndarray:
        """
        :param iput_vec:
        :param rept:
        :param dim:
        :return:
        2020-10-6
        """
        return iput_vec.repeat(rept, axis=dim)
        # return np.repeat(iput_vec, rept, axis=dim)

    # </editor-fold>

    # <editor-fold desc="Method group of 'get_ajcn'">
    def _get_ajcn(self, iput_mat: np.ndarray, thld: float, ajcn_type: AjcnEnum) -> np.ndarray:
        """ Get adjacent matrix from input matrix (Protected Method)
        :param iput_mat:
        :return:
        2020-10-5
        """
        #
        if ajcn_type == AjcnEnum.Grt:
            return np.where(iput_mat > thld, 1, 0)
        #
        if ajcn_type == AjcnEnum.GrtEq:
            return np.where(iput_mat >= thld, 1, 0)
        #
        if ajcn_type == AjcnEnum.Sml:
            return np.where(iput_mat < thld, 1, 0)
        #
        if ajcn_type == AjcnEnum.SmlEq:
            return np.where(iput_mat <= thld, 1, 0)
        #
        prnt_log.error('Unknown Input AjcnEnum!')
        #
        return None

    # </editor-fold>

    # <editor-fold desc="Method group of 'pow'">
    def _pow(self, iput_base, iput_exp):
        """ Calculate iput_base^iput_exp (Protected Method)
        :param iput_base:
        :param iput_exp:
        :return:
        2020-10-5
        """
        return np.power(iput_base, iput_exp)

    # </editor-fold>

    # <editor-fold desc="Method group of 'exp'">
    def _exp(self, iput):
        """ Calculate exp(iput) (Protected Method)
        :param iput:
        :return:
        2020-10-5
        """
        return np.exp(iput)

    # </editor-fold>

    # <editor-fold desc="Method group of 'sum2col'">
    def _sum(self, iput_mat, neg_dim, keepdim: bool):
        """
        :param iput_mat:
        :return:
        2020-10-26
        """
        return np.sum(iput_mat, axis=len(iput_mat.shape) + neg_dim, keepdims=keepdim)

    # </editor-fold>

    # <editor-fold desc="Method group of 'ones'">
    def _ones(self, iput):
        return np.ones(iput)
    # </editor-fold>

    # <editor-fold desc="Method group of 'zeros'">
    def _zeros(self, iput):
        return np.zeros(iput)
    # </editor-fold>

    # <editor-fold desc="Method group of 'eye'">
    def _eye(self, n):
        return np.eye(n)
    # </editor-fold>

    # <editor-fold desc="Method group of 'mul_diag'">
    def _mul_diag(self, fir_iput, sec_iput):
        #
        return fir_iput * sec_iput

    # </editor-fold>

    # <editor-fold desc="Method group of 'cvrt2float'">
    def _cvrt2float(self, iput):
        """
        :param iput:
        :return:
        2020-10-5
        """
        return iput.astype(float)

    # </editor-fold>

    # <editor-fold desc="Method group of 'cvrt2int'">
    def _cvrt2int(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        return iput.astype(int)

    # </editor-fold>

    # <editor-fold desc="Method group of 'cvrt2flt'">
    def _cvrt2flt(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        return iput.astype(float)

    # </editor-fold>

    # <editor-fold desc="Method group of 'copy_mat'">
    def _copy_mat(self, iput: np.ndarray):
        return np.array(iput, copy=True)

    # </editor-fold>

    # <editor-fold desc="Method group of 'topk_2d'">
    def _topk_2d(self, iput_mat: np.ndarray, k_nn: int, dim: int = 0, rvrs: bool = False) \
            -> (np.ndarray, np.ndarray):
        """ Get kth
        :param iput_mat:
        :param k_nn:
        :param dim:
        :param rvrs:
        :return:
        """
        #
        if rvrs:
            tmp_mat = -iput_mat
        else:
            tmp_mat = iput_mat
        #
        kth_idx = k_nn - 1
        #
        arg_part = np.argpartition(tmp_mat, kth_idx, axis=dim)
        #
        if dim == 0:
            col_ids = np.arange(tmp_mat.shape[1 - dim])
            tmp_vals = tmp_mat[arg_part[:k_nn, :], col_ids]
            #
            row_sorts = np.argsort(tmp_vals, axis=dim)
            #
            if rvrs:
                ret_vals = -tmp_vals[row_sorts, col_ids]
            else:
                ret_vals = tmp_vals[row_sorts, col_ids]
            #
            ret_ids = arg_part[:k_nn, :][row_sorts, col_ids]
        else:
            row_ids = np.arange(tmp_mat.shape[1 - dim])[:, None]
            tmp_vals = tmp_mat[row_ids, arg_part[:, :k_nn]]
            #
            col_sorts = np.argsort(tmp_vals, axis=dim)
            #
            if rvrs:
                ret_vals = -tmp_vals[row_ids, col_sorts]
            else:
                ret_vals = tmp_vals[row_ids, col_sorts]
            #
            ret_ids = arg_part[:, :k_nn][row_ids, col_sorts]
        #
        return ret_vals, ret_ids

    # </editor-fold>

    # <editor-fold desc="Method group of 'min_1d'">
    def _minmax_1d(self, iput_vector, max_flg: bool):
        #
        if not max_flg:
            idx = np.argmin(iput_vector)
        else:
            idx = np.argmax(iput_vector)
        #
        val = iput_vector[idx]
        #
        return val, idx

    # </editor-fold>

    # <editor-fold desc="Method group of 'mul_3d'">
    def _mul_3d(self, iput1, iput2) -> np.ndarray:
        return np.matmul(iput1, iput2)

    # </editor-fold>

    # <editor-fold desc="Method group of 'reshape'">
    def _reshape(self, iput: np.ndarray, upd_shap):
        """
        Reshape input matrix by given shape
        :param iput: Iputmatrix
        :param upd_shap:
        :return:
        """
        return np.reshape(iput, upd_shap)

    # </editor-fold>

    # <editor-fold desc="Convert to list">
    def _cvrt2lst(self, iput) -> list:
        return iput.tolist()

    # </editor-fold>

    # <editor-fold desc="Transpose of input-matrix">
    def _xpos(self, iput: np.ndarray, tpl):
        return iput.transpose(tpl)

    # </editor-fold>

    # <editor-fold desc="Method group of 'repl'">
    def _repl(self, cond, rslt_pos, rslt_neg):
        """
        :param cond:
        :param rslt_pos:
        :param rslt_neg:
        :return:
        2020-12-10
        """
        return np.where(cond, rslt_pos, rslt_neg)
    # </editor-fold>
