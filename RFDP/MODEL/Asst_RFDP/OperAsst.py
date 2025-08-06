#
import sys
#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Enum_RFDP.AjcnEnum import AjcnEnum
from MODEL.Enum_RFDP.DimEnum import DimEnum
from RfdpLog.PrntLog_RFDP import prnt_log

MAX_INT = sys.maxsize


class OperAsst(metaclass=ABCMeta):
    """
    Related to Matrix
    2020-2-18
    """

    # Witout Test
    # <editor-fold desc="Method group of '_chk_sq'">
    def _chk_sq(self, iput, dim: DimEnum = None) -> bool:
        """ Check whether input matrix is square
        :param iput: Input Matrix
        :param dim:
        :return: Return 'True' if input is n×n or m×n×n
        2020-10-5
        """
        #
        tmp_dim = DimEnum.get_dim(iput)
        #
        if dim is not None and dim != tmp_dim:
            return False
        #
        if tmp_dim == DimEnum.Dim2 and iput.shape[0] == iput.shape[1]:
            return True
        elif tmp_dim == DimEnum.Dim3 and iput.shape[1] == iput.shape[2]:
            return True
        else:
            return False

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'get_ele_num'">
    def get_ele_num(self, iput, dim: DimEnum = None) -> int:
        """ Get the dimension of input square matrix
        :param iput: Input Matrix, n×n nor m×n×n
        :param dim:
        :return: Return n if input is n×n or m×n×n
        2020-10-5
        """
        if self._chk_sq(iput, dim):
            return iput.shape[-1]
        else:
            prnt_log.error('The shape of input is not correct!')

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'chksmsq'">
    def chksmsq(self, iput_mat1, iput_mat2) -> bool:
        """
        :param iput_mat1: First input matrix
        :param iput_mat2: Second input matrix
        :return:
        """
        return self._chk_sq(iput_mat1) and iput_mat1.shape == iput_mat2.shape

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'chg_fmt'">
    def chg_fmt(self, iput_mat):
        """ Change input matrix into the expected format
        :param iput_mat:
        :return:
        2020-10-5
        """
        return self._chg_fmt(iput_mat)

    @ABCFctn
    def _chg_fmt(self, iput_mat):
        """ Change input matrix into the expected format (Protected Method)
        :param iput_mat:
        :return:
        2020-10-5
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Method group of 'expand_strt'">
    def expand_strt(self, iput_mat, num: int):
        """ Expand input-matrix with a extra dim0 at the start
        :param iput_mat: p×q
        :param num: n
        :return: n×p×q
        2020-12-18
        """
        return self._expand_strt(iput_mat, num)

    @ABCFctn
    def _expand_strt(self, iput_mat, num: int):
        """ Expand input-matrix with a extra dim0
        :param iput_mat: p×q
        :param num: n
        :return: n×p×q
        2020-12-18
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Method group of 'expand_dim'">
    def expand_dim(self, iput_mat, idx: int):
        """ Expand input-matrix in axis idx
        :param iput_mat: p×q
        :param idx: insert index = 0, 1 or 2
        :return: 1×p×q, p×1×q or p×q×1
        2020-12-18
        """
        if 0 <= idx <= 2:
            return self._expand_dim(iput_mat, idx)
        else:
            prnt_log.error('The shape of input is not correct!')

    @ABCFctn
    def _expand_dim(self, iput_mat, idx: int):
        """ Expand input-matrix in axis idx
        :param iput_mat: p×q
        :param idx: insert index = 0, 1 or 2
        :return: 1×p×q, p×1×q or p×q×1
        2020-12-18
        """
        pass
    # </editor-fold>

    # Tested
    # <editor-fold desc="Method group of 'repeat_2d'">
    def repeat_2d(self, iput_vec, rept: int):
        """
        :param iput_vec:
        :param rept:
        :param dim:
        :return:
        2020-10-5
        """
        if len(iput_vec.shape) != 2 or (iput_vec.shape[0] != 1 and iput_vec.shape[1] != 1):
            prnt_log.error('Input is not 2D vector!')
        #
        if iput_vec.shape[0] == 1:
            return self._repeat_2d(iput_vec, rept, 0)
        else:
            return self._repeat_2d(iput_vec, rept, 1)

    @ABCFctn
    def _repeat_2d(self, iput_vec, rept: int, dim: int = 0):
        """
        :param iput_vec:
        :param rept:
        :param dim:
        :return:
        2020-10-5
        """
        pass

    # </editor-fold>

    # Tested
    # <editor-fold desc="Method group of 'get_ajcn'">
    def get_ajcn(self, iput_mat, thld: float = 0, ajcn_type: AjcnEnum = AjcnEnum.Grt):
        """ Get adjacent matrix from input matrix
        :param iput_mat:
        :param thld:
        :param ajcn_type:
        :return: adjacent matrix
        2020-10-5
        """
        return self._get_ajcn(iput_mat, thld, ajcn_type)

    @ABCFctn
    def _get_ajcn(self, iput_mat, thld: float, ajcn_type: AjcnEnum):
        """ Get adjacent matrix from input matrix (Protected Method)
        :param iput_mat:
        :return:
        2020-10-5
        """
        pass

    # </editor-fold>

    # Tested
    # <editor-fold desc="Method group of 'pow'">
    def pow(self, iput_base, iput_exp):
        """ Calculate iput_base^iput_exp
        :param iput_base:
        :param iput_exp:
        :return:
        2020-10-5
        """
        return self._pow(iput_base, iput_exp)

    @ABCFctn
    def _pow(self, iput_base, iput_exp):
        """ Calculate iput_base^iput_exp (Protected Method)
        :param iput_base:
        :param iput_exp:
        :return:
        2020-10-5
        """
        pass

    # </editor-fold>

    # Tested
    # <editor-fold desc="Method group of 'exp'">
    def exp(self, iput):
        """ Calculate exp(iput)
        :param iput:
        :return:
        2020-10-5
        """
        return self._exp(iput)

    @ABCFctn
    def _exp(self, iput):
        """ Calculate exp(iput) (Protected Method)
        :param iput:
        :return:
        2020-10-5
        """
        pass

    # </editor-fold>

    # Tested
    # <editor-fold desc="Method group of 'sum2col'">
    def sum2last(self, iput_mat, keepdim: bool = True):
        """
        :param iput_mat:
        :param keepdim:
        :return:
        2020-10-26
        """
        return self._sum(iput_mat, -1, keepdim)

    def sum2lbo(self, iput_mat, keepdim: bool = True):
        """
        Last before one
        :param iput_mat:
        :param keepdim:
        :return:
        2020-10-26
        """
        return self._sum(iput_mat, -2, keepdim)

    @ABCFctn
    def _sum(self, iput_mat, neg_dim, keepdim: bool):
        """
        :param iput_mat:
        :param neg_dim: -1 = (1 or 2) for column, and -2 = (0 or 1) for row
        :param keepdim:
        :return:
        2020-12-2
        """
        pass

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'ones'">
    def ones(self, iput):
        return self._ones(iput)

    @ABCFctn
    def _ones(self, iput):
        pass
    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'zeros'">
    def zeros(self, iput):
        return self._zeros(iput)

    @ABCFctn
    def _zeros(self, iput):
        pass
    # </editor-fold>

    # <editor-fold desc="Method group of 'eye'">
    def eye(self, n: int):
        """
        Construct n×n identity matrix
        :param n:
        :return:
        """
        #
        return self._eye(n)

    def _eye(self, n: int):
        pass
    # </editor-fold>

    # Tested
    # <editor-fold desc="Method group of 'mul_diag'">
    def mul_diag(self, iput1, iput2):
        """ Multiple a 2D matrix with a diagonal matrix (expressed as a vector)
        :param iput1: Column vector (diag(vec)) or matrix (mat)
        :param iput2: Matrix (mat) or a row vector (diag(vec))
        :return: diag(vec)*mat or mat*diag(vec)
        """
        return self._mul_diag(iput1, iput2)

    @ABCFctn
    def _mul_diag(self, fir_iput, sec_iput):
        pass

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'cvrt2float'">
    def cvrt2float(self, iput):
        """
        :param iput:
        :return:
        2020-10-5
        """
        return self._cvrt2float(iput)

    @ABCFctn
    def _cvrt2float(self, iput):
        """
        :param iput:
        :return:
        2020-10-5
        """
        pass

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'cvrt2int'">
    def cvrt2int(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        return self._cvrt2int(iput)

    @ABCFctn
    def _cvrt2int(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Method group of 'cvrt2flt'">
    def cvrt2flt(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        return self._cvrt2flt(iput)

    @ABCFctn
    def _cvrt2flt(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        pass

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'copy_mat'">
    def copy_mat(self, iput):
        """
        :param iput:
        :return:
        2020-11-9
        """
        return self._copy_mat(iput)

    @ABCFctn
    def _copy_mat(self, iput):
        """
        :param iput:
        :return:
        2020-11-9
        """
        pass

    # </editor-fold>

    # Tested
    # <editor-fold desc="Method group of 'topk_2d'">
    def topk_2d(self, iput_mat, k_nn: int, dim: int = 0, rvrs: bool = False):
        """ Get top-K values and element indexes from input matrix
        :param iput_mat:
        :param k_nn:
        :param dim:
        :param rvrs: False->Ascending, True->Descending
        :return:
        2020-10-5
        """
        #
        if iput_mat.shape[dim] < k_nn:
            prnt_log.error('Concerned dimension is grater than that of input!')
        #
        if k_nn <= 0:
            prnt_log.error('Number of kth is not correct!')
        #
        return self._topk_2d(iput_mat, k_nn, dim, rvrs)

    @ABCFctn
    def _topk_2d(self, iput_mat, k_nn: int, dim: int, rvrs: bool):
        """ Get top-K values and element indexes from input matrix
        :param iput_mat:
        :param k_nn:
        :param dim:
        :param rvrs:
        :return:
        2020-10-5
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Method group of 'minmax_1d'">
    def minmax_1d(self, iput_vector, max_flg: bool = False):
        """
        :param iput_vector:
        :param max_flg:
        :return:
        2020-11-10
        """
        #
        if len(iput_vector.shape) > 1:
            prnt_log.error('Input is not 1D vector!')
        else:
            return self._minmax_1d(iput_vector, max_flg)

    def _minmax_1d(self, iput_vector, max_flg: bool):
        """
        :param iput_vector:
        :return:
        2020-11-10
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Method group of 'mul_3d'">
    def mul_3d(self, iput1, iput2):
        """
        :param iput1:
        :param iput2:
        :return:
        2020-11-15
        """
        return self._mul_3d(iput1, iput2)

    @ABCFctn
    def _mul_3d(self, iput1, iput2):
        """
        :param iput1:
        :param iput2:
        :return:
        2020-11-15
        """
        pass

    # </editor-fold>

    # Witout Test
    # <editor-fold desc="Method group of 'reshape'">
    def reshape(self, iput, upd_shap):
        """
        Reshape input matrix by given shape
        :param iput: Iputmatrix
        :param upd_shap:
        :return:
        """
        return self._reshape(iput, upd_shap)

    @ABCFctn
    def _reshape(self, iput, upd_shap):
        """
        Reshape input matrix by given shape
        :param iput: Iputmatrix
        :param upd_shap:
        :return:
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Convert to list">
    def cvrt2lst(self, iput) -> list:
        """
        :param iput:
        :return:
        2020-12-8
        """
        #
        if not DimEnum.Dim1.chk_dim(iput):
            prnt_log("Input is not 1D vector!")
            return None
        else:
            return self._cvrt2lst(iput)

    @ABCFctn
    def _cvrt2lst(self, iput) -> list:
        """
        :param iput:
        :return:
        2020-12-8
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Transpose of input-matrix">
    def xpos(self, iput, pos1: int, pos2: int):
        """
        :param iput:
        :param pos1:
        :param pos2:
        :return:
        2020-12-10
        """
        tmp = [idx for idx in range(len(iput.shape))]
        tmp[pos1], tmp[pos2] = tmp[pos2], tmp[pos1]
        return self._xpos(iput, tuple(tmp))

    @ABCFctn
    def _xpos(self, iput, tpl: tuple):
        """
        :param iput:
        :return:
        2020-12-10
        """
        pass

    # </editor-fold>

    # <editor-fold desc="Method group of 'repl'">
    def repl(self, cond, rslt_pos, rslt_neg):
        """ Replace elements
        :param cond:
        :param rslt_pos:
        :param rslt_neg:
        :return:
        2020-12-10
        """
        return self._repl(cond, self.chg_fmt(rslt_pos), self.chg_fmt(rslt_neg))

    @ABCFctn
    def _repl(self, cond, rslt_pos, rslt_neg):
        """
        :param cond:
        :param rslt_pos:
        :param rslt_neg:
        :return:
        2020-12-10
        """
        pass

    # </editor-fold>

    # <editor-fold desc="0->infinity">
    def overturn(self, iput):
        """
        :param iput:
        :return:
        2020-12-21
        """
        return self.repl(iput == 0, float('inf'), iput)

    # </editor-fold>

    def all(self, iput):
        """
        :param iput:
        :return:
        2020-12-21
        """
        return iput.all()

