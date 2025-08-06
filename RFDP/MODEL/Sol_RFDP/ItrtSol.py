#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Sol_RFDP.SolProc import SolProc
from MODEL.Asst_RFDP.OperAsst import OperAsst
#
from RfdpLog.PrntLog_RFDP import prnt_log


class ItrtSol(SolProc, metaclass=ABCMeta):
    """
    Abstract class for iteration solution
    2020-12-11
    """

    def __init__(self):
        """
        2020-10-27
        """
        super(ItrtSol, self).__init__()

    @ABCFctn
    def _get_itrtnum(self):
        """
        Get the number of iteration
        :return:
        2020-12-26
        """
        pass

    def _clac_rslt(self, d_cols, w_mats, coef_cols, cnst_cols, fctr_parm: float,
                   is_sim: bool, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2020-12-11
        """
        # Get the inverse of Lambda matrices
        inv_lambdas = ItrtSol.__get_inv_lambdas(fctr_parm, coef_cols, d_cols, asst_rela)
        # Get the matrix in the power series
        pwr_mul = ItrtSol.__get_pwr(inv_lambdas, w_mats)
        # Get post multiply column-vector for one query, or matrix for multi-queries
        pst_mul = ItrtSol._get_post(inv_lambdas, cnst_cols)
        # Get iteration result
        return self._itrt_proc(pwr_mul, pst_mul, asst_rela)

    # <editor-fold desc="The solving process">
    @staticmethod
    def __get_inv_lambdas(fctr_parm, coef_cols, d_cols, asst_rela: OperAsst):
        """ Calculate the inverse of one or more Lambda matrices, where Lambda = alpha·Coef + D
        'coef' is a column-vector (diagonal matrix) or a matrix (each column corresponds to a diagonal matrix)
        :param fctr_parm: Factor parameter of fitting constraint
        :param coef_cols: Coefficient matrix of X
        :param d_cols: Diagonal matrix D in Laplacian matrix
        :param asst_rela: Omit
        :return: Inverse matrix of Lambda and the output is 𝓡^(n×m),
                 where inv(Lambda_k) = diag(_inv_lambda[:,k])
        2020-10-30
        """
        # There may be one 'coef' or different 'coef's
        sum_cols = fctr_parm * coef_cols + d_cols
        #
        return 1 / asst_rela.overturn(sum_cols)

    @staticmethod
    def __get_pwr(inv_lambdas, w_mats):
        """ Calculate power-matrix [inv(Lam)·W]_n×n or power-matrices [inv(Lam_1)·W, ..., inv(Lam_m)·W]_n×n×m
        :param inv_lambdas: Inverse matrix of Lambda matrix
        :param w_mats:
        :return: Broadcast format output m×n×n, where m≥1
        2020-10-31
        """
        # Return power-matrix
        return inv_lambdas * w_mats

    @staticmethod
    def _get_post(inv_lambdas, cnst_cols):
        """ Calculate column-vectors in right hand side
        inv(La)·y, [inv(La)·y1, ..., inv(La)·ym]', or [inv(La_1)·y1, ..., inv(La_m)·ym]
        :param inv_lambdas: Inverse matrix of Lambda matrix
        :param cnst_cols: Constants in right hand side expressed as column-vectors
        :return: Matrix in right hand side m×n×1, where m≥1
        2020-10-30
        """
        #
        return inv_lambdas * cnst_cols
    # </editor-fold>

    # <editor-fold desc="">
    @ABCFctn
    def _itrt_proc(self, pwr_mul, post_mul, asst_rela: OperAsst):
        """
        Main part of iteration process
        :param pwr_mul: matrix in matrix series
        :param post_mul: Column-vector in right hand side
        :param asst_rela: Omit
        :return: Learned similarity/distance Result-matrix, and each column corresponds to one query
        2020-12-27
        """
        pass
    # </editor-fold>
