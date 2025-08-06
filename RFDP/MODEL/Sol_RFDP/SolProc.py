#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Asst_RFDP.OperAsst import OperAsst


class SolProc(metaclass=ABCMeta):
    """
    Abstract class for solving the optimization problem of RFDP
    2020-10-9
    """

    def __init__(self):
        """
        2020-10-30
        """

    def __call__(self, qries_lst: list, d_cols, w_mats, coef_cols, cnst_cols,
                 fctr_parm: float, is_sim: bool, asst_rela: OperAsst) -> list:
        """
        :param qries_lst: k queries with indexes from sub-matrix
        :param d_cols: D = diag(d_col), d_col∈𝓡^(n×1)
        :param w_mats: W = w_mat, w_mat∈𝓡^(n×n)
        :param coef_cols: Coefficient matrix diag(coef_cols[:,k]), where coef_cols∈𝓡^(n×m)
        :param cnst_cols: cnst_cols∈𝓡^(n×m)
        :param fctr_parm: Weight factor as the hyper-parameter of the fitting constraint
        :param is_sim: Record the result is similarity or distance
        :param asst_rela:
        :return:
        2020-10-30
        """
        # Learning similarity/distance
        rslts_mat = self._clac_rslt(d_cols, w_mats, coef_cols, cnst_cols,
                                    fctr_parm, is_sim, asst_rela)
        # Get ranking results
        return SolProc.__rslt_rank(qries_lst, rslts_mat, is_sim, asst_rela)

    @ABCFctn
    def _clac_rslt(self, d_cols, w_mats, coef_cols, cnst_cols, fctr_parm: float,
                   is_sim: bool, asst_rela: OperAsst):
        """
        :param d_cols: d_cols∈𝓡^(k×m×1 or 1×m×1)
        :param w_mats: w_mats∈𝓡^(k×m×m or 1×m×m)
        :param coef_cols: coef_cols∈𝓡^(k×m×1)
        :param cnst_cols: cnst_cols∈𝓡^(k×m×1)
        :param fctr_parm:
        :param is_sim:
        :param asst_rela: Omit
        :return: Learned similarity/distance Result-matrix, and each column corresponds to one query
        2020-12-11
        """
        pass

    @staticmethod
    def __rslt_rank(qries_lst: list, rslts_mat, is_sim: bool, asst_rela: OperAsst) -> list:
        """
        :param rslts_mat:
        :param is_sim:
        :return:
        2020-12-16
        """
        #
        qry_num, ele_num = list(rslts_mat.shape)[:-1]
        #
        for idx in range(len(qries_lst)):
            #
            rslts_mat[idx, qries_lst[idx], 0] = float('inf') if is_sim else -float('inf')
        #
        _, sort_ids = asst_rela.topk_2d(asst_rela.reshape(rslts_mat, [qry_num, ele_num]),
                                        ele_num, dim=1, rvrs=is_sim)
        #
        rslt_lst = [asst_rela.cvrt2lst(sort_ids[idx, :]) for idx in range(qry_num)]
        #
        for idx in range(len(rslt_lst)):
            rslt = rslt_lst[idx]
            rslt[:len(qries_lst[idx])] = qries_lst[idx]
        #
        return rslt_lst
