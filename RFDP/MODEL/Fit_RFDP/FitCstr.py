#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Iput_RFDP.MdlIput import MdlIput
from MODEL.Asst_RFDP.OperAsst import OperAsst


class FitCstr(metaclass=ABCMeta):
    """ (Abstract Class) Fitting Constraint
    Given F = f(x-y) e.g.‖x-y‖^2, calculate ∂F/∂x, where x is a column-vector
    and y is a column-vector predefined values
    2020-2-16
    """

    def __init__(self, fctr_parm: float):
        """ Initialize fitting constraint with the normalization flag
        :param fctr_parm: Weight factor as the hyper-parameter of the fitting constraint
        2020-10-26
        """
        #
        self._fctr_parm = fctr_parm
        self._is_sim = self._check_sim()

    def __call__(self, mdl_iput: MdlIput, ids_lst: list, qries_insub: list):
        """
        :param mdl_iput: Inputs of the model
        :param ids_lst: m elements with indexes from sub-matrix
        :param qries_insub: k queries with indexes from sub-matrix
        :return: [The same as that of '__get_xfrm']
        2020-10-28
        """
        # Get the transformed input matrix
        subw_xfrm = self.__get_xfrm(mdl_iput.fit_w, ids_lst, mdl_iput.asst_rela)
        # Get the coefficient matrix diag(coef), and the constant as a column vector cnst
        coef_cols, cnst_cols = self._get_fit(subw_xfrm, qries_insub, mdl_iput.incl_qry, mdl_iput.asst_rela)
        #
        return coef_cols, cnst_cols

    # <editor-fold desc="Operation of f(W)">
    def __get_xfrm(self, w_init, ids_lst: list, asst_rela: OperAsst):
        """
        Get the output f(W)
        :param w_init: w_init∈𝓡^(n×n)
        :param ids_lst: m elements with indexes from sub-matrix
        :param asst_rela: Omit
        :return: [The same as that of '_get_fit']
        2020-10-28
        """
        # Get f(W)
        return self._xfrm_iput(w_init[ids_lst, :][:, ids_lst], asst_rela)

    @ABCFctn
    def _xfrm_iput(self, sub_w, asst_rela: OperAsst):
        """
        Get the output-matrix transformed from the input
        :param sub_w: W = [wij]_m×m
        :param asst_rela: Omit
        :return: f(W)∈𝓡^(m×m)
        2020-10-27
        """
        pass
    # </editor-fold>

    @ABCFctn
    def _get_fit(self, sub_w, qries_insub: list, incl_qry: bool, asst_rela: OperAsst):
        """ (Abstract Method) Get factor matrix and constant matrix
        :param sub_w: W = [wij]_m×m
        :param qries_insub: k queries with indexes from sub-matrix
        :param incl_qry: Whether self-similarity/self-distance is learned
        :param asst_rela: Omit
        :return: (1)coef_cols: coef_cols∈𝓡^(k×m×1)
                 (2)cnst_cols: cnst_cols∈𝓡^(k×m×1)
        """
        pass

    # <editor-fold desc="Check whether the learned result is similarity">
    @ABCFctn
    def _check_sim(self) -> bool:
        """
        Whether the learned result is similarity
        :return: Similarity: True; Dissimilarity: False
        2020-12-20
        """
        pass
    # </editor-fold>

    # <editor-fold desc="Properties">
    @property
    def fctr_parm(self) -> float:
        return self._fctr_parm

    @property
    def is_sim(self) -> bool:
        return self._is_sim
    # </editor-fold>

