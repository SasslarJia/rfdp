#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Iput_RFDP.MdlIput import MdlIput
from MODEL.Asst_RFDP.OperAsst import OperAsst


class SmthCstr(metaclass=ABCMeta):
    """ (Abstract Class) Smoothness Constraint (SC)
    Given S = x‘·L·x, calculate ∂S/∂x = L·x, where x is a column-vector
    2020-2-16
    """

    def __init__(self, is_norm: bool):
        """
        Smoothness constraint
        :param is_norm: Whether W is normalized
        2020-10-26
        """
        self._is_norm = is_norm

    def __call__(self, mdl_iput: MdlIput, ids_lst: list, qries_insub: list):
        """
        :param mdl_iput: Input of the model
        :param ids_lst: m elements with indexes from sub-matrix
        :param qries_insub: k queries with indexes from sub-matrix
        :return: [The same as that of '__get_lapl']
        2020-10-28
        """
        # Get the transformed input matrix
        subw_xfrm = self.__get_xfrm(mdl_iput.smth_w, ids_lst, mdl_iput.asst_rela)
        # Get Laplacian matrix
        lapl_mats = self._get_lapl(subw_xfrm, qries_insub, mdl_iput.incl_qry, mdl_iput.asst_rela)
        # Remove selves from Laplacian matrix, L = lapl_mats[k,:,:]
        lapl_mats = self.__lapl_post(lapl_mats, qries_insub, mdl_iput.incl_qry)
        # Normalize Laplacian Matrix, D = diag(d_col[k, :, :])
        if self._is_norm:
            return self.__norm_lapl(lapl_mats, qries_insub, mdl_iput.incl_qry, mdl_iput.asst_rela)
        else:
            return self._lapl_diag(lapl_mats, mdl_iput.asst_rela), lapl_mats

    # <editor-fold desc="Operation of f(W)">
    def __get_xfrm(self, w_init, ids_lst: list, asst_rela: OperAsst):
        """
        Get the output f(W)
        :param w_init: w_init∈𝓡^(n×n)
        :param ids_lst: m elements with indexes from sub-matrix
        :param asst_rela: Omit
        :return: [The same as that of '_xfrm_iput']
        2020-10-27
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
    def _get_lapl(self, iput_w, qries_insub: list, incl_qry: bool, asst_rela: OperAsst):
        """
        Get Laplacian matrix
        :param iput_w: Input matrix W = [wij]_m×m
        :param qries_insub: k queries with indexes from sub-matrix
        :param asst_rela: Omit
        :return: lapl_w∈𝓡^(m×m)
        2020-10-27
        """
        pass

    @staticmethod
    def __lapl_post(lapl_mats, qries_insub: list, incl_qry: bool):
        """
        :param lapl_mats: 𝓡^(m×m) or 𝓡^(k×m×m)
        :param qries_insub: k queries with indexes from sub-matrix
        :param incl_qry: Whether self-similarity/self-distance is learned
        :param asst_rela: Omit
        :return: w_mats∈𝓡^(k×m×m or 1×m×m)
        """
        #
        if not incl_qry:
            # Zero setting for the elements related to the query
            for qry_idx in range(len(qries_insub)):
                qry = qries_insub[qry_idx]
                lapl_mats[qry_idx, :, qry] = 0
                lapl_mats[qry_idx, qry, :] = 0
        #
        return lapl_mats

    @staticmethod
    def _lapl_diag(lapl_mats, asst_rela: OperAsst):
        """
        :param lapl_mats: w_mats∈𝓡^(k×m×m or 1×m×m)
        :param asst_rela: Omit
        :return: 𝓡^(k×m×1 or 1×m×1)
        """
        # Get diagonal matrix: D, simplified as column-vector: vec, D=diag(vec)
        return asst_rela.sum2last(lapl_mats)

    def __norm_lapl(self, lapl_mats, qries_insub: list, incl_qry: bool, asst_rela: OperAsst):
        """
        Norm Laplacian sub-matrix
        :param lapl_mats: w_mats∈𝓡^(k×m×m or 1×m×m)
        :param qries_insub: k queries with indexes from sub-matrix
        :param incl_qry: Whether self-similarity/self-distance is learned
        :param asst_rela: Omit
        :return: (1)diag_cols∈𝓡^(k×m×1 or 1×m×1)
                 (2)norm_mats∈𝓡^(k×m×m or 1×m×m)
        2020-10-27
        """
        #
        d_cols = self._lapl_diag(lapl_mats, asst_rela)
        # Normalize w_mats
        d_norms = asst_rela.pow(asst_rela.overturn(d_cols), -0.5)
        norm_mats = d_norms * lapl_mats
        norm_mats = asst_rela.xpos(d_norms, 1, 2) * norm_mats
        # Define identity matrix
        diag_cols = asst_rela.ones(d_cols.shape)
        if not incl_qry:
            for qry_idx in range(len(qries_insub)):
                diag_cols[qry_idx, qries_insub[qry_idx], 0] = 0
        #
        return diag_cols, norm_mats
