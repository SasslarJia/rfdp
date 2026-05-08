#
try:
    import torch
except ModuleNotFoundError:
    class _TorchUnavailable(object):
        class Tensor(object):
            pass

        class device(object):
            def __init__(self, *_args, **_kwargs):
                self.type = 'cpu'

        class cuda(object):
            @staticmethod
            def is_available() -> bool:
                return False

        def __getattr__(self, name):
            raise ModuleNotFoundError("torch is required for OperAsstTorch")

    torch = _TorchUnavailable()
#
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Enum_RFDP.AjcnEnum import AjcnEnum
from RfdpLog.PrntLog_RFDP import prnt_log


class OperAsstTorch(OperAsst):
    #
    def __init__(self, dvc: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self._dvc = dvc

    # <editor-fold desc="Method group of 'chg_fmt'">
    def _chg_fmt(self, iput_mat) -> torch.Tensor:
        """ Change input matrix into the expected format (Protected Method)
        :param iput_mat:
        :return:
        2020-10-5
        """
        if type(iput_mat) != torch.Tensor:
            return torch.tensor(iput_mat, dtype=torch.float)
        else:
            return iput_mat

    # </editor-fold>

    # <editor-fold desc="Method group of 'expand_strt'">
    def _expand_strt(self, iput_mat: torch.Tensor, num: int):
        """ Expand input-matrix with a extra dim0
        :param iput_mat: p×q
        :param num: n
        :return: n×p×q
        2020-12-18
        """
        tmp = [1] * len(iput_mat.shape)
        tmp.insert(0, num)
        return self.__to_dfltdvc(iput_mat.unsqueeze(dim=0).repeat(tuple(tmp)))
    # </editor-fold>

    # <editor-fold desc="Method group of 'expand_dim'">
    def _expand_dim(self, iput_mat, idx: int):
        """ Expand input-matrix in axis idx
        :param iput_mat: p×q
        :param idx: insert index = 0, 1 or 2
        :return: 1×p×q, p×1×q or p×q×1
        2020-12-18
        """
        return self.__to_dfltdvc(iput_mat.unsqueeze(dim=idx))
    # </editor-fold>

    # <editor-fold desc="Method group of 'repeat_2d'">
    def _repeat_2d(self, iput_vec: torch.Tensor, rept: int, dim: int = 0) -> torch.Tensor:
        """
        :param iput_vec:
        :param rept:
        :param dim:
        :return:
        2020-10-6
        """
        #
        if dim == 0:
            return self.__to_dfltdvc(iput_vec.repeat(rept, 1))
        else:
            return self.__to_dfltdvc(iput_vec.repeat(1, rept))

    # </editor-fold>

    # <editor-fold desc="Method group of 'get_ajcn'">
    def _get_ajcn(self, iput_mat: torch.Tensor, thld: float, ajcn_type: AjcnEnum) -> None:
        """ Get adjacent matrix from input matrix (Protected Method)
        :param iput_mat:
        :return:
        2020-10-5
        """
        #
        if ajcn_type == AjcnEnum.Grt:
            return torch.where(self.__to_dfltdvc(iput_mat) > thld,
                               torch.ones(iput_mat.shape), torch.zeros(iput_mat.shape))
        #
        if ajcn_type == AjcnEnum.GrtEq:
            return torch.where(self.__to_dfltdvc(iput_mat) >= thld,
                               torch.ones(iput_mat.shape), torch.zeros(iput_mat.shape))
        #
        if ajcn_type == AjcnEnum.Sml:
            return torch.where(self.__to_dfltdvc(iput_mat) < thld,
                               torch.ones(iput_mat.shape), torch.zeros(iput_mat.shape))
        #
        if ajcn_type == AjcnEnum.SmlEq:
            return torch.where(self.__to_dfltdvc(iput_mat) <= thld,
                               torch.ones(iput_mat.shape), torch.zeros(iput_mat.shape))
        #
        prnt_log.error('Unknown Input AjcnEnum!')
        #
        return None

    # </editor-fold>

    # <editor-fold desc="Method group of 'pow'">
    def _pow(self, iput_base, iput_exp: torch.Tensor) -> torch.Tensor:
        """ Calculate iput_base^iput_exp (Protected Method)
        :param iput_base:
        :param iput_exp:
        :return:
        2020-10-5
        """
        return iput_base.pow(iput_exp)

    # </editor-fold>

    # <editor-fold desc="Method group of 'exp'">
    def _exp(self, iput_mat: torch.Tensor) -> torch.Tensor:
        """ Calculate exp(iput) (Protected Method)
        :param iput_mat:
        :return:
        2020-10-5
        """
        return self.__to_dfltdvc(iput_mat).exp()

    # </editor-fold>

    # <editor-fold desc="Method group of 'sum2col'">
    def _sum(self, iput_mat, neg_dim, keepdim: bool):
        """
        :param iput_mat:
        :return:
        2020-10-26
        """
        return torch.sum(iput_mat, dim=len(iput_mat.shape) + neg_dim, keepdim=keepdim)

    # </editor-fold>

    # <editor-fold desc="Method group of 'ones'">
    def _ones(self, iput):
        return torch.ones(iput)
    # </editor-fold>

    # <editor-fold desc="Method group of 'zeros'">
    def _zeros(self, iput):
        return torch.zeros(iput)
    # </editor-fold>

    # <editor-fold desc="Method group of 'eye'">
    def _eye(self, n):
        return torch.eye(n)
    # </editor-fold>

    # <editor-fold desc="Method group of '_mul_diag'">
    def _mul_diag(self, fir_iput, sec_iput):
        #
        return fir_iput * sec_iput

    # </editor-fold>

    # <editor-fold desc="Method group of 'cvrt2float'">
    def _cvrt2float(self, iput: torch.Tensor) -> torch.Tensor:
        """
        :param iput:
        :return:
        2020-10-5
        """
        return iput.float()

    # </editor-fold>

    # <editor-fold desc="Method group of 'cvrt2int'">
    def _cvrt2int(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        return iput.int()

    # </editor-fold>

    # <editor-fold desc="Method group of 'cvrt2flt'">
    def _cvrt2flt(self, iput):
        """
        :param iput:
        :return:
        2020-11-12
        """
        return iput.float()

    # </editor-fold>

    # <editor-fold desc="Method group of 'copy_mat'">
    def _copy_mat(self, iput: torch.Tensor):
        return iput.clone()

    # </editor-fold>

    # <editor-fold desc="Method group of 'topk_2d'">
    def _topk_2d(self, iput_mat: torch.Tensor, k_nn: int, dim: int = 0, rvrs: bool = False) \
            -> (torch.Tensor, torch.Tensor):
        #
        return torch.topk(self.__to_dfltdvc(iput_mat), k_nn, dim=dim, largest=rvrs)

    # </editor-fold>

    # <editor-fold desc="Method group of 'min_1d'">
    def _minmax_1d(self, iput_vector, max_flg: bool):
        #
        if not max_flg:
            idx = torch.argmin(iput_vector)
        else:
            idx = torch.argmax(iput_vector)
        #
        val = iput_vector[idx]
        #
        return val, idx

    # </editor-fold>

    # <editor-fold desc="Method group of 'mul_3d'">
    def _mul_3d(self, iput1, iput2) -> torch.Tensor:
        return self.__to_dfltdvc(torch.matmul(iput1, iput2))

    # </editor-fold>

    # <editor-fold desc="Method group of 'reshape'">
    def _reshape(self, iput: torch.Tensor, upd_shap):
        """
        Reshape input matrix by given shape
        :param iput: Iputmatrix
        :param upd_shap:
        :return:
        """
        return torch.reshape(iput, upd_shap)

    # </editor-fold>

    # <editor-fold desc="Method of 'to device'">
    def _to_dvc(self, iput_mat: torch.Tensor) -> torch.Tensor:
        return OperAsstTorch._to_spcfdvc(iput_mat, self._dvc)

    def __to_dfltdvc(self, iput_mat: torch.Tensor) -> torch.Tensor:
        return OperAsstTorch._to_spcfdvc(iput_mat, torch.device("cpu"))

    @staticmethod
    def _to_spcfdvc(iput_mat: torch.Tensor, iput_dvc: torch.device) -> torch.Tensor:
        #
        if not iput_mat.device.type == iput_dvc.type:
            return iput_mat.to(iput_dvc)
        else:
            return iput_mat

    # </editor-fold>

    # <editor-fold desc="Convert to list">
    def _cvrt2lst(self, iput) -> list:
        return self.__to_dfltdvc(iput).numpy().tolist()

    # </editor-fold>

    # <editor-fold desc="Transpose of input-matrix">
    def _xpos(self, iput: torch.Tensor, tpl):
        return iput.permute(tpl)
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
        return torch.where(cond, rslt_pos, rslt_neg)
    # </editor-fold>
