#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Enum_RFDP.DimEnum import DimEnum
from MODEL.Asst_RFDP.OperAsst import OperAsst


class AffXfrm(metaclass=ABCMeta):
    """Abstract Class for Constructing Weighted Matrix
    2018-12-1
    """

    #
    def __init__(self, asst_rela: OperAsst):
        """
        2020-10-3
        """
        # Initialize Global Variables
        self._asst_rela, self._oput_mat = asst_rela, None

    def __call__(self, iput_mat):
        """
        :param iput_mat: Input Matrix
        :return:
        """
        #
        self._iput_mat = AffXfrm.__get_sym(self._asst_rela.chg_fmt(iput_mat), self._asst_rela)
        # Get dim of matrix
        self._ele_num = self._asst_rela.get_ele_num(self._iput_mat, DimEnum.Dim2)
        #
        self._oput_mat = self._calc_oputmat(self._iput_mat)
        #
        return self._oput_mat

    @ABCFctn
    def _calc_oputmat(self, iput_mat):
        """ Abstract function for calculating output matrix
        :param iput_mat: Input matrix
        :return: Output matrix
        """
        pass

    # <editor-fold desc="Check whether input-matrix is symmetric,
    # and convert non-symmetric matrix to symmetric matrix">
    @staticmethod
    def __get_sym(iput, asst_rela: OperAsst):
        """
        :param iput: Input 2D matrix
        :return:
        2020-12-10
        """
        #
        xpos = asst_rela.xpos(iput, 0, 1)
        #
        tmp = asst_rela.cvrt2int((iput-xpos) != 0)
        #
        if asst_rela.sum2last(asst_rela.sum2lbo(tmp)) > 0:
            return (iput + xpos) * 0.5
        else:
            return iput
    # </editor-fold>

    def __repr__(self):
        #
        return self.__class__.__name__
