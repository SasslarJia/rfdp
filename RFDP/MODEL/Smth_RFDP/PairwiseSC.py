#
from MODEL.Smth_RFDP.SmthCstr import SmthCstr
from MODEL.Asst_RFDP.OperAsst import OperAsst


class PairwiseSC(SmthCstr):
    """ Smoothness constraint reflects pairwise relationship
    2020-10-27
    """

    def __init__(self, is_norm: bool = False):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        super(PairwiseSC, self).__init__(is_norm)

    def _xfrm_iput(self, sub_w, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        return sub_w

    def _get_lapl(self, iput_w, qries_insub: list, incl_qry: bool, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        #
        if not incl_qry:
            return asst_rela.expand_strt(iput_w, len(qries_insub))
        else:
            return asst_rela.expand_strt(iput_w, 1)
