#
import scipy.io as scio
#
import torch
#
from MODEL.Aff_RFDP.Dist2Aff import Dist2Aff
from MODEL.Aff_RFDP.KnnRstct import KnnRstct
from MODEL.Frmwk_RFDP.RfdpCtofOtcm import RfdpCtofOtcm
from MODEL.Iput_RFDP.IputWhl import IputWhl
#
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
#
from MODEL.Fit_RFDP.HyDiffFC import HyDiffFC
from MODEL.Frmwk_RFDP.RfdpSuplExtn import RfdpSuplExtn
from MODEL.Sol_RFDP.ItrtExpGrw import ItrtExpGrw
#
from EVAL.RefRslt import RefRslt
from EVAL.EvalPrcsRcal import EvalPrcsRcal

dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data():
    # path = '..\\..\\..\\DataSet\\SC_1000.mat'
    path = '../../DataSet/IDSC_1000.mat'
    dis_mat = scio.loadmat(path)['all_dists']
    cls_num = 50
    ele_num = 20
    #
    # path = '..\\..\\..\\DataSet\\SC_ANIM.mat'
    # path = '..\\..\\..\\DataSet\\IDSC_ANIM.mat'
    # dis_mat = scio.loadmat(path)['all_dists']
    # cls_num = 20
    # ele_num = 100
    #
    # path = '..\\..\\..\\DataSet\\SC_SWDLEAF.mat'
    # path = '..\\..\\..\\DataSet\\IDSC_SWDLEAF.mat'
    # dis_mat = scio.loadmat(path)['all_dists']
    # cls_num = 15
    # ele_num = 75
    #
    # qry_lst = list([list([idx]) for idx in range(1000)])
    qry_lst = list([list([idx]) for idx in range(cls_num * ele_num)])
    #
    k_num = 20
    #
    return dis_mat, cls_num, ele_num, qry_lst, k_num


def iput_whl(dis_mat, k_num):
    # asst = AsstRelaNumpy()
    asst = OperAsstTorch()
    #
    # aff_mat = 100*Dist2Aff(k_num, asst_rela=asst)(dis_mat)
    aff_mat = Dist2Aff(k_num, asst_rela=asst)(dis_mat)
    knn_aff = KnnRstct(k_num, asst_rela=asst)(aff_mat)
    #
    return IputWhl(smth_w=knn_aff, fit_w=knn_aff, full_w=aff_mat, asst_rela=asst)
    # return IputLyr(smth_w=knn_aff, fit_w=knn_aff, full_w=aff_mat, asst_rela=asst)


def test_rfdp():
    print('case test_affxfrom')
    #
    dis_mat, cls_num, ele_num, qry_lst, k_num = get_data()
    #
    mdl_iput = iput_whl(dis_mat, k_num)
    #
    qry_lst = list([[1, 2], [2, 1], [3], [5], [6], [7], [8], [6]])
    #
    # Hybrid
    # oput_rslts = Rfdp(mdl_iput, fit_cstr=HybrdFC(), sol_proc=ItrtExpGrw())(qry_lst, ele_num, False)
    # oput_rslts = Rfdp(mdl_iput, fit_cstr=HybrdFC(), sol_proc=ItrtLinGrw())(qry_lst, ele_num, False)
    #
    # oput_rslts = Rfdp(mdl_iput=mdl_iput, fit_cstr=HybrdFC(100), sol_proc=ItrtExpGrw(),
    #                   incl_qry=False)(iput_qries=qry_lst, rerank_num=ele_num, cptl_oput=True)
    # oput_rslts = RfdpCtofOtcm(mdl_iput=mdl_iput, fit_cstr=HybrdFC(), sol_proc=ItrtExpGrw(),
    #                           incl_qry=False)(iput_qries=qry_lst, ctof_num=ele_num)
    #
    # oput_rslts = RfdpCtofOtcm(mdl_iput=mdl_iput, fit_cstr=HyDiffFC(0.1), sol_proc=ItrtExpGrw(),
    #                           incl_qry=False)(iput_qries=list([[1, 2, 3, 4, 5], [6], [7, 8]]), iput_num=ele_num)
    oput_rslts = RfdpCtofOtcm(mdl_iput=mdl_iput, fit_cstr=HyDiffFC(0.1), sol_proc=ItrtExpGrw(),
                              incl_qry=False)(iput_qries=qry_lst, iput_len=ele_num)
    #
    # oput_rslts = Rfdp(mdl_iput=mdl_iput, fit_cstr=WgtFC(100), sol_proc=ItrtExpGrw(),
    #                   incl_qry=False)(iput_qries=qry_lst, rerank_num=ele_num, cptl_oput=True)
    # oput_rslts = Rfdp(mdl_iput=mdl_iput, fit_cstr=WgtFC(), sol_proc=ItrtExpGrw(),
    #                   incl_qry=False)(iput_qries=qry_lst, rerank_num=ele_num, cptl_oput=True)
    #
    # oput_rslts = Rfdp(mdl_iput=mdl_iput, fit_cstr=OazFC(100), sol_proc=ItrtExpGrw(),
    #                   incl_qry=True)(iput_qries=qry_lst, rerank_num=ele_num, cptl_oput=True)
    # oput_rslts = Rfdp(mdl_iput=mdl_iput, fit_cstr=OazFC(), sol_proc=ItrtExpGrw(),
    #                   incl_qry=True)(iput_qries=qry_lst, rerank_num=ele_num, cptl_oput=True)
    #
    # oput_rslts = Rfdp(mdl_iput=mdl_iput, smth_cstr=InclangSC(), fit_cstr=WgtFC(),
    #                   sol_proc=ItrtExpGrw())(qry_lst, ele_num, False)
    #
    EvalPrcsRcal.plot(oput_rslts, RefRslt.easy_cstr(qry_lst, cls_num, ele_num), 'bx-')
    EvalPrcsRcal.dply(y_min=0.8, y_max=1.0)
    #
    print('TestRFDP')


test_rfdp()
