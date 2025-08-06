#
import unittest as untest
import scipy.io as scio
#
import time
import torch
import numpy as np
import random
#
from MODEL.Aff_RFDP.Dist2Aff import Dist2Aff
from MODEL.Aff_RFDP.KnnRstct import KnnRstct
from MODEL.Iput_RFDP.MdlIput import MdlIput
#
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
from MODEL.Smth_RFDP.PairwiseSC import PairwiseSC
from MODEL.Fit_RFDP.OazFC import OazFC
from MODEL.Fit_RFDP.WgtFC import WgtFC
from MODEL.Fit_RFDP.HyFC import HyFC
from MODEL.Frmwk_RFDP.Rfdp import Rfdp
#
from EVAL.RefRslt import RefRslt
from EVAL.EvalPrcsRcal import EvalPrcsRcal

dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestRFDP(untest.TestCase):
    """
    TestCase for AffXfrm
    2020-10-20
    """

    # Setup Environment
    def setUp(self):
        print('\nTestRFDP TestCase setup')
        #
        path = '..\\..\\DataSet\\SC_1000.mat'
        self.dis_mat = scio.loadmat(path)['all_dists']
        #
        # path = '..\\..\\DataSet\\IDSC_1400.mat'
        # self.dis_mat = scio.loadmat(path)['Score']
        #
        self.base_err = 0.00001
        #
        self.aff_mat1 = np.array([[10, 6, 1, 4],
                                  [6, 9, 2, 3],
                                  [1, 2, 10, 7],
                                  [4, 3, 7, 9]])
        self.qry1 = np.array([[0, 1, 2, 3]])
        #
        self.aff_mat2 = np.array([[100, 10, 88, 20, 66, 10],
                                  [10, 95, 15, 77, 30, 80],
                                  [88, 15, 110, 90, 25, 15],
                                  [20, 77, 90, 105, 20, 85],
                                  [66, 30, 25, 20, 115, 20],
                                  [10, 80, 15, 85, 20, 111]])
        self.qry2 = np.array([[0, 1, 2, 3, 4, 5]])
        #
        print('Data is loaded')

    # Clear Environment
    def tearDown(self):
        print('TestRFDP TestCase teardown')

    # def test_Fctn(self):
    #     print('Test Functions in Rfdp')
    #     #
    #     knn_val1 = np.array([[10, 6, 0, 0],
    #                          [6, 9, 0, 0],
    #                          [0, 0, 10, 7],
    #                          [0, 0, 7, 9]])
    #     #
    #     a = np.expand_dims(knn_val1, 1)
    #     asst = AsstRelaNumpy()
    #     shp = list(knn_val1.shape())
    #     shp.insert(0, 1)
    #     knn_val1.repeat(2, axis=2)
    #     a = 1

    #     xtra_mat1 = np.array([[10, 6, 0, 4],
    #                          [6, 9, 0, 0],
    #                          [0, 0, 10, 7],
    #                          [4, 0, 7, 9]])
    #     #
    #     d_col1 = np.array([20, 15, 17, 20])[:, None]
    #     a = d_col1.shape
    #     b = a[::-1]
    #     coef1 = np.array([1, 1, 1, 1])[:, None]
    #     cnst1 = np.array([[1, 0, 0, 0],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]])
    #     #
    #     asst1 = AsstRelaNumpy()
    #     # asst = AsstRelaTorch()
    #     # self.aff_mat1 = asst.chg_fmt(self.aff_mat1)
    #     #
    #     # Check KNN Affinity
    #     knn_aff1 = KnnRstct(1, asst_rela=asst1)(self.aff_mat1)
    #     self.mat_err(knn_val1 - knn_aff1, self.base_err)
    #     #
    #     # Check MdlIput
    #     mdl_iput = MdlIput(knn_aff1, knn_aff1, asst_rela=asst1)
    #     #
    #     # Check Smoothness Constraint and Fitting Constraint
    #     quad1 = Rfdp(mdl_iput, PairwiseSC(), OazFC(), SolProc())(list([1, 2]), 100)
    #     self.mat_err(d_col1 - quad1.d_col, self.base_err)
    #     self.mat_err(coef1 - quad1.coef, self.base_err)
    #     self.mat_err(cnst1 - quad1.cnst, self.base_err)

    # <editor-fold desc="Check Matrix Error">
    def mat_err(self, iput_mat, err):
        self.assertEqual(abs(iput_mat.sum()) < err, True)

    # </editor-fold>

    def test_Rfdp(self):
        print('case test_RFDP')
        #
        # asst = AsstRelaNumpy()
        asst = OperAsstTorch()
        # qries_lst = self._rndm_qries(5, 1125, 10)
        # qries_lst = list([list([idx]) for idx in range(1000)])
        qries_lst = list([list([idx]) for idx in range(60)])
        #
        k_num = 20
        #
        aff_mat = Dist2Aff(k_num, asst_rela=asst)(self.dis_mat)
        knn_aff = KnnRstct(k_num, asst_rela=asst)(aff_mat)
        #
        # mdl_iput = MdlIput(knn_aff, full_w=aff_mat, asst_rela=asst)
        # oput_rslts = Rfdp(mdl_iput, fit_cstr=OazFC())(qries_lst, 1000, True)
        #
        # mdl_iput = MdlIput(knn_aff, fit_w=knn_aff, full_w=aff_mat, asst_rela=asst)
        # oput_rslts = Rfdp(mdl_iput, fit_cstr=WgtFC())(qries_lst, 1000, False)
        #
        mdl_iput = MdlIput(knn_aff, fit_w=knn_aff, full_w=aff_mat, asst_rela=asst)
        oput_rslts = Rfdp(mdl_iput, smth_cstr=PairwiseSC(True), fit_cstr=HyFC())(qries_lst, 20, False)
        EvalPrcsRcal.plot(oput_rslts, RefRslt.easy_cstr(qries_lst, 50, 20))
        #
        print('TestRFDP')

    def _rndm_qries(self, qry_num: int, smpl_num: int, ceil_num: int) -> list:
        #
        qries_lst = list()
        #
        for idx in range(qry_num):
            tmp_num = random.randint(1, ceil_num)
            qries_lst.append(random.sample(range(smpl_num), tmp_num))
        #
        return qries_lst

# if __name__ == '__main__':
#     untest.main()
