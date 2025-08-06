#
import unittest as untest
import scipy.io as scio
#
import time
import numpy as np
#
from MODEL.Aff_RFDP.CnctCpnt import CnctCpnt
from MODEL.Aff_RFDP.Compose import Compose
from MODEL.Aff_RFDP.Dist2Aff import Dist2Aff
from MODEL.Aff_RFDP.KnnRstct import KnnRstct
#
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy


class TestAffXfrm(untest.TestCase):
    """
    TestCase for AffXfrm
    2020-10-1
    """

    # Setup Environment
    def setUp(self):
        print('\nAffXfrm TestCase setup')
        #
        path = '..\\..\\DataSet\\IDSC_1400.mat'
        self.dis_mat = scio.loadmat(path)['Score']  # nonsymmetric matrix
        #
        # path = '..\\..\\DataSet\\IDSC_1125.mat'
        # path = '..\\..\\DataSet\\SC_1000.mat'
        # self.dis_mat = scio.loadmat(path)['all_dists']
        #
        self.err = 0.00001
        #
        print('Data is loaded')

    # Clear Environment
    def tearDown(self):
        print('AffXfrm TestCase teardown')

    # <editor-fold desc="Test primary functions in Aff_RFDP">
    def test_affxfrom(self):
        print('case test_affxfrom')
        #
        k_num = 20
        #
        fir = time.clock()
        fir_lst = Compose([Dist2Aff(k_num, asst_rela=OperAsstTorch()), KnnRstct(k_num, asst_rela=OperAsstTorch())])
        fir_knn = fir_lst(self.dis_mat)
        fir_vec, fir_cc = CnctCpnt(OperAsstTorch())(fir_knn)
        print("Fir is ", time.clock() - fir)
        #
        sec = time.clock()
        sec_lst = Compose([Dist2Aff(k_num, asst_rela=OperAsstNumpy()), KnnRstct(k_num, asst_rela=OperAsstNumpy())])
        sec_knn = sec_lst(self.dis_mat)
        sec_vec, sec_cc = CnctCpnt(OperAsstNumpy())(sec_knn)
        print("Sec is ", time.clock() - sec)
        #
        self.assertEqual((np.array(fir_vec) - np.array(sec_vec)).sum() < self.err, True)
        #
        print('Test primary functions in Aff_RFDP is finished')
    # </editor-fold>


# if __name__ == '__main__':
#     untest.main()
