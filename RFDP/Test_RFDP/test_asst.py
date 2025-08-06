#
import unittest as untest
import random
#
import numpy as np
#
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Enum_RFDP.AjcnEnum import AjcnEnum


class TestAsst(untest.TestCase):
    """
    TestCase for AffXfrm
    2020-10-1
    """

    # Setup Environment
    def setUp(self):
        print('\nAsst TestCase setup')
        #
        self.base_err = 0.00001
        #
        print('Data is loaded')

    # Clear Environment
    def tearDown(self):
        print('AffXfrm TestCase teardown')

    # <editor-fold desc="Test primary functions in Asst_RFDP">
    def test_Fctn(self):
        print('Test Functions in Asst_RFDP')
        #
        mat1 = np.array([[2, 1, 3], [6, 5, 4], [8, 9, 7]])
        mat2 = np.array([[3.3, 2.2, 1.1], [4.4, 5.5, 6.6], [8.8, 9.9, 7.7]])
        mat3 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]])
        mat4 = np.array([[1, 2, 3], [4, 5, 6]])
        #
        asst = OperAsstNumpy()
        #
        # Check repeat(vector)
        rept_val1 = np.array([[16, 16], [15, 15], [14, 14]])
        self.mat_err(rept_val1 - asst.repeat_2d(mat1.sum(0)[:, None], 2), self.base_err)
        rept_val2 = np.array([[16.5, 17.6, 15.4], [16.5, 17.6, 15.4], [16.5, 17.6, 15.4]])
        self.mat_err(rept_val2 - asst.repeat_2d(mat2.sum(0)[None, :], 3), self.base_err)
        #
        # Check get_ajcn
        ajcn_val = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 0]])
        self.mat_err(ajcn_val - asst.get_ajcn(mat1, 5.0, AjcnEnum.GrtEq), self.base_err)
        #
        # Check z = x^y
        pow_val = np.array([10.89, 2.2, 1.331])
        self.mat_err(pow_val - asst.pow(mat2[0, :], mat1[0, :]), self.base_err)
        #
        # Check y = exp(x)
        exp_val = np.array([[7.389056, 2.718282, 20.0855369],
                            [403.428793, 148.413159, 54.598150],
                            [2980.957987, 8103.0839276, 1096.6331584]])
        self.mat_err(exp_val - asst.exp(mat1), self.base_err)
        #
        # Check y = sum(iput_mat, 1)
        col_val = np.array([[6], [15], [24]])
        self.mat_err(col_val - asst.sum2last(mat1), self.base_err)
        col_vals = np.array([[[6], [15], [24]], [[9], [18], [27]]])
        self.mat_err(col_vals - asst.sum2last(mat3), self.base_err)
        #
        # Check y = mul_diag(mat1, mat2)
        mul_val = np.array([[6.6, 2.2, 3.3], [26.4, 27.5, 26.4], [70.4, 89.1, 53.9]])
        self.mat_err(mul_val - asst.mul_diag(mat1, mat2), self.base_err)
        mul_vals = np.array([[[6, 12, 18], [60, 75, 90], [168, 192, 216]],
                             [[18, 27, 36], [90, 108, 126], [216, 243, 270]]])
        self.mat_err(mul_vals - asst.mul_diag(asst.sum2last(mat3), mat3), self.base_err)
        #
        # Check topk(matrix)
        topk_ids = np.array([[0, 1], [1, 2], [0, 2]])
        _, tmp_top = asst.topk_2d(mat1, 2, dim=1)
        self.mat_err(topk_ids - tmp_top, self.base_err)
        #

        a = 1

    # </editor-fold>

    # <editor-fold desc="Comparing different functions in Asst_RFDP">
    def test_Cmpr(self):
        print('Comparing Different Functions in Asst_RFDP')
        #
        low_val = 2
        up_val = 40
        #
        dim = np.random.randint(low_val, up_val)
        parm = np.random.randint(low_val, dim)
        mat1 = np.random.rand(dim, dim)
        mat2 = np.random.rand(dim, dim)
        self.cmpr_err = self.base_err * dim
        thld = random.random()
        #
        self.torch_asst = OperAsstTorch()
        # asst_lst = [AsstRelaTorch()]
        asst_lst = [OperAsstNumpy(), OperAsstTorch()]
        # Check functions in Asst_RFDP
        rept_row, rept_col, expan_val, ajcn_val, pow_val, exp_val, sum2col_val, \
        mul_diag_val, topk_val = None, None, None, None, None, None, None, None, None
        #
        for asst in asst_lst:
            # Check repeat(vector)
            rept_row, rept_col = self.subtest_repeat_2d(asst, mat1, parm, rept_row, rept_col)

            # Check Expand
            expan_val = self.subtest_expand_strt(asst, mat1, parm, expan_val)

            # Check get_ajcn
            ajcn_val = self.subtest_get_ajcn(asst, mat1, ajcn_val, thld)

            # Check z = x^y
            pow_val = self.subtest_pow(asst, mat1, parm, pow_val)

            # Check y = exp(x)
            exp_val = self.subtest_exp(asst, mat1, exp_val)

            # Check y = sum(iput_mat, 1)
            sum2col_val = self.subtest_sum2col(asst, mat1, sum2col_val)

            # Check y = mul_diag(mat1, mat2)
            mul_diag_val = self.subtest_mul_diag(asst, mat1, mat2, mul_diag_val)

            # Check topk(matrix)
            topk_val = self.subtest_topk_2d(asst, mat1, parm, topk_val)

        #
        print('Test primary functions in Asst_RFDP is finished')

    # </editor-fold>

    # <editor-fold desc="Check Matrix Error">
    def mat_err(self, iput_mat, err):
        self.assertEqual((abs(iput_mat.float()) < err).all(), True)
    # </editor-fold>

    # <editor-fold desc="Test 'repeat_2d'">
    def subtest_repeat_2d(self, asst, iput_mat, iput_parm, rept_row, rept_col):
        tmp_row = self.torch_asst.chg_fmt(asst.repeat_2d(asst.chg_fmt(iput_mat).sum(0)[:, None], iput_parm))
        if rept_row is not None:
            self.mat_err(tmp_row - rept_row, self.cmpr_err)
        # Check repeat(vector')
        tmp_col = self.torch_asst.chg_fmt(asst.repeat_2d(asst.chg_fmt(iput_mat).sum(0)[None, :], iput_parm))
        if rept_col is not None:
            self.mat_err(tmp_col - rept_col, self.cmpr_err)
        #
        return tmp_row, tmp_col

    # </editor-fold>

    def subtest_expand_strt(self, asst, iput_mat, iput_parm, expan_val):
        tmp_expan = self.torch_asst.chg_fmt(asst.expand_strt(asst.chg_fmt(iput_mat), iput_parm))
        if expan_val is not None:
            self.mat_err(tmp_expan - expan_val, self.cmpr_err)
        #
        return tmp_expan

    # <editor-fold desc="Test 'get_ajcn'">
    def subtest_get_ajcn(self, asst, iput_mat, ajcn_val, thld):
        #
        tmp_ajcn = self.torch_asst.chg_fmt(asst.get_ajcn(asst.chg_fmt(iput_mat), thld))
        #
        if ajcn_val is not None:
            self.mat_err(tmp_ajcn - ajcn_val, self.cmpr_err)
        #
        return tmp_ajcn

    # </editor-fold>

    # <editor-fold desc="Test 'pow'">
    def subtest_pow(self, asst, iput_mat, iput_parm, pow_val):
        # Check z = x^y
        tmp_pow = self.torch_asst.chg_fmt(asst.pow(asst.chg_fmt(iput_mat), iput_parm))
        if pow_val is not None:
            self.mat_err(tmp_pow - pow_val, self.cmpr_err)
        #
        return tmp_pow

    # </editor-fold>

    # <editor-fold desc="Test 'exp'">
    def subtest_exp(self, asst, iput_mat, exp_val):
        # Check y = exp(x)
        tmp_exp = self.torch_asst.chg_fmt(asst.exp(asst.chg_fmt(iput_mat)))
        if exp_val is not None:
            self.mat_err(tmp_exp - exp_val, self.cmpr_err)
        #
        return tmp_exp

    # </editor-fold>

    # <editor-fold desc="Test 'sum2col'">
    def subtest_sum2col(self, asst, iput_mat, sum2col_val):
        # Check y = sum(iput_mat, 1)
        tmp_sum2col = self.torch_asst.chg_fmt(asst.sum2last(asst.chg_fmt(iput_mat)))
        if sum2col_val is not None:
            self.mat_err(tmp_sum2col - sum2col_val, self.cmpr_err)
        #
        return tmp_sum2col

    # </editor-fold>

    # <editor-fold desc="Test 'mul_diag'">
    def subtest_mul_diag(self, asst, mat1, mat2, mul_diag_val):
        # Check y = mul_diag(mat1, mat2)
        tmp_mul_diag = self.torch_asst.chg_fmt(asst.mul_diag(asst.chg_fmt(mat1), asst.chg_fmt(mat2)))
        if mul_diag_val is not None:
            self.mat_err(tmp_mul_diag - mul_diag_val, self.cmpr_err)
        #
        return tmp_mul_diag

    # </editor-fold>

    # <editor-fold desc="Test 'topk_2d'">
    def subtest_topk_2d(self, asst, iput_mat, iput_parm, topk_val):
        # Check topk(matrix)
        _, tmp_topk = asst.topk_2d(asst.chg_fmt(iput_mat), iput_parm)
        tmp_topk = self.torch_asst.cvrt2int(self.torch_asst.chg_fmt(tmp_topk))
        if topk_val is not None:
            self.mat_err(tmp_topk - topk_val, self.cmpr_err)
        return tmp_topk

    # </editor-fold>

# if __name__ == '__main__':
#     untest.main()
