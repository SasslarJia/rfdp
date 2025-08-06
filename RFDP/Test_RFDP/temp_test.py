#
import scipy.io as scio
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Aff_RFDP.Dist2Aff import Dist2Aff
from MODEL.Aff_RFDP.KnnRstct import KnnRstct
#
# path = '..\\..\\..\\DataSet\\SC_SWDLEAF.mat'
path = '../../DataSet/IDSC_SWDLEAF.mat'
dis_mat = scio.loadmat(path)['all_dists']
cls_num = 15
ele_num = 75
k_num = 20
#
asst = OperAsstTorch()
#
aff_mat = Dist2Aff(k_num, asst_rela=asst)(dis_mat)
knn_aff = KnnRstct(k_num, asst_rela=asst)(aff_mat)
#
newData = dict()
newData['aff_mat'] = OperAsstNumpy().chg_fmt(aff_mat)
newData['knn_aff'] = OperAsstNumpy().chg_fmt(knn_aff)
#
newPath = 'E://dataNew.mat'
scio.savemat(newPath, newData)
#
# dataFile = 'E://data.mat'
# data = scio.loadmat(dataFile)
a = 1



