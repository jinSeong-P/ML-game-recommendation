import unittest
import numpy as np
from scipy import sparse
from fast_gram_v4 import weighted_views


class GramConversionTests(unittest.TestCase):
    def test_one_conversion_is_bit_identical_to_repeated_csc_products(self):
        rng=np.random.default_rng(44)
        dense=(rng.random((200,53))<.15).astype(np.uint8);dense[:,0]=1
        x=sparse.csr_matrix(dense);degree=np.diff(x.indptr).astype(np.float64)
        for beta in (0.,.5,1.):
            old=x.astype(np.float32).tocsc()
            old.data=np.sqrt(np.power(degree,-beta).astype(np.float32)[old.indices])
            csr,rows,diagonal=weighted_views(x,beta)
            expected=np.vstack([(old.T.tocsr()[b:b+7]@old).toarray() for b in range(0,x.shape[1],7)])
            actual=np.vstack([(rows[b:b+13]@csr).toarray() for b in range(0,x.shape[1],13)])
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_allclose(np.diag(actual),diagonal,rtol=2e-6)
            direct=dense.astype(float).T@(np.power(degree,-beta)[:,None]*dense)
            np.testing.assert_allclose(actual,direct,rtol=3e-6,atol=1e-6)


if __name__=='__main__':unittest.main()
