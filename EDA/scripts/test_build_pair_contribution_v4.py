import unittest
import numpy as np
from build_pair_contribution_v4 import mixture_cosine


class PairStatisticTests(unittest.TestCase):
    def test_mixture_equals_direct_account_weighted_cosine(self):
        x=np.array([[1.,1,0],[1,1,1],[0,0,1],[1,0,1]])
        degree=x.sum(axis=1);w0=1/degree;w1=1/(degree*np.maximum(1,degree-1))
        a=x.T@(w0[:,None]*x);b=x.T@(w1[:,None]*x)
        for rho in (0.,.25,.5,1.):
            gram=x.T@(((1-rho)*w0+rho*w1)[:,None]*x)
            expected=gram[0]/np.sqrt(gram[0,0]*np.diag(gram));expected[0]=0
            actual=mixture_cosine(a[0],b[0],np.diag(a),np.diag(b),0,rho)
            np.testing.assert_allclose(actual,expected,atol=1e-14)


if __name__=='__main__':unittest.main()
