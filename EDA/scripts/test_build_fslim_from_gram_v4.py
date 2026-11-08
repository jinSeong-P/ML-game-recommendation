import unittest
import numpy as np
from scipy.optimize import nnls, minimize
from build_fslim_from_gram_v4 import fit_column, neighbor_order


class SufficientStatisticFslimTests(unittest.TestCase):
    def test_l2_fit_matches_augmented_nonnegative_least_squares(self):
        x=np.array([[1.,0,1],[0,1,1],[1,1,0],[1,0,0],[0,1,0]])
        y=np.array([1.,0,1,1,0]);l2=.7
        expected,_=nnls(np.vstack([x,np.sqrt(l2)*np.eye(3)]),np.r_[y,np.zeros(3)])
        actual,info=fit_column(x.T@x,x.T@y,y@y,0,l2,tol=1e-12)
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
        self.assertLess(info['normalized_kkt'],1e-8)

    def test_l1_fit_matches_independent_bound_constrained_optimizer(self):
        x=np.array([[1.,0,1],[0,1,1],[1,1,0],[1,0,0],[0,1,0]])
        y=np.array([1.,0,1,1,0]);q=x.T@x;c=x.T@y;l1=.4;l2=.7
        objective=lambda w:.5*w@q@w-c@w+l1*w.sum()+.5*l2*(w@w)
        gradient=lambda w:q@w-c+l1+l2*w
        expected=minimize(objective,np.ones(3)*.1,jac=gradient,bounds=[(0,None)]*3,
            method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-15})
        actual,info=fit_column(q,c,y@y,l1,l2,tol=1e-12)
        np.testing.assert_allclose(actual,expected.x,atol=1e-7,rtol=1e-6)
        self.assertTrue(info['converged'])

    def test_neighbor_order_excludes_self_and_breaks_ties_by_app_id(self):
        ix,_=neighbor_order(np.array([4,2,2,0]),0,np.array([4,4,4,0]),np.array([10,20,30,40]),500)
        np.testing.assert_array_equal(ix,[1,2])


if __name__=='__main__':unittest.main()
