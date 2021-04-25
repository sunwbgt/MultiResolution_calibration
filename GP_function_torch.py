from scipy.linalg import solve_triangular as trisolve
from scipy import optimize as opt
from numpy.linalg import inv as inv
#Power exponential correlation funciton
import numpy as np
import torch as tch
tch.set_default_dtype(tch.float64)

def __covmat(x1, x2, gammav):
    """Return the covariance between x1 and x2 given parameter gammav."""

    x1 = x1/tch.exp(gammav)
    x2 = x2/tch.exp(gammav)
    V = tch.zeros([x1.size()[0], x2.size()[0]])
    R = tch.ones((x1.size()[0], x2.size()[0]))*(1-(10**-6))
    for k in range(0, gammav.shape[0]):
        S = tch.abs(x1[:, k].reshape(-1, 1) - x2[:, k])
        R *= (1 + S)
        V -= S
    R *= tch.exp(V)
    R += (10**-6)*(V > (-10**(-12)))
    return R

def __covmat_pred(x1, x2, gammav):
    """Return the covariance between x1 and x2 given parameter gammav."""

    x1 = x1/tch.exp(gammav)
    x2 = x2/tch.exp(gammav)
    V = tch.zeros([x1.size()[0], x2.size()[0]])
    R = tch.ones((x1.size()[0], x2.size()[0]))*(1-(10**-6))
    for k in range(0, gammav.shape[0]):
        S = tch.abs(x1[:, k].reshape(-1, 1) - x2[:, k])
        R *= (1 + S)
        V -= S
    R *= tch.exp(V)
    return R


def Internal_neglogpost(theta, ys, Xs,theta0):
    n, d = Xs.shape
    ys = ys - tch.mean(ys)
    Sigma_t = __covmat(Xs,Xs,theta)
    _,logdet = tch.linalg.slogdet(Sigma_t)
    sigma2_hat = tch.mean(tch.solve(ys[:,None], Sigma_t).solution*ys[:,None],0)

    neglogpost = 0.1*tch.sum((theta-theta0)**2)
    neglogpost += n/2*np.squeeze(tch.log(sigma2_hat))+1/2*logdet#
    return neglogpost


def gpbuild(theta, ys, Xs):
    n, d = Xs.shape
    Sigma_t = __covmat(Xs,Xs,theta)
    fitdict = {}
    fitdict['mu'] = tch.mean(ys)
    fitdict['pw'] = tch.solve(ys[:,None]-fitdict['mu'], Sigma_t).solution
    fitdict['sigma2_hat'] = tch.sum(fitdict['pw']*ys[:,None],0)
    fitdict['theta'] = theta
    fitdict['Si'] = tch.linalg.inv(Sigma_t)
    fitdict['Xs'] = Xs
    return fitdict

def gppred(Xn,fitdict):
    muhat = fitdict['mu'] + __covmat_pred(Xn,fitdict['Xs'],fitdict['theta']) @ fitdict['pw']
    return muhat

def covpred(Xn,fitdict):
    S1 = (10**-6) +  __covmat_pred(Xn,Xn,fitdict['theta'])
    S2 =  __covmat_pred(Xn,fitdict['Xs'],fitdict['theta'])
    S = S1 - S2 @ fitdict['Si'] @ S2.T
    return fitdict['sigma2_hat']*S