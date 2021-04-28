from scipy.linalg import solve_triangular as trisolve
from scipy import optimize as opt
from numpy.linalg import inv as inv
#Power exponential correlation funciton
import numpy as np
import torch as tch
tch.set_default_dtype(tch.float64)

def __covmat(x1, x2, gammav):
    """Return the covariance between x1 and x2 given parameter gammav."""

    x1 = x1/tch.exp(gammav[:-1])
    x2 = x2/tch.exp(gammav[:-1])
    V = tch.zeros([x1.size()[0], x2.size()[0]])
    R = tch.ones((x1.size()[0], x2.size()[0]))*1/(1+tch.exp(gammav[-1]))
    for k in range(0, gammav.shape[0]-1):
        S = tch.abs(x1[:, k].reshape(-1, 1) - x2[:, k])
        R *= (1 + S)
        V -= S
    R *= tch.exp(V)
    R += tch.exp(gammav[-1])/(1+tch.exp(gammav[-1]))*(V > (-10**(-8)))
    return R

def __covmat_pred(x1, x2, gammav):
    """Return the covariance between x1 and x2 given parameter gammav."""

    x1 = x1/tch.exp(gammav[:-1])
    x2 = x2/tch.exp(gammav[:-1])
    V = tch.zeros([x1.size()[0], x2.size()[0]])
    R = tch.ones((x1.size()[0], x2.size()[0]))*1/(1+tch.exp(gammav[-1]))
    for k in range(0, gammav.shape[0]-1):
        S = tch.abs(x1[:, k].reshape(-1, 1) - x2[:, k])
        R *= (1 + S)
        V -= S
    R *= tch.exp(V)
    return R


def Internal_neglogpost(theta, ys, Xs,theta0):
    n, d = Xs.shape
    ys = ys - tch.mean(ys)
    Sigma_t = __covmat(Xs,Xs,theta)
    logdet = tch.logdet(Sigma_t)
    sigma2_hat = tch.sum(tch.solve(ys[:,None], Sigma_t).solution*ys[:,None],0)

    neglogpost =  10*tch.sum((theta-theta0)**2) #start out with prior #second roder
    neglogpost = neglogpost+n/2*tch.log(sigma2_hat)+1/2*logdet#
    return neglogpost


def gpbuild(theta, ys, Xs):
    n, d = Xs.shape
    Sigma_t = __covmat(Xs,Xs,theta)
    fitdict = {}
    fitdict['mu'] = tch.mean(ys)
    fitdict['pw'] = tch.solve(ys[:,None]-fitdict['mu'], Sigma_t).solution
    fitdict['sigma2_hat'] = tch.sum(fitdict['pw']*ys[:,None],0)
    fitdict['theta'] = theta
    fitdict['Xs'] = Xs
    fitdict['sigmainv'] = tch.inverse(Sigma_t)
    return fitdict

def gppred(Xn, fitdict):
    muhat = fitdict['mu'] + __covmat_pred(Xn,fitdict['Xs'],fitdict['theta']) @ fitdict['pw']
    return muhat

def covpred(Xn, fitdict):
    corrmat = __covmat_pred(Xn, fitdict['Xs'], fitdict['theta'])
    covhat = __covmat_pred(Xn, Xn, fitdict['theta']) - corrmat @ fitdict['sigmainv'] @ corrmat.transpose(0, 1)
    return covhat