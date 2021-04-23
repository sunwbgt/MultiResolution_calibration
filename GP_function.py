from scipy.linalg import solve_triangular as trisolve
from scipy import optimize as opt
from numpy.linalg import inv as inv
#Power exponential correlation funciton
import numpy as np

def Internal_CorrMatPowerExp(x1,x2,theta,return_dCdtheta=False):
    tmax = 3
    expLS = np.exp(tmax*theta[0])
    minpower = 1
    maxpower = 1.95
    alpha = minpower + (theta[1]+1)/2 * (maxpower - minpower)
    diffmat=np.abs(np.subtract.outer(x1,x2))
    h = diffmat/expLS+1e-10
    C = np.exp(-(h)**alpha)
    if return_dCdtheta:
        col1=tmax*alpha*C*(diffmat/expLS)**alpha
        #print(h)
        col2=-C*h**alpha*(np.log(h)/2) * (maxpower - minpower)
        dCdtheta = np.hstack((col1,col2))
      
        dCdtheta[np.isnan(dCdtheta)] = 0
        out = [C, dCdtheta]
        return out
    else:
      return C

def Internal_CorrMatCauchySQ(x1, x2, theta, return_dCdtheta = False):
    diffmat =diffmat=np.abs(np.subtract.outer(x1,x2))
    
    expLS = np.exp(3*theta[0])
    expHE = np.exp(3*theta[1])
    h = diffmat/expLS+1e-10
    alpha = 2*np.exp(0+6)/(1+np.exp(0+6))
    halpha = h**alpha
    powr = -expHE/alpha    
    C = (1+halpha)^powr
    
    if return_dCdtheta:
        col1=3*expHE*((1+halpha)**(powr-1))*(halpha)
        col2=3*C*powr*np.log(1+halpha)
        dCdtheta = np.hstack((col1,col2))
        dCdtheta[np.isnan(dCdtheta)] = 0
        out = [C, dCdtheta]
        return out
    else:
      return C
def log_th(theta):
    res=np.log(1-theta)-np.log(1+theta)
    return res
def Internal_neglogpost(theta, ys, Xs):
    n, d = Xs.shape
    numpara = 2
    nugget = 1e-10
    Sigma_t = np.ones((n,n))
  
    for dimlcv in range(d):  # Loop over dimensions
        V =Internal_CorrMatPowerExp(Xs[:,dimlcv], Xs[:,dimlcv],theta[dimlcv*numpara+np.arange(numpara)])
        Sigma_t = Sigma_t*V
  
    Sigma_t = (1-nugget)*Sigma_t+ np.eye(Sigma_t.shape[0])*nugget
    
    try:
        Sigma_chol = np.linalg.cholesky(Sigma_t).T #LU decomposition (aka cholesky decomposition)
    except:
        return np.inf
    
    nsamples = ys.shape[0]    
    Sti_resid = inv(Sigma_chol)@ys#trisolve(Sigma_chol,trisolve(Sigma_chol,ys,trans=1)) #trans=1 is for transposing r i.e. r.Tx=b   
    sigma2_hat = np.sum(ys*Sti_resid,axis=0)/n   

    lDet = 2*np.sum(np.log(np.diag(Sigma_chol)))
    
    neglogpost =  0.1*np.sum(log_th(theta)**2) #start out with prior #second roder
    neglogpost =  neglogpost+0.1*np.sum(log_th(theta)**4) #start out with prior # 4th order
    
    if ys.ndim == 1:
      neglogpost = neglogpost+1/2*nsamples*np.log(sigma2_hat)+1/2*lDet#
    else:
      neglogpost = neglogpost+1/2*(nsamples*np.mean(np.log(sigma2_hat.reshape(-1)))+lDet)
    return neglogpost

#' Gradient of negative log likelihood posterior
def Internal_gneglogpost(theta,ys,Xs):
    n, d = Xs.shape
    numpara = 2
    nugget = 1e-10
    
    if(ys.ndim>1):
        ndim = ys.shape[1]

    Sigma_t = np.ones((n,n))
    dSigma_to = np.ones((d,n,numpara*n)) 
      
    for dimlcv in range(d): # Loop over dimensions
        V = Internal_CorrMatPowerExp(Xs[:,dimlcv], Xs[:,dimlcv],theta[dimlcv*numpara+np.arange(numpara)], return_dCdtheta=True)
        Sigma_t = Sigma_t*V[0]
        
        TV = np.hstack((V[0], V[0]))
        
        for dimlcv2 in range(d):
            if(dimlcv2==dimlcv):
                dSigma_to[dimlcv2,:,:] =  dSigma_to[dimlcv2,:,:]*V[1]
            else:
                dSigma_to[dimlcv2,:,:] =  dSigma_to[dimlcv2,:,:]*TV 
     
    Sigma_t = (1-nugget)*Sigma_t+ np.eye(Sigma_t.shape[0])*nugget  

    for dimlcv in range(d):
        dSigma_to[dimlcv,:,:] = (1-nugget)*dSigma_to[dimlcv,:,:]   
   
    try:
        Sigma_chol = np.linalg.cholesky(Sigma_t).T #LU decomposition (aka cholesky decomposition)
    except:
        print("chol error in gneglogpost #1, this can happen when neglogpost is Inf")     
    
    tempvec1 = inv(Sigma_chol)@ys#trisolve(Sigma_chol,trisolve(Sigma_chol,ys,trans=1))       
    sigma2_hat_supp = np.sum(ys*tempvec1,axis=0)/n
    
    if(ys.ndim>1):
        dsigma2_hat_supp = np.zeros((d*numpara, ys.shape[1]))
    else:
        dsigma2_hat_supp = np.zeros((d*numpara))
    
    dlDet_supp=np.zeros((d*numpara))
    
    for dimlcv in range(d):
        for paralcv in range(numpara):
            dSigma_supp = dSigma_to[dimlcv,:,(paralcv*n):((paralcv+1)*n)]##check this line
            tempvec2= dSigma_supp @ tempvec1##check this line
            
            if(dsigma2_hat_supp.ndim>1):
                if(dsigma2_hat_supp.shape[0]>1.5):
                    dsigma2_hat_supp[dimlcv*numpara+paralcv,:] =-np.sum(tempvec1*tempvec2,axis=0)/n
                else:
                    dsigma2_hat_supp[:,dimlcv*numpara+paralcv] =-np.sum(tempvec1*tempvec2,axis=0)/n
            else:
                dsigma2_hat_supp[dimlcv*numpara+paralcv] =-np.sum(tempvec1*tempvec2,axis=0)/n
        
            dlDet_supp[dimlcv*numpara+paralcv] =np.sum(np.diag(inv(Sigma_chol)@dSigma_supp))# trisolve(Sigma_chol,trisolve(Sigma_chol,dSigma_supp,trans=1))))
    lDet_supp = 2*np.sum(np.log(np.diag(Sigma_chol)))
    
    sigma2_hat = sigma2_hat_supp
    dsigma2_hat = dsigma2_hat_supp
    dlDet = dlDet_supp
    lDet = lDet_supp
    
    neglogpost =  0.1*np.sum(log_th(theta)**2) #start out with prior
    gneglogpost = -0.2*(log_th(theta))*((1/(1-theta))+1/(1+theta))
  
    neglogpost =  neglogpost+0.1*np.sum(log_th(theta)**4) #start out with prior
    gneglogpost = gneglogpost-0.1*4*(log_th(theta)**3)*((1/(1-theta))+1/(1+theta))
  
    if ys.ndim==1:
        neglogpost =neglogpost +1/2*(n*np.log(sigma2_hat)+lDet)#
        gneglogpost = gneglogpost+1/2*(n*dsigma2_hat / sigma2_hat+dlDet)#n
    else:
        neglogpost = neglogpost+1/2*(n*np.mean(np.log(sigma2_hat.reshape(-1)))+lDet)
        gneglogpost = gneglogpost+1/2*dlDet
    
        for i in range(ndim):
            gneglogpost = gneglogpost+1/2*1/ndim*n*dsigma2_hat[:,i]/sigma2_hat[i]
  
    return gneglogpost

# Est GP model given data
def GPfitting(Xs,Ys):
    theta0 = np.zeros(2*Xs.shape[1])
    GP = {}
    nugget = 1e-10
    n, d = Xs.shape
    numpara = 2
    
    xoffset = np.mean(Xs,axis=0)
    Xcenter = Xs-xoffset
    xscale = np.mean(np.abs(Xcenter),axis=0)
    Xrescaled =  Xcenter/xscale
    
    if Ys.ndim==1:
        GP["mu"] = np.mean(Ys)
        ys = Ys-GP["mu"]
    else:
        GP["mu"] = np.mean(Ys,axis=1)
        ys = Ys- GP["mu"]
    
    bnd=tuple(map(tuple,np.repeat(np.array([[-0.9999999,0.9999999]]),2*d,0)))
    optm = opt.minimize(
        fun = Internal_neglogpost,
        x0=theta0,
        args=(ys,Xrescaled),
        method='L-BFGS-B',
        jac = Internal_gneglogpost,
        bounds = bnd)
    
    thetaMAP = optm.x
    
    Sigma_t = np.ones((n,n))
    for dimlcv in range(d): # Loop over dimensions
        V =Internal_CorrMatPowerExp(Xrescaled[:,dimlcv], Xrescaled[:,dimlcv],thetaMAP[dimlcv*numpara+np.arange(numpara)])
        Sigma_t = Sigma_t*V
      
    Sigma_t = (1-nugget)*Sigma_t+ np.eye(Sigma_t.shape[0])*nugget 
    Sigma_chol = np.linalg.cholesky(Sigma_t).T
    
    GP["pw"] = inv(Sigma_chol)@ys #trisolve(Sigma_chol,trisolve(Sigma_chol,ys,trans=1))
    GP["sigma2_hat"] = np.sum(ys*GP["pw"],axis=0)/n
    GP["thetaMAP"] = thetaMAP
    GP["Sigma_chol"] = Sigma_chol
    GP["Sigmainv"]= np.linalg.inv(Sigma_t)
    GP["xoffset"] = xoffset
    GP["xscale"] = xscale 
    GP["Xrescaled"]=Xrescaled
    GP["ismultipleoutputs"] = (ys.ndim!=1)
    return GP

#' Predict with GP object
def GPpred(GP, Xp):
  
    Xpcenter = Xp - GP["xoffset"]
    Xprescaled =  Xpcenter/GP["xscale"]
    numpara = 2
    d = Xp.shape[1]
    
    n = GP["Sigma_chol"].shape[0]
    n_p = Xp.shape[0]
    Sigma_pt = np.ones((n_p,n))
    
    for dimlcv in range(d): # Loop over dimensions
        V =Internal_CorrMatPowerExp(Xprescaled[:,dimlcv], GP["Xrescaled"][:,dimlcv],GP["thetaMAP"][dimlcv*numpara+np.arange(numpara)])
        Sigma_pt = Sigma_pt*V
        
    #[:,dimlcv]
    Sigma_p = np.ones((n_p,n_p))
    for dimlcv in range(d):# Loop over dimensions
        V =Internal_CorrMatPowerExp(Xprescaled[:,dimlcv], Xprescaled[:,dimlcv],GP["thetaMAP"][dimlcv*numpara+np.arange(numpara)])
        Sigma_p = Sigma_p*V
        
    
    yhat = np.matmul(Sigma_pt,GP["pw"])+ GP["mu"]
    
    if GP["ismultipleoutputs"]:
        var_scale = np.diag(GP["sigma2_hat"])
    else:
        var_scale = GP["sigma2_hat"]
        
    corrmat_pred = Sigma_p- Sigma_pt@GP["Sigmainv"]@Sigma_pt.T
    
    
    return {"pred":yhat,"corr_mat":corrmat_pred,"var_scale":var_scale}



