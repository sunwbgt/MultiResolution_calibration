import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch as tch

import sys
sys.path.insert(1, 'E:\Research Projects\Low and High Resolution Calibration\python code')
import GP_function_torch as gpt
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

#%% read data
os.chdir("E:\Research Projects\Low and High Resolution Calibration\EmuAccord")
data_s = pd.read_csv("Accord_Simulation_Results_LL_Airbag_300_Large_Range.csv")
data_f = pd.read_csv("Field_Test_Data_Full_Size_Sedan.csv")
data_l = pd.read_csv("injury_function_data_chest.csv")
data_e = pd.read_csv("Experimental Data - All Sedans.csv")

n_var = 3
n_para = 7
n_res = 1
parameter_default = tch.tensor([1, 0.45, 1, 1, 0.55, 1, 1])
val_nugget = 10 ** (-4)

n_s = len(data_s)
x_s = tch.tensor(data_s.iloc[:, 1 : (n_var + n_para + 1)].values)
y_s = tch.tensor(data_s.iloc[:, n_var + n_para + 1])

n_f = len(data_f)
x_f = tch.tensor(data_f.iloc[:, 3 : 6].values)
a_f = tch.tensor(data_f['Age'].values)
z_f = (tch.tensor(data_f.iloc[:, 7].values) >= 3).long()

n_l = len(data_l)
a_l = tch.tensor(data_l['Age'].values)
z_l = tch.tensor(data_l['Injury Indicator'].values)
y_l = tch.tensor(data_l['ChestD'].values)
y_l[y_l == 0] = 0.01

n_e = len(data_e)
x_e = tch.tensor([25.47, 1.75, 35]).repeat(1, 1)
y_e = tch.tensor(data_e.iloc[:, 8].values)

s_f = tch.ones(n_f)
s_l = tch.zeros(n_l)
s_e = tch.ones(n_e) * 2

#%% train the emulator
theta0 = tch.cat((tch.squeeze(tch.log(tch.std(x_s, 0))), tch.tensor(-1)[None]))
theta = theta0.clone().detach()
theta.requires_grad=True

niter = 100

optim = tch.optim.SGD([theta], lr = 1e-2, momentum=0.8, nesterov=True)
for _ in tqdm(range(niter)):
    optim.zero_grad()
    
    loss = gpt.Internal_neglogpost(theta,y_s,x_s,theta0)
    loss.backward()
    optim.step()

theta.requires_grad=False
gpinfo = gpt.gpbuild(theta,y_s,x_s)

x_s.requires_grad=True
gpt.gppred(x_s,gpinfo).sum().backward()
print(x_s.grad)

#%% define useful functions for sampler
# mean function for logit of injury risk
def func_mean(age, zeta):
    return(-12.5972 * tch.ones(len(age)) + 0.058614 * age + 26.90118 * (zeta / 0.2486))

# covariance function for logit of injury risk
def corrmat_h(Xa1, Xa2, Xa3, Xb1, Xb2, Xb3):
    
    R1 = tch.exp(-((Xa1.reshape(-1, 1) - Xb1) / 40) ** 2)
        
    R2 = tch.exp(-((Xa2.reshape(-1, 1) - Xb2) / 0.035) ** 2)
        
    Xa3[Xa3 == 2] = 0
    Xb3[Xb3 == 2] = 0
    R3 = 0.95 + 0.05 * (tch.ones((len(Xa3), len(Xb3))) - tch.abs(Xa3.reshape(-1, 1) - Xb3))
    
    return(R1 * R2 * R3)

# covariance function for discrepancy
def corrmat_delta(Xa1, Xa2, Xa3, Xb1, Xb2, Xb3):
    R1 = tch.exp(-((Xa1.reshape(-1, 1) - Xb1) / 0.4) ** 2)
    R2 = tch.exp(-((Xa2.reshape(-1, 1) - Xb2) / 0.6) ** 2)
    R3 = tch.exp(-((Xa3.reshape(-1, 1) - Xb3) / 0.6) ** 2)
    
    return(R1 * R2 * R3)

# Gram matrices and inverses
def K_delta(X, inv):
    if inv == False:
        return corrmat_delta(X[:, 0], X[:, 1], X[:, 2], X[:, 0], X[:, 1], X[:, 2])
    else:
        return tch.inverse(corrmat_delta(X[:, 0], X[:, 1], X[:, 2], X[:, 0], X[:, 1], X[:, 2]) + tch.diag(tch.ones(len(X[:, 0]))) * val_nugget)
    
def K_delta_cross(Xa, Xb):
    return(corrmat_delta(Xa[:, 0], Xa[:, 1], Xa[:, 2], Xb[:, 0], Xb[:, 1], Xb[:, 2]))

def K_h(age, zeta, sign, inv):
    if inv == False:
        return corrmat_h(age, zeta, sign, age, zeta, sign)
    else:
        return tch.inverse(corrmat_h(age, zeta, sign, age, zeta, sign) + tch.diag(tch.ones(len(age))) * val_nugget)

def K_h_cross(age1, zeta1, sign1, age2, zeta2, sign2):
    return(corrmat_h(age1, zeta1, sign1, age2, zeta2, sign2))

#%% calculate the joint likelihood
lambda_eta = 1 / gpinfo['sigma2_hat']
# initial states
lambda_l = tch.tensor(1000.0)
lambda_h = tch.tensor(1.0)
lambda_delta = tch.tensor(1000.0)
lambda_e = tch.tensor(1000.0)
theta = parameter_default.clone().detach()

zeta_l = tch.abs(y_l + tch.normal(0.0, 0.01, (1, n_l)).squeeze(0))
h_l = func_mean(a_l, y_l)

x_f_s = tch.hstack((x_f, parameter_default.repeat(n_f, 1)))
eta_f = gpt.gppred(x_f_s, gpinfo).T.squeeze(0).detach()
delta_f = tch.ones(n_f) * 0.05
h_f = func_mean(a_f, eta_f + delta_f).detach()

# define log-likelihood functions
def log_prob_fn_joint(lambda_h, lambda_l, lambda_delta, lambda_e, theta, h_l, zeta_l, h_f, eta_f, delta_f):
    
    prior_parameter = tch.log(lambda_h) + tch.log(lambda_l) + tch.log(lambda_delta) + tch.log(lambda_e) - tch.sum(50 * (theta - parameter_default) ** 2)
    prior_random_effect = -tch.sum(100 * zeta_l ** 2) - tch.sum(0.1 * h_l ** 2) - tch.sum(0.1 * h_f ** 2) - tch.sum(100 * eta_f ** 2) - tch.sum(100 * delta_f ** 2)
    prior = prior_parameter + prior_random_effect
    
    return(log_prob_fn_lab(lambda_h, lambda_l, h_l, zeta_l) + log_prob_fn_field(lambda_h, lambda_delta, h_l, theta, zeta_l, h_f, eta_f, delta_f) + log_prob_fn_exp(lambda_delta, lambda_e, theta, delta_f) + prior)

# log-likelihood for lab data
def log_prob_fn_lab(lambda_h, lambda_l, h_l, zeta_l):
    
    # p(z_l | h_l)
    log_likelihood1 = tch.sum(z_l * tch.log(tch.exp(h_l) / (tch.ones((1, n_l)) + tch.exp(h_l))) + (1 - z_l) * tch.log(1 / (tch.ones((1, n_l)) + tch.exp(h_l))))
    
    # p(h_l | zeta_l, a_l)
    log_likelihood2 = n_l / 2 * tch.log(lambda_h) - 1 / 2 * tch.logdet(K_h(a_l, zeta_l, s_l, inv = False) + tch.diag(tch.ones(n_l) * val_nugget)) - lambda_h / 2 * (h_l - func_mean(a_l, zeta_l)) @ K_h(a_l, zeta_l, s_l, inv = True) @ (h_l - func_mean(a_l, zeta_l)).T
 
    # p(zeta_l | y_l)
    log_likelihood3 = n_l / 2 * tch.log(lambda_l) - lambda_l / 2 * tch.sum((y_l - zeta_l) ** 2)
    
    return(log_likelihood1 + log_likelihood2 + log_likelihood3)

# log-likelihood for field data
def log_prob_fn_field(lambda_h, lambda_delta, h_l, theta, zeta_l, h_f, eta_f, delta_f):
    
    # p(z_f | h_f)
    log_likelihood1 = tch.sum(z_f * tch.log(tch.exp(h_f) / (tch.ones((1, n_f)) + tch.exp(h_f))) + (1 - z_f) * tch.log(1 / (tch.ones((1, n_f)) + tch.exp(h_f))))
    
    # p(h_f | eta_f, delta_f, a_f)
    h_hat_f = (K_h_cross(a_f, eta_f + delta_f, s_f, a_l, zeta_l, s_l) @ K_h(a_l, zeta_l, s_l, inv = True) @ h_l.T).T
    K_hat_h_f = K_h_cross(a_f, eta_f + delta_f, s_f, a_l, zeta_l, s_l) @ K_h(a_l, zeta_l, s_l, inv = True) @ K_h_cross(a_f, eta_f + delta_f, s_f, a_l, zeta_l, s_l).T
    K_hat_h_f_inv = tch.inverse(K_hat_h_f + tch.diag(tch.ones(n_f) * val_nugget))
    
    log_likelihood2 = n_f / 2 * tch.log(lambda_h) - 1 / 2 * tch.logdet(K_hat_h_f  + tch.diag(tch.ones(n_f) * val_nugget)) - lambda_h / 2 * (h_f - h_hat_f) @ K_hat_h_f_inv @ (h_f - h_hat_f).T
    
    # p(eta_f | x_f, theta, y_eta)    
    x_f_new = tch.hstack((x_f, theta.repeat(n_f, 1)))
    eta_hat_f = gpt.gppred(x_f_new, gpinfo).T
    K_hat_eta_f = gpt.covpred(x_f_new, gpinfo)
    K_hat_eta_f_inv = tch.inverse(K_hat_eta_f  + tch.diag(tch.ones(n_f) * val_nugget))
    
    log_likelihood3 = -1 / 2 * tch.logdet(K_hat_eta_f + tch.diag(tch.ones(n_f) * val_nugget)) - lambda_eta / 2 * (eta_f - eta_hat_f) @ K_hat_eta_f_inv @ (eta_f - eta_hat_f).T

    # p(delta_f | x_f)
    log_likelihood4 = n_f / 2 * tch.log(lambda_delta) - 1 / 2 * tch.logdet(K_delta(x_f, inv = False) + tch.diag(tch.ones(n_f) * val_nugget)) - lambda_delta / 2 * delta_f @ K_delta(x_f, inv = True) @ delta_f.T
    
    return(log_likelihood1 + log_likelihood2 + log_likelihood3 + log_likelihood4)
    
# log-likelihood for experimental data
def log_prob_fn_exp(lambda_delta, lambda_e, theta, delta_f):
    
    # p(y_e | x_e, theta)
    delta_hat_e = K_delta_cross(x_e, x_f) @ K_delta(x_f, inv = True) @ delta_f.T
    K_hat_delta_e = K_delta_cross(x_e, x_f) @ K_delta(x_f, inv = True) @ K_delta_cross(x_e, x_f).T
    
    x_e_new = tch.cat((x_e, theta.unsqueeze(0)), 1)
    eta_hat_e = gpt.gppred(x_e_new, gpinfo)
    K_hat_eta_e = gpt.covpred(x_e_new, gpinfo)
    
    lambda_mix_inv = 1 / n_e / lambda_e + K_hat_delta_e / lambda_delta + K_hat_eta_e / lambda_eta   
    log_likelihood = -1 / 2 * tch.log(lambda_mix_inv) - (tch.mean(y_e) - eta_hat_e - delta_hat_e) ** 2 / 2 / lambda_mix_inv
    
    return(log_likelihood)
    
    
#%% conditional optimization zeta_l 
zeta_l.requires_grad=True
niter = 2000
optimizer = tch.optim.Adam([zeta_l], lr = 1e-3)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    
    negloglik = -log_prob_fn_joint(lambda_h, lambda_l,
                                    lambda_delta, lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

plt.plot(zeta_l.detach().numpy())
plt.plot(y_l)
    
#%% conditional optimization h_l
h_l.requires_grad=True
niter = 2000
optimizer = tch.optim.Adam([h_l], lr = 1e-1)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(lambda_h, lambda_l,
                                    lambda_delta, lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

sort_idx = np.argsort(z_l)
plt.plot(np.exp(h_l.detach().numpy()[sort_idx]) / (1 + np.exp(h_l.detach().numpy()[sort_idx])))
plt.plot(z_l[sort_idx])

p_l = np.exp(h_l.detach().numpy()) / (1 + np.exp(h_l.detach().numpy()))
print(confusion_matrix(p_l > 0.5, z_l == 1))

#%% conditional optimization eta_f
eta_f.requires_grad=True
niter = 1000
optimizer = tch.optim.Adam([eta_f], lr = 1e-3)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(lambda_h, lambda_l,
                                    lambda_delta, lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

plt.plot(eta_f.detach().numpy())
plt.plot(gpt.gppred(x_f_s, gpinfo).T.squeeze(0).detach())

#%% conditional optimization h_f
h_f.requires_grad=True
niter = 2000
optimizer = tch.optim.Adam([h_f], lr = 1e-1)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(lambda_h, lambda_l,
                                    lambda_delta, lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

sort_idx = np.argsort(z_f)
plt.plot(np.exp(h_f.detach().numpy()[sort_idx]) / (1 + np.exp(h_f.detach().numpy()[sort_idx])))
plt.plot(z_f[sort_idx])

p_f = np.exp(h_f.detach().numpy()) / (1 + np.exp(h_f.detach().numpy()))
print(confusion_matrix(p_f > 0.5, z_f == 1))

#%% conditional optimization delta_f
delta_f = tch.zeros(n_f)
delta_f.requires_grad=True
niter = 2000
optimizer = tch.optim.Adam([delta_f], lr = 1e-3)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(lambda_h, lambda_l,
                                    lambda_delta, lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

plt.plot(delta_f.detach().numpy())

#%% conditional optimization lambda_l
log_lambda_l = tch.tensor(1.0)
log_lambda_l.requires_grad=True
niter = 500
optimizer = tch.optim.Adam([log_lambda_l], lr = 0.1)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(lambda_h, tch.exp(log_lambda_l),
                                    lambda_delta, lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

print(tch.exp(log_lambda_l))

#%% conditional optimization lambda_h
log_lambda_h = tch.tensor(1.0)
log_lambda_h.requires_grad=True
niter = 500
optimizer = tch.optim.Adam([log_lambda_h], lr = 0.1)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(tch.exp(log_lambda_h), tch.exp(log_lambda_l),
                                    lambda_delta, lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

print(tch.exp(log_lambda_h))

#%% conditional optimization lambda_delta
log_lambda_delta = tch.tensor(1.0)
log_lambda_delta.requires_grad=True
niter = 500
optimizer = tch.optim.Adam([log_lambda_delta], lr = 0.1)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(tch.exp(log_lambda_h), tch.exp(log_lambda_l),
                                    tch.exp(log_lambda_delta), lambda_e,
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

print(tch.exp(log_lambda_delta))

#%% conditional optimization lambda_e
log_lambda_e = tch.tensor(1.0)
log_lambda_e.requires_grad=True
niter = 200
optimizer = tch.optim.Adam([log_lambda_e], lr = 0.1)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(tch.exp(log_lambda_h), tch.exp(log_lambda_l),
                                    tch.exp(log_lambda_delta), tch.exp(log_lambda_e),
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    with tch.no_grad():
        log_lambda_e.clamp_(0.1, 12.0)
        
    iters += 1 

print(tch.exp(log_lambda_e))

#%% conditional optimization theta
niter = 200
theta.requires_grad=True
optimizer = tch.optim.Adam([theta], lr = 0.1)

iters = 0
running_loss = 0.0
for _ in tqdm(range(niter)):
    optimizer.zero_grad()
    negloglik = -log_prob_fn_joint(tch.exp(log_lambda_h), tch.exp(log_lambda_l),
                                    tch.exp(log_lambda_delta), tch.exp(log_lambda_e),
                                    theta, h_l, zeta_l,
                                    h_f, eta_f, delta_f)
    negloglik.backward(retain_graph = True)
    optimizer.step()
        
    running_loss += negloglik
    
    if iters % 50 == 49:
        print('log_likelihood: %.3f' %(-running_loss / 50))
        running_loss = 0.0
    
    iters += 1 

print(theta)
