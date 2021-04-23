import csv
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import operator

import torch

import sys
sys.path.insert(1, 'E:\Research Projects\Low and High Resolution Calibration\python code')
import GP_function as gp
import nuts as nuts

#%% read data
os.chdir("E:\Research Projects\Low and High Resolution Calibration\EmuAccord")
data_s = pd.read_csv("Accord_Simulation_Results_LL_Airbag_300_Large_Range.csv")
data_f = pd.read_csv("Field_Test_Data_Full_Size_Sedan.csv")
data_l = pd.read_csv("injury_function_data_chest.csv")
data_e = pd.read_csv("Experimental Data - All Sedans.csv")

n_var = 3
n_para = 7
n_res = 1
parameter_default = np.array([1, 0.45, 1, 1, 0.55, 1, 1])
val_nugget = 10 ^ (-4)

n_s = len(data_s)
x_s = data_s.iloc[:, 1 : (n_var + n_para + 1)].to_numpy()
y_s = data_s.iloc[:, n_var + n_para + 1].to_numpy()

n_f = len(data_f)
x_f = data_f.iloc[:, 3 : 6].to_numpy()
a_f = data_f['Age'].to_numpy()
z_f = (data_f.iloc[:, 7] >= 3).to_numpy()

n_l = len(data_l)
a_l = data_l['Age'].to_numpy()
z_l = data_l['Injury Indicator'].to_numpy()
y_l = data_l['ChestD'].to_numpy()
y_l[y_l == 0] = 0.01

n_e = len(data_e)
x_e = np.tile([25.47, 1.75, 35], (1, 1))
y_e = data_e.iloc[:, 8].to_numpy()

s_f = np.ones(n_f)
s_l = np.zeros(n_l)
s_e = np.ones(n_e) * 2

#%% build emulator
GP_s = gp.GPfitting(x_s, np.log(y_s))

#%% define useful functions for sampler
# mean function for logit of injury risk
def func_mean(age, zeta):
    return(-12.597 * np.ones(len(age)) + 0.05861 * age + 1.568 * np.sqrt(np.abs(zeta * 1000)))

# covariance function for logit of injury risk
def corrmat_h(Xa1, Xa2, Xa3, Xb1, Xb2, Xb3):
    R1 = np.exp(-np.abs(np.subtract.outer(Xa1, Xb1) / 40) ** 1.95)
    R2 = np.exp(-np.abs(np.subtract.outer(Xa2, Xb2) / 0.035) ** 1.95)
    
    Xa3[Xa3 == 2] = 0
    Xb3[Xb3 == 2] = 0
    R3 = 0.95 + 0.05 * (np.ones((len(Xa3), len(Xb3))) - np.abs(np.subtract.outer(Xa3, Xb3)))
    
    return(R1 * R2 * R3)

# covariance function for discrepancy
def corrmat_delta(Xa1, Xa2, Xa3, Xb1, Xb2, Xb3):
    R1 = np.exp(-np.abs(np.subtract.outer(Xa1, Xb1) / 0.4) ** 1.95)
    R2 = np.exp(-np.abs(np.subtract.outer(Xa2, Xb2) / 0.6) ** 1.95)
    R3 = np.exp(-np.abs(np.subtract.outer(Xa3, Xb3) / 0.6) ** 1.95)
    
    return(R1 * R2 * R3)

# Gram matrices and inverses
def K_delta(X, inv):
    if inv == False:
        return(corrmat_delta(X[:, 0], X[:, 1], X[:, 2], X[:, 0], X[:, 1], X[:, 2]))
    else:
        return np.linalg.inv(corrmat_delta(X[:, 0], X[:, 1], X[:, 2], X[:, 0], X[:, 1], X[:, 2]))
    
def K_delta_cross(Xa, Xb):
    return(corrmat_delta(Xa[:, 0], Xa[:, 1], Xa[:, 2], Xb[:, 0], Xb[:, 1], Xb[:, 2]))

def K_h(age, zeta, sign, inv):
    if inv == False:
        return(corrmat_h(age, zeta, sign, age, zeta, sign))
    else:
        return(np.linalg.inv(corrmat_h(age, zeta, sign, age, zeta, sign)))

def K_h_cross(age1, zeta1, sign1, age2, zeta2, sign2):
    return(corrmat_h(age1, zeta1, sign1, age2, zeta2, sign2))

# predictive distribution of GPs
# mean and covariance for eta_e
def eta_hat_e(x, theta):
    GP_pred_e = gp.GPpred(GP_s, np.concatenate((x, np.tile(theta, (1, 1))), axis = 1))
    
    return(np.exp(GP_pred_e['pred']))

def K_hat_eta_e(x, theta): # no need to inverse because it is univariate
    GP_pred_e = gp.GPpred(GP_s, np.concatenate((x, np.tile(theta, (1, 1))), axis = 1))
    
    return(np.exp(GP_pred_e['pred']) ** 2 * GP_pred_e['corr_mat'])

# mean and covariance for eta_f
def eta_hat_f(x, theta):
    return(np.exp(gp.GPpred(GP_s, np.column_stack((x, np.tile(theta, (n_f, 1)))))['pred']))

def K_hat_eta_f(x, theta, inv):
    GP_pred_f = gp.GPpred(GP_s, np.concatenate((x, np.tile(theta, (n_f, 1))), axis = 1))
    
    if inv == False:
        return(np.matmul(np.matmul(np.diag(np.exp(GP_pred_f['pred'])), GP_pred_f['corr_mat']), np.diag(np.exp(GP_pred_f['pred']))))
    else:
        return(np.linalg.inv(np.matmul(np.matmul(np.diag(np.exp(GP_pred_f['pred'])), GP_pred_f['corr_mat']), np.diag(np.exp(GP_pred_f['pred'])))))
        
# mean and covariance for delta_e
def delta_hat_e(x_e, x_f, delta_f):
    return(np.matmul(np.matmul(K_delta_cross(x_e, x_f), K_delta(x_f, inv = True)), delta_f))

def K_hat_delta_e(x_e, x_f): # no need to inverse because it is univariate
    return(np.matmul(np.matmul(K_delta_cross(x_e, x_f), K_delta(x_f, inv = True)), K_delta_cross(x_e, x_f).transpose()))

# mean and covariance for h_f
def h_hat_f(a_f, zeta_f, s_f, a_l, zeta_l, s_l, h_l):
    return(np.matmul(np.matmul(K_h_cross(a_f, zeta_f, s_f, a_l, zeta_l, s_l), K_h(a_l, zeta_l, s_l, inv = True)), h_l))

def K_hat_h_f(a_f, zeta_f, s_f, a_l, zeta_l, s_l, inv):
    if inv == False:
        return(np.matmul(np.matmul(K_h_cross(a_f, zeta_f, s_f, a_l, zeta_l, s_l), K_h(a_l, zeta_l, s_l, inv = True)), K_h_cross(a_f, zeta_f, s_f, a_l, zeta_l, s_l).transpose()))
    else:
        return(np.linalg.inv(np.matmul(np.matmul(K_h_cross(a_f, zeta_f, s_f, a_l, zeta_l, s_l), K_h(a_l, zeta_l, s_l, inv = True)), K_h_cross(a_f, zeta_f, s_f, a_l, zeta_l, s_l).transpose())))

# full conditional log-likelihood functions    
# lambda_delta with non-informative prior
def log_prob_fn_lambda_delta(lambda_delta):
    log_likelihood1 = n_f / 2 * np.log(lambda_delta) - lambda_delta / 2 * np.matmul(np.matmul(sample_delta_f[-1, :].transpose(), K_delta(x_f, True)), sample_delta_f[-1, :])
    
    lambda_mix_inv = 1 / n_e / sample_lambda_e[-1] + K_hat_delta_e(x_e, x_f) / lambda_delta + K_hat_eta_e(x_e, sample_theta[-1, :]) * GP_s['sigma2_hat']
    log_likelihood2 = -1 / 2 * np.log(lambda_mix_inv) - (np.mean(y_e) - eta_hat_e(x_e, sample_theta[-1, :]) - delta_hat_e(x_e, x_f, sample_delta_f[-1, :])) ** 2 / 2 / lambda_mix_inv
    
    prior = -np.log(lambda_delta)
    
    return((log_likelihood1 + log_likelihood2 + prior)[0])

# theta with Gaussian prior # omit likelihood3 under modularization
def log_prob_fn_theta(theta):
    log_likelihood1 = -n_f / 2 * np.linalg.slogdet(K_hat_eta_f(x_f, theta, inv = False))[1] - 1 / 2 / GP_s['sigma2_hat'] * np.matmul(np.matmul((sample_eta_f[-1, :] - eta_hat_f(x_f, theta)).transpose(), K_hat_eta_f(x_f, theta, inv = True)), sample_eta_f[-1, :] - eta_hat_f(x_f, theta))
    
    lambda_mix_inv = 1 / n_e / sample_lambda_e[-1] + K_hat_delta_e(x_e, x_f) / sample_lambda_delta[-1] + K_hat_eta_e(x_e, theta) * GP_s['sigma2_hat']    
    log_likelihood2 = -1 / 2 * np.log(lambda_mix_inv) - (np.mean(y_e) - eta_hat_e(x_e, theta) - delta_hat_e(x_e, x_f, sample_delta_f[-1, :])) ** 2 / 2 / lambda_mix_inv
    
#    h_hat_f_temp = h_hat_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, sample_h_l[-1, :])
#    K_hat_h_f_temp = K_hat_h_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, inv = False)
#    K_hat_h_f_inv_temp = np.linalg.inv(K_hat_h_f_temp + np.diag(np.ones(n_f) * val_nugget))
#    log_likelihood3 = -n_f / 2 * np.linalg.slogdet(K_hat_h_f_temp)[1] - sample_lambda_h[-1] / 2 * np.matmul(np.matmul((sample_h_f[-1, :] - h_hat_f_temp).transpose(), K_hat_h_f_inv_temp), sample_h_f[-1, :] - h_hat_f_temp)
    
    prior = -np.sum(50 * (theta - parameter_default) ** 2)
    
    return(log_likelihood1 + log_likelihood2 + prior)
    
# h_l with Gaussian prior
def log_prob_fn_h_l(h_l):
    log_likelihood1 = np.sum(z_l * np.log(np.exp(h_l) / (np.ones((1, n_l)) + np.exp(h_l))) + (1 - z_l) * np.log(1 / (np.ones((1, n_l)) + np.exp(h_l))))
    log_likelihood2 = -sample_lambda_h[-1] / 2 * np.matmul(np.matmul((h_l - func_mean(a_l, sample_zeta_l[-1, :])).transpose(), K_h(a_l, sample_zeta_l[-1, :], s_l, inv = True)), h_l - func_mean(a_l, sample_zeta_l[-1, :]))
    
    h_hat_f_temp = h_hat_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, h_l)
    K_hat_h_f_temp = K_hat_h_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, inv = False)
    K_hat_h_f_inv_temp = np.linalg.inv(K_hat_h_f_temp + np.diag(np.ones(n_f) * val_nugget))
    log_likelihood3 = -sample_lambda_h[-1] / 2 * np.matmul(np.matmul((sample_h_f[-1, :] - h_hat_f_temp).transpose(), K_hat_h_f_inv_temp), sample_h_f[-1, :] - h_hat_f_temp)
   
    prior = -np.sum(10 * h_l ** 2)
    
    return(log_likelihood1 + log_likelihood2 + log_likelihood3 + prior)

# h_f with Gaussian prior
def log_prob_fn_h_f(h_f):
    log_likelihood1 = np.sum(z_f * np.log(np.exp(h_f) / (np.ones((1, n_f)) + np.exp(h_f))) + (1 - z_f) * np.log(1 / (np.ones((1, n_f)) + np.exp(h_f))))

    h_hat_f_temp = h_hat_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, sample_h_l[-1, :])
    K_hat_h_f_temp = K_hat_h_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, inv = False)
    K_hat_h_f_inv_temp = np.linalg.inv(K_hat_h_f_temp + np.diag(np.ones(n_f) * val_nugget))
    log_likelihood2 = -sample_lambda_h[-1] / 2 * np.matmul(np.matmul((h_f - h_hat_f_temp).transpose(), K_hat_h_f_inv_temp), h_f - h_hat_f_temp)
    
    prior = -np.sum(10 * h_f ** 2)
    
    return(log_likelihood1 + log_likelihood2 + prior)

# eta_f with Gaussian prior
def log_prob_fn_eta_f(eta_f):
    h_hat_f_temp = h_hat_f(a_f, eta_f + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, sample_h_l[-1, :])
    K_hat_h_f_temp = K_hat_h_f(a_f, eta_f + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, inv = False)
    K_hat_h_f_inv_temp = np.linalg.inv(K_hat_h_f_temp + np.diag(np.ones(n_f) * val_nugget))
    log_likelihood1 = -sample_lambda_h[-1] / 2 * np.matmul(np.matmul((sample_h_f[-1, :] - h_hat_f_temp).transpose(), K_hat_h_f_inv_temp), sample_h_f[-1, :] - h_hat_f_temp)
    
    log_likelihood2 = -n_f / 2 * np.linalg.slogdet(K_hat_eta_f(x_f, sample_theta[-1, :], inv = False))[1] - 1 / 2 / GP_s['sigma2_hat'] * np.matmul(np.matmul((eta_f - eta_hat_f(x_f, sample_theta[-1, :])).transpose(), K_hat_eta_f(x_f, sample_theta[-1, :], inv = True)), eta_f - eta_hat_f(x_f, sample_theta[-1, :]))
    
    prior = -np.sum(1000 * eta_f ** 2)
    
    return(log_likelihood1 + log_likelihood2 + prior)

# delta_f with Gaussian prior
def log_prob_fn_delta_f(delta_f):
    h_hat_f_temp = h_hat_f(a_f, sample_eta_f[-1, :] + delta_f, s_f, a_l, sample_zeta_l[-1, :], s_l, sample_h_l[-1, :])
    K_hat_h_f_temp = K_hat_h_f(a_f, sample_eta_f[-1, :] + delta_f, s_f, a_l, sample_zeta_l[-1, :], s_l, inv = False)
    K_hat_h_f_inv_temp = np.linalg.inv(K_hat_h_f_temp + np.diag(np.ones(n_f) * val_nugget))
    log_likelihood1 = -sample_lambda_h[-1] / 2 * np.matmul(np.matmul((sample_h_f[-1, :] - h_hat_f_temp).transpose(), K_hat_h_f_inv_temp), sample_h_f[-1, :] - h_hat_f_temp)
    
    log_likelihood2 = - sample_lambda_delta[-1] / 2 * np.matmul(np.matmul(delta_f.transpose(), K_delta(x_f, True)), delta_f)

    lambda_mix_inv = 1 / n_e / sample_lambda_e[-1] + K_hat_delta_e(x_e, x_f) / sample_lambda_delta[-1] + K_hat_eta_e(x_e, sample_theta[-1, :]) * GP_s['sigma2_hat']    
    log_likelihood3 = - (np.mean(y_e) - eta_hat_e(x_e, sample_theta[-1, :]) - delta_hat_e(x_e, x_f, delta_f)) ** 2 / 2 / lambda_mix_inv

    prior = -np.sum(1000 * delta_f ** 2)
    
    return(log_likelihood1 + log_likelihood2 +log_likelihood3 + prior)
    
# zeta_l with Gaussian prior
def log_prob_fn_zeta_l(zeta_l):
    log_likelihood1 = -sample_lambda_l[-1] / 2 * np.sum((y_l - zeta_l) ** 2)

    h_hat_f_temp = h_hat_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, zeta_l, s_l, sample_h_l[-1, :])
    K_hat_h_f_temp = K_hat_h_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, zeta_l, s_l, inv = False)
    K_hat_h_f_inv_temp = np.linalg.inv(K_hat_h_f_temp + np.diag(np.ones(n_f) * val_nugget))
    log_likelihood2 = -sample_lambda_h[-1] / 2 * np.matmul(np.matmul((sample_h_f[-1, :] - h_hat_f_temp).transpose(), K_hat_h_f_inv_temp), sample_h_f[-1, :] - h_hat_f_temp)
    
    prior = -np.sum(1000 * zeta_l ** 2)
    
    return(log_likelihood1 + log_likelihood2 + prior)

# define samplers
sampler_lambda_delta = nuts.emcee_nuts.NUTSSampler(1, log_prob_fn_lambda_delta)
sampler_theta = nuts.emcee_nuts.NUTSSampler(7, log_prob_fn_theta)
sampler_h_l = nuts.emcee_nuts.NUTSSampler(n_l, log_prob_fn_h_l)
sampler_h_f = nuts.emcee_nuts.NUTSSampler(n_f, log_prob_fn_h_f)
sampler_eta_f = nuts.emcee_nuts.NUTSSampler(n_f, log_prob_fn_eta_f)
sampler_delta_f = nuts.emcee_nuts.NUTSSampler(n_f, log_prob_fn_delta_f)
sampler_zeta_l = nuts.emcee_nuts.NUTSSampler(n_l, log_prob_fn_zeta_l)

#%% NUTS sampler
# Allow external control of sampling to reduce test runtimes.
delta_hmc = 0.2
num_results = 1000
num_burn_in_sub = 5
num_burn_in = 100

# initial states
sample_lambda_l = [100.0]
sample_lambda_h = [100.0]
sample_lambda_delta = [100.0]
sample_lambda_e = [100.0]
sample_theta = np.tile(parameter_default, (1, 1))

sample_zeta_l = np.expand_dims(y_l, axis = 0) + np.random.normal(0, 0.1, n_l)
sample_h_l = np.zeros((1, n_l))
sample_h_f = np.zeros((1, n_f))
sample_eta_f = np.exp(np.tile(gp.GPpred(GP_s, np.column_stack((x_f, np.tile(parameter_default, (n_f, 1)))))['pred'], (1, 1)))
sample_delta_f = np.zeros((1, n_f))

for iter_ind in range(num_results):
    # sample lambda_l from Gamma distribution with non-informative prior
    lambda_l_new = np.random.gamma(n_l / 2, 2 / np.sum((y_l - sample_zeta_l[-1, :]) ** 2), 1)
    sample_lambda_l.append(lambda_l_new[0])
    
    # sample lambda_h from Gamma distribution with non-informative prior
    mat_1 = np.matmul(np.matmul((sample_h_l[-1, :] - func_mean(a_l, sample_zeta_l[-1, :])).transpose(), K_h(a_l, sample_zeta_l[-1, :], s_l, inv = True)), sample_h_l[-1, :] - func_mean(a_l, sample_zeta_l[-1, :]))
    
    h_hat_f_temp = h_hat_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, sample_h_l[-1, :])
    K_hat_h_f_inv_temp = K_hat_h_f(a_f, sample_eta_f[-1, :] + sample_delta_f[-1, :], s_f, a_l, sample_zeta_l[-1, :], s_l, inv = True)
    mat_2 = np.matmul(np.matmul((sample_h_f[-1, :] - h_hat_f_temp).transpose(), K_hat_h_f_inv_temp), sample_h_f[-1, :] - h_hat_f_temp)
    
    lambda_h_new = np.random.gamma(n_l / 2 + n_f / 2, 2 / (mat_1 + mat_2), 1)
    sample_lambda_h.append(lambda_h_new[0])
    
    # sample lambda_delta from one-step NUTS
    samples = sampler_lambda_delta.run_mcmc(np.expand_dims(sample_lambda_delta[-1], 0), 1, num_burn_in_sub, delta_hmc)
    sample_lambda_delta.append(np.squeeze(samples))

    # sample lambda_e from proposal distribution
    # temp distribution for lambda_e
    while True:
        dist_lambda_e_temp = np.random.gamma(3 / 2, 2 / (np.mean(y_e) - eta_hat_e(x_e, sample_theta[-1, :]) - delta_hat_e(x_e, x_f, sample_delta_f[-1, :])) ** 2, 1)
        temp_val = 1 / dist_lambda_e_temp - K_hat_delta_e(x_e, x_f) / sample_lambda_delta[-1] - K_hat_eta_e(x_e, sample_theta[-1, :]) * GP_s['sigma2_hat']
        if temp_val > 0:
            lambda_e_new = 1 / temp_val / n_e
            break

    sample_lambda_e.append(lambda_e_new)

    # sample theta from one-step NUTS
    samples = sampler_theta.run_mcmc(sample_theta[-1, :], 1, num_burn_in_sub, delta_hmc)
    sample_theta = np.vstack((sample_theta, samples))
    
    # sample h_l from one-step NUTS
    samples = sampler_h_l.run_mcmc(sample_h_l[-1, :], 1, num_burn_in_sub, delta_hmc)
    sample_h_l = np.vstack((sample_h_l, samples))
    
    # sample h_f from one-step NUTS
    samples = sampler_h_f.run_mcmc(sample_h_f[-1, :], 1, num_burn_in_sub, delta_hmc)
    sample_h_f = np.vstack((sample_h_f, samples))

    # sample eta_f from one-step NUTS
    samples = sampler_eta_f.run_mcmc(sample_eta_f[-1, :], 1, num_burn_in_sub, delta_hmc)
    sample_eta_f = np.vstack((sample_eta_f, samples))

    # sample delta_f from one-step NUTS
    samples = sampler_delta_f.run_mcmc(sample_delta_f[-1, :], 1, num_burn_in_sub, delta_hmc)
    sample_delta_f = np.vstack((sample_delta_f, samples))

    # sample zeta_l from one-step NUTS
    samples = sampler_zeta_l.run_mcmc(sample_zeta_l[-1, :], 1, num_burn_in_sub, delta_hmc)
    sample_zeta_l = np.vstack((sample_zeta_l, samples))