import csv
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as tch
import pandas as pd
import operator

import torch as tch

import sys
import GP_function as gp
import GP_function_torch as gpt
import nuts as nuts

#%% read data
data_s = pd.read_csv("Accord_Simulation_Results_LL_Airbag_300_Large_Range.csv")
data_f = pd.read_csv("Field_Test_Data_Full_Size_Sedan.csv")
data_l = pd.read_csv("injury_function_data_chest.csv")
data_e = pd.read_csv("Experimental Data - All Sedans.csv")

n_var = 3
n_para = 7
n_res = 1
parameter_default = tch.tensor([1, 0.45, 1, 1, 0.55, 1, 1])
val_nugget = 10 ^ (-4)

n_s = len(data_s)
x_s = data_s.iloc[:, 1 : (n_var + n_para + 1)].to_numpy()
y_s = tch.tensor(data_s.iloc[:, n_var + n_para + 1])

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
x_e = tch.tile(tch.tensor([25.47, 1.75, 35]), (1, 1))
y_e = data_e.iloc[:, 8].to_numpy()

s_f = tch.ones(n_f)
s_l = tch.zeros(n_l)
s_e = tch.ones(n_e) * 2

tch_x_s = tch.from_numpy(x_s)
theta0 = tch.cat( (tch.squeeze(tch.log(tch.std(tch_x_s, 0))), tch.tensor(-1)[None]))
theta = theta0.clone().detach()
theta.requires_grad=True
import GP_function_torch as gpt

y_st = tch.log(y_s)

optim = tch.optim.SGD([theta], lr = 1e-2, momentum=0.8, nesterov=True)
for k in range(0,100):
    optim.zero_grad()
    gpt.Internal_neglogpost(theta,y_st,tch_x_s,theta0).backward()
    optim.step()

theta.requires_grad=False
gpinfo = gpt.gpbuild(theta,y_st,tch_x_s)


tch_x_s.requires_grad=True
gpt.gppred(tch_x_s,gpinfo).sum().backward()
print(tch_x_s.grad)
