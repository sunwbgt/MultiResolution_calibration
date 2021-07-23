import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.utils.data as Data
import torch as tch

import sys
sys.path.insert(1, 'E:\Research Projects\Low and High Resolution Calibration\python code')
import GP_function_torch as gpt

#%% read data
os.chdir("E:\Research Projects\Low and High Resolution Calibration\EmuAccord")
data_s = pd.read_csv("Accord_Simulation_Results_1000.csv")
data_f = pd.read_csv("cds_simplified.csv")
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
x_f = tch.tensor(data_f.iloc[:, 0 : 3].values)
a_f = tch.tensor(data_f['Age'].values)
z_f = (tch.tensor(data_f.iloc[:, 4].values) >= 3).long()

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

#%% train the gp emulator
pred_gp_cross = np.zeros(n_s)
mse_gp = np.zeros(10)

# 10-fold validation
for k in range(10):
    theta0 = tch.cat((tch.squeeze(tch.log(tch.std(x_s, 0))), tch.tensor(-1)[None]))
    theta = theta0.clone().detach()
    theta.requires_grad=True
    niter = 100
    optim = tch.optim.SGD([theta], lr = 1e-2, momentum=0.8, nesterov=True)


    leaveoutind = k + 10 * np.arange(98)
    allind = np.arange(n_s)
    trainind = np.delete(allind, leaveoutind)
    
    x_train = x_s[trainind, :]
    y_train = y_s[trainind]
    
    x_test = x_s[leaveoutind, :]
    y_test = y_s[leaveoutind]   

    loss_func = tch.nn.MSELoss()

    for i in range(niter):
        optim.zero_grad()
        
        loss = gpt.Internal_neglogpost(theta,y_train,x_train,theta0)
        loss.backward()
        optim.step()
    
    theta.requires_grad=False
    gpinfo = gpt.gpbuild(theta,y_train,x_train)

    pred_gp_cross[leaveoutind] = gpt.gppred(x_test,gpinfo).squeeze().numpy()
    mse_gp[k] = loss_func(gpt.gppred(x_test,gpinfo).squeeze(), y_test)
    
plt.scatter(pred_gp_cross, y_s)
plt.ylim((0, 0.08))
plt.xlim((0, 0.08))


#%% train the nn emulator
tch.manual_seed(1)    # reproducible

BATCH_SIZE = 10
EPOCH = 30

pred_nn_cross = np.zeros(n_s)
mse_nn = np.zeros(10)

# 10-fold validation
for k in range(10):
    leaveoutind = k + 10 * np.arange(98)
    allind = np.arange(n_s)
    trainind = np.delete(allind, leaveoutind)
    
    x_train = x_s[trainind, :]
    y_train = y_s[trainind]
    
    x_test = x_s[leaveoutind, :]
    y_test = y_s[leaveoutind]   
    
    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x_train), Variable(y_train)
    torch_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)
    
    # another way to define a network
    net = tch.nn.Sequential(
            tch.nn.Linear(10, 100),
            tch.nn.LeakyReLU(),
            tch.nn.Linear(100, 50),
            tch.nn.LeakyReLU(),
            tch.nn.Linear(50, 1),
        )
    
    optimizer = tch.optim.Adam(net.parameters(), lr=0.05)
    loss_func = tch.nn.MSELoss()  # this is for regression mean squared loss
    
    # start training
    for epoch in range(EPOCH):
        net.train()
        
        for step, (batch_x, batch_y) in enumerate(train_loader): # for each training step
            
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
    
            prediction = net(b_x)     # input x and predict based on x
    
            loss = loss_func(prediction.squeeze(), b_y)     # must be (1. nn output, 2. target)
    
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
    
    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x_test), Variable(y_test)
    
    pred_nn_cross[leaveoutind] = net(x).squeeze().detach().numpy()
    mse_nn[k] = loss_func(net(x).squeeze(), y)
    
plt.scatter(pred_nn_cross, y_s)
plt.ylim((0, 0.08))
plt.xlim((0, 0.08))