import torch as th
import numpy as np
import matplotlib.pyplot as plt
import src.potentials as pot
import src.algorithms as algo
import src.utils as util
import os,sys
import pandas as pd
from pathlib import Path
import torchist

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

methods = {
            'diffusion':True,
            'ULA': True,
            'tempering':True,
            'dilation':True,
            'DAZ':True
}

print(methods.keys())
N = 5000   # number of samples
N_gt = 5000   # number of samples gt
check_iter = 100
show_plot = False
save_sample = [i*1000 for i in range(10)] + [i*10000 for i in range(10)]
tmax=10
x_init = th.zeros([N,1]).to(device)-1

means = th.Tensor([[-2],[0],[2]]).to(device)
sigmas=th.Tensor([[0.2],[0.1],[0.3]]).to(device)
L=1/sigmas.min()**2
a = 1/sigmas.max()**2
step = a/L**2
mixture_weights = th.Tensor([0.3,0.4,0.3]).to(device)
gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)

times_list = []


T=1
tau = lambda t: th.exp(-t/T)*.99

if methods['diffusion'] == True:
    # compute allowed step sizes
    times = [0]
    while times[-1]<tmax:
        tau_val = tau(th.tensor([times[-1]])).to(device)
        sigma_max_sq = (1-tau_val)*sigmas.max()**2 + tau_val
        sigma_min_sq = (1-tau_val)*sigmas.min()**2 + tau_val
        L=1/sigma_min_sq
        a = 1/sigma_max_sq
        step = a/L**2
        times.append(times[-1]+step)
    times = th.tensor(times)#.to(device)
    times_list.append(times)
    print('Diffusion')


if methods['ULA'] == True:
    L=1/sigmas.min()**2
    a = 1/sigmas.max()**2
    step = a/L**2
    times = th.Tensor(np.arange(0,tmax,step.cpu()))
    print('ULA')
    times_list.append(times)

if methods['tempering'] == True:
    times = [0]
    while times[-1]<tmax:
        tau_val = tau(th.tensor([times[-1]])).to(device)
        L=(1-tau_val)*1/sigmas.min()**2 + tau_val
        a = (1-tau_val)*1/sigmas.max()**2 + tau_val
        step = a/L**2
        times.append(times[-1]+step)
    times = th.tensor(times)#.to(device)
    times_list.append(times)


if methods['dilation'] == True:
    times = [0]
    while times[-1]<tmax:
        tau_val = tau(th.tensor([times[-1]])).to(device)
        L=1/sigmas.min()**2 / (1-tau_val)
        a = 1/sigmas.max()**2 / (1-tau_val)
        step = a/L**2
        times.append(times[-1]+step)
    times = th.tensor(times)#.to(device)
    times_list.append(times)

    print('dilation')


if methods['DAZ'] == True:
    times = [0]
    while times[-1]<tmax:
        tau_val = tau(th.tensor([times[-1]])).to(device)
        L_gt=1/sigmas.min()**2
        a_gt = 1/sigmas.max()**2

        L = 1/tau_val
        if 1-L_gt*tau_val>0:
            L = th.minimum(L, L_gt/(1-L_gt*tau_val))
        a = th.minimum(a_gt,1/tau_val)
        step = a/L**2
        times.append(times[-1]+step)
    times = th.tensor(times)#.to(device)
    times_list.append(times)


plt.figure()
for i in range(len(methods)):
    plt.plot(list(range(len(times_list[i]))),times_list[i],label=list(methods.keys())[i])

plt.legend()
plt.show()
