import torch as th
import numpy as np
import matplotlib.pyplot as plt
import src.potentials as pot
import src.algorithms as algo
import src.utils as util
import os,sys
import pandas as pd
from pathlib import Path

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
folder = f'results/gmm_1d/'
Path(folder).mkdir(parents=True,exist_ok=True)

methods = {
            'ULA': True,
            'dilation':True,
            'tempering':True,
            'diffusion':True,
            'DAZ':True
}

T_list = [.1, 1., 2.,10.]
tmax_dict = {0.1:2,1.:10,2.:25,10.0:100}

means = th.Tensor([[-2],[0],[2]]).to(device)
sigmas=th.Tensor([[0.2],[0.1],[0.3]]).to(device)
mixture_weights = th.Tensor([0.3,0.4,0.3]).to(device)
gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)

times_dict = {}
taus_dict = {}


def downsample_df(df, target_length):
    """
    Downsample a pandas DataFrame to a target length using simple subsampling.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame to downsample.
    - target_length: int
        The desired number of rows in the output DataFrame.

    Returns:
    - pandas.DataFrame
        The downsampled DataFrame.
    """
    if target_length >= len(df):
        return df.copy()

    # Calculate the step size for subsampling
    step = len(df) / target_length
    indices = [int(i * step) for i in range(target_length)]
    return df.iloc[indices]#.reset_index(drop=True)



for T in T_list:
    tmax = tmax_dict[T]
    print(f'T={T}')
    l_max = 0
    folder_ = f'results/gmm_1d/T_{T}'

    tau = lambda t: th.exp(-t/T)*.99
    Path(folder_).mkdir(parents=True,exist_ok=True)


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
        taus = tau(times)
        l_max = max(l_max,len(taus))
        times_dict['diffusion'] = times
        taus_dict['diffusion'] = taus

    if methods['ULA'] == True:
        L=1/sigmas.min()**2
        a = 1/sigmas.max()**2
        step = a/L**2
        times = th.Tensor(np.arange(0,tmax,step.cpu()))
        print('ULA')
        taus_ = times*0.
        l_max = max(l_max,len(taus))

        times_dict['ULA'] = times
        taus_dict['ULA'] = taus

    if methods['tempering'] == True:
        times = [0]
        while times[-1]<tmax:
            tau_val = tau(th.tensor([times[-1]])).to(device)
            L=(1-tau_val)*1/sigmas.min()**2 + tau_val
            a = (1-tau_val)*1/sigmas.max()**2 + tau_val
            step = a/L**2
            times.append(times[-1]+step)
        times = th.tensor(times)#.to(device)
        taus = tau(times)
        l_max = max(l_max,len(taus))

        times_dict['tempering'] = times
        taus_dict['tempering'] = taus


    if methods['dilation'] == True:
        times = [0]
        while times[-1]<tmax:
            tau_val = tau(th.tensor([times[-1]])).to(device)
            L=1/sigmas.min()**2 / (1-tau_val)
            a = 1/sigmas.max()**2 / (1-tau_val)
            step = a/L**2
            times.append(times[-1]+step)
        times = th.tensor(times)#.to(device)
        taus = tau(times)
        l_max = max(l_max,len(taus))

        print('dilation')

        times_dict['dilation'] = times
        taus_dict['dilation'] = taus


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
        taus = tau(times)
        l_max = max(l_max,len(taus))

        times_dict['DAZ'] = times
        taus_dict['DAZ'] = taus

        print('DAZ')

    
    df_times = pd.DataFrame(data = {k:pd.Series(v) for k,v in times_dict.items()})
                            # columns=['diffusion', 'ULA', 'tempering', 'dilation','DAZ'])
    df_taus = pd.DataFrame({k:pd.Series(v) for k,v in taus_dict.items()})

    df_times.index.rename('iter',inplace=True)
    df_taus.index.rename('iter',inplace=True)


    df_times = downsample_df(df_times,2000)
    df_taus = downsample_df(df_taus,2000)

    df_times.to_csv(f'{folder_}/steps.csv')
    df_taus.to_csv(f'{folder_}/taus.csv')

    df_times.plot()
    plt.savefig(f'{folder_}/steps.png')