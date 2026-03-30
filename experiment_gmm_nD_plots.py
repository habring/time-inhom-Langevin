import torch as th
import numpy as np
import matplotlib.pyplot as plt
import src.potentials as pot
import src.algorithms as algo
import src.utils as util
import os
import pandas as pd
from pathlib import Path

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 20
})


check_iter = 1
daz_scale = 1

methods = {
            'ULA': True,
            'dilation':True,
            'tempering':True,
            'diffusion':True,
            f'daz_scale{1}':True,
            # f'daz_scale{2}':True
}

method_label_dict = {
            'ULA': 'ULA',
            'dilation':'Dilation',
            'tempering':'Tempering',
            'diffusion':'Diffusion',
            'daz':'DAZ'
}

folder = f'results/gmm_nd_right/'

N_gt = 5000
means = th.load(f'{folder}means.th',map_location=device)
sigmas = th.load(f'{folder}sigmas.th',map_location=device)
mixture_weights = th.load(f'{folder}mixture_weights.th',map_location=device)
gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)
# ground truth samples
# sample_gt = gmm.sample(N_gt,tau=0.0)



for T in [.1,1.,2.0,10.]:
    folder_ = f'{folder}T_{T}'
    dfs = []

    for method in methods.keys():
        if methods[method]:
            df = pd.read_csv(f'{folder_}/err_{method}.txt',sep=';',dtype=np.float64)
            df = df.rename(columns={fi:f'{fi}_{method}' for fi in df.columns})
            dfs.append(df)


    dfs = pd.concat(dfs,axis=1)
    dfs.index.rename('iter',inplace=True)

    for i in range(4):
        df_tmp = dfs[[fi for fi in dfs.columns if f'KL_{i}' in fi]]
        k = 50
        df_tmp = df_tmp.groupby(df_tmp.index // k).mean()#.reset_index(drop=True)
        t = np.arange(len(df_tmp))*k*check_iter
        df_tmp.index = t
        df_tmp.index.rename('iter',inplace=True)
        name = f'{folder}/T{T}_KLmarginal{i}'.replace('.','')
        name = f'{name}.csv'
        df_tmp.to_csv(name)
        # df_tmp.iloc[:5000].plot()
        # plt.title(f'T={T}')
        # plt.show()

    # l = len(dfs[0])

    # df = pd.concat(dfs,axis=1)
    # t = np.arange(len(df))
    # df.index = t
    # k = 1#5
    # df = df.groupby(df.index // k).mean()#.reset_index(drop=True)
    # t = np.arange(len(df))*k*check_iter
    # df.index = t
    # df.index.rename('iter',inplace=True)

