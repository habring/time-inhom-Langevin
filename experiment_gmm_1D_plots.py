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
device = th.device('cpu')

# choose which plots to generate. "unif_steps" refers to the experiment where the same step size is used for all methods.
folder_ = f'results/gmm_1d/unif_steps/'
# folder_ = f'results/gmm_1d/'



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 20
})

methods = {
            'ULA': True,
            'dilation':True,
            'tempering':True,
            'diffusion':True,
            'daz':True
}

N = 10000   # number of samples gt
N_gt = 10000   # number of samples gt
check_iter = 100
means = th.Tensor([[-2],[0],[2]]).to(device)
sigmas=th.Tensor([[0.2],[0.1],[0.3]]).to(device)
mixture_weights = th.Tensor([0.3,0.4,0.3]).to(device)
gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)

# ground truth samples
sample = gmm.sample(N_gt,tau=0.0)
direct_sample_hist, direct_sample_bins = th.histogram(sample.cpu(),bins=100,density=True)
x_vals = (direct_sample_bins[1:]+direct_sample_bins[:-1])/2
xx = th.tensor(np.linspace(x_vals.min(),x_vals.max(),1000))
yy = gmm(xx[...,None].to(device),tau=0).cpu().numpy()
df_gt = pd.DataFrame(index = xx.cpu().numpy(), data={'Ground truth density':yy})
folder = f'results/gmm_1d'
df_gt.index.rename('x',inplace=True)
df_gt.to_csv(f'{folder}/gt_density.csv')


gt_hist = th.zeros(direct_sample_hist.shape).to('cpu')
for i in range(len(direct_sample_bins)-1):
    x_left = direct_sample_bins[i]
    x_right = direct_sample_bins[i+1]
    gt_hist[i] = 0.5*(gmm(x_left,tau=0) + gmm(x_right,tau=0))


sample_hist, sample_bins = th.histogram(sample.cpu()[:N],density=True,bins = direct_sample_bins)
KL_gt = util.KL(sample_hist,gt_hist,sample_bins)
TV_gt = util.TV(sample_hist,gt_hist,sample_bins)


for T in [.1,1.,2.,10.]:
    folder = f'{folder_}T_{T}'
    dfs = []
    

    for method in methods.keys():
        if methods[method]:
            df = pd.read_csv(f'{folder}/err_{method}.txt',sep=';',dtype=np.float64)
            {'TV':f'TV_{method}','KL':f'KL_{method}'}
            df = df.rename(columns={'TV':f'TV {method}','KL':f'KL {method}'})
            dfs.append(df)

    l = np.array([len(f) for f in dfs]).max()
    KLs = []
    TVs = []
    for i in range(l):
        sample = gmm.sample(N_gt,tau=0.0)
        sample_hist, sample_bins = th.histogram(sample.cpu()[:N],density=True,bins = direct_sample_bins)
        KL_gt = util.KL(sample_hist,gt_hist,sample_bins)
        TV_gt = util.TV(sample_hist,gt_hist,sample_bins)
        KLs.append(KL_gt.item())
        TVs.append(TV_gt.item())

    KLs = th.tensor(KLs)
    TVs = th.tensor(TVs)

    df = pd.DataFrame(data = {f'TV_gt':TVs,f'KL_gt':KLs})
    dfs = [df] + dfs

    df = pd.concat(dfs,axis=1)
    t = np.arange(len(df))
    df.index = t
    k = 5
    df = df.groupby(df.index // k).mean()#.reset_index(drop=True)
    t = np.arange(len(df))*k*check_iter
    df.index = t
    df.index.rename('iter',inplace=True)

    cols1 = [l for l in df.columns if 'TV' in l]
    df_ = df[cols1].copy()
    df_ = df_.rename(columns={f'TV {method}':f'{method}' for method in methods.keys()})
    df_ = df_.rename(columns={'daz':'DAZ'})
    df_.plot(figsize=(15,10))
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\mathrm{TV}(\hat \mu_k,\pi)$')
    plt.yscale('log')
    plt.savefig(f'{folder}/TV_comparison.png')
    plt.close('all')
    df_.to_csv(f'{folder}/TV_comparison.csv')

    cols2 = [l for l in df.columns if 'KL' in l]
    df_ = df[cols2].copy()
    df_ = df_.rename(columns={f'KL {method}':f'{method}' for method in methods.keys()})
    df_ = df_.rename(columns={'daz':'DAZ'})
    df_.plot(figsize=(15,10))
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\mathrm{KL}(\hat \mu_k,\pi)$')
    plt.yscale('log')
    plt.savefig(f'{folder}/KL_comparison.png')
    plt.close('all')
    df_.to_csv(f'{folder}/KL_comparison.csv')

    for iter in [0,1000,2000,3000,4000,5000,40000]:

        dfs = []

        for method in methods.keys():
            if methods[method]:
                sample = th.load(f'{folder}/{method}_samples/sample_iter_{iter}',map_location='cpu')
                hist, _ = th.histogram(sample.cpu(),bins=direct_sample_bins,density=True)

                df = pd.DataFrame(index = x_vals.cpu().numpy(), data={method:hist})
                dfs.append(df)

        df = pd.concat(dfs,axis=1)
        df.index.rename('x',inplace=True)

        df.to_csv(f'{folder}/histo_comparison_iter_{iter}.csv')

