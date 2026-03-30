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

N = 5000   # number of samples
N_gt = 5000
check_iter = 10
show_plot = False
save_sample = 1000
x_init = th.zeros([N,2]).to(device)-1

methods = {
            'ULA': True,
            'dilation':True,
            'tempering':True,
            'diffusion':True,
            'daz':True
}

method_label_dict = {
            'ULA': 'ULA',
            'dilation':'Dilation',
            'tempering':'Tempering',
            'diffusion':'Diffusion',
            'daz':'DAZ'
}


means = th.tensor([[0.,0],
                   [2,0],
                   [0,2],
                   [2,2]]).to(device)

sigmas = th.tensor([[0.2,0.2],
                   [0.1,0.2],
                   [0.3,0.1],
                   [.1,.1]]).to(device)
mixture_weights = th.Tensor([0.2,0.4,0.2,.2]).to(device)

L=1/sigmas.min()**2
a = 1/sigmas.max()**2
step = a/L**2


gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)
# ground truth samples
sample = gmm.sample(N_gt,tau=0.0)
direct_sample_hist, direct_sample_bins = th.histogramdd(sample.cpu(),bins=[200,202],density=True)
x_vals = (direct_sample_bins[0][1:]+direct_sample_bins[0][:-1])/2
y_vals = (direct_sample_bins[1][1:]+direct_sample_bins[1][:-1])/2
xmin = x_vals.min()
xmax = x_vals.max()
ymin = y_vals.min()
ymax = y_vals.max()

xx,yy = th.meshgrid(x_vals,y_vals)
coordinates = th.cat([xx[...,None],yy[...,None]],dim=-1)
coordinates = coordinates.view((coordinates.shape[:-1].numel(),coordinates.shape[-1])).to(device)
gt_hist = gmm(coordinates,tau=0).view(xx.shape).to('cpu')


KL_gt = util.KL2D(direct_sample_hist,gt_hist,direct_sample_bins)
TV_gt = util.TV2D(direct_sample_hist,gt_hist,direct_sample_bins)

for T in [.1,1.,2.0,10.]:
    folder = f'results/gmm_2d/T_{T}'
    Path(folder).mkdir(exist_ok=True,parents=True)
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
        sample_hist, sample_bins = th.histogramdd(sample.cpu()[:N],density=True,bins = direct_sample_bins)
        KL_gt = util.KL2D(sample_hist,gt_hist,sample_bins)
        TV_gt = util.TV2D(sample_hist,gt_hist,sample_bins)
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
    t = np.arange(len(df))*check_iter*k
    df.index = t
    df.index.rename('iter',inplace=True)

    cols1 = [l for l in df.columns if 'TV' in l]
    df_ = df[cols1].copy()
    df_ = df_.rename(columns={f'TV {method}':f'{method}' for method in methods.keys()})
    df_ = df_.rename(columns={'daz':'DAZ'})
    df_.plot(figsize=(15,10))
    df_.to_csv(f'{folder}/TV_comparison.csv')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\mathrm{TV}(\hat \mu_k,\pi)$')
    plt.yscale('log')
    plt.savefig(f'{folder}/TV_comparison.png')
    plt.close('all')

    cols2 = [l for l in df.columns if 'KL' in l]
    df_ = df[cols2].copy()
    df_ = df_.rename(columns={f'KL {method}':f'{method}' for method in methods.keys()})
    df_ = df_.rename(columns={'daz':'DAZ'})
    df_.plot(figsize=(15,10))
    df_.to_csv(f'{folder}/KL_comparison.csv')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\mathrm{KL}(\hat \mu_k,\pi)$')
    plt.yscale('log')
    plt.savefig(f'{folder}/KL_comparison.png')
    plt.close('all')


    plot_rows = 2
    method_names = [m for m in methods.keys() if methods[m]]
    plot_cols = int(np.ceil((len(method_names))/plot_rows))

    for iter in [0,1000,2000,3000,4000,5000,40000]:

        # fig, ax = plt.subplots(plot_rows,plot_cols, figsize = (plot_rows*10,plot_cols*10),
        #                         sharex=True,
        #                         sharey=True,
        #                         constrained_layout=True,)

        # ax[0,0].imshow(gt_hist,extent=(xmin, xmax, ymin, ymax),)
        # ax[0,0].set_title('Ground truth')
        
        plt.imshow(gt_hist,extent=(xmin, xmax, ymin, ymax))
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.savefig(f'{folder}/histo_comparison_iter_{iter}_gt.png',bbox_inches='tight',
    pad_inches=0)

        for i, m in enumerate(method_names):
            sample = th.load(f'{folder}/{m}_samples/sample_iter_{iter}',map_location=device)
            hist, _ = th.histogramdd(sample.cpu(),bins=direct_sample_bins,density=True)

            # jj = (i+1) % plot_cols
            # ii = int((i+1)/plot_cols)
            # ax[ii,jj].imshow(hist,extent=(xmin, xmax, ymin, ymax),)
            # ax[ii,jj].set_title(method_label_dict[m])

            plt.imshow(hist,extent=(xmin, xmax, ymin, ymax))
            plt.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            plt.savefig(f'{folder}/histo_comparison_iter_{iter}_{m}.png',bbox_inches='tight',
    pad_inches=0)

        
        # fig.tight_layout()
        # fig.savefig(f'{folder}/histo_comparison_iter_{iter}.png')
        # plt.show()
        plt.close('all')