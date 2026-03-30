import torch as th
import numpy as np
import matplotlib.pyplot as plt
import src.potentials as pot
import src.algorithms as algo
import src.utils as util
import os
import pandas as pd
from pathlib import Path
import torchist

device = th.device('cuda:3' if th.cuda.is_available() else 'cpu')
print(device)

# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "Helvetica",
#     "font.size": 20,
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })

N = 5000   # number of samples
N_gt = 5000   # number of samples gt
d = 10      # space dimension
maxit = 50000
daz_scale = 1
n_modes =   4
check_iter = 10
save_sample = [i*1000 for i in range(10)] + [i*10000 for i in range(10)]
x_init = th.zeros([N,d]).to(device)-1
show_plot=False

methods = {
            'ULA': True,
            'dilation':True,
            'tempering':True,
            'diffusion':True,
            'DAZ':True
}


# generate random means
th.manual_seed(0)
means = th.randn(n_modes,d).to(device)
sigmas = th.rand(n_modes,d).to(device)*0.3+0.1
mixture_weights = th.Tensor([0.2,0.4,0.2,.2]).to(device)

folder = f'results/gmm_nd'
Path(folder).mkdir(parents=True,exist_ok=True)
th.save(means,f'{folder}/means.th')
th.save(sigmas,f'{folder}/sigmas.th')
th.save(mixture_weights,f'{folder}/mixture_weights.th')
L=1/sigmas.min()**2
a = 1/sigmas.max()**2
step = a/L**2
print(f'step: {step}')
gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)
# ground truth samples
sample = gmm.sample(N_gt,tau=0.0)
th.save(sample,f'{folder}/gt_sample')

direct_sample_hist = []
direct_sample_bins = []

marginals = list(range(5))

for i in marginals:
    direct_sample_hist_i, direct_sample_bins_i = th.histogram(sample.cpu()[:,i],bins=200,density=True)
    direct_sample_hist_i = direct_sample_hist_i.to(device)
    direct_sample_bins_i = direct_sample_bins_i.to(device)

    direct_sample_hist.append(direct_sample_hist_i)
    direct_sample_bins.append(direct_sample_bins_i)
    
tmax=50


for T in [10.0]:
    folder = f'results/gmm_nd/T_{T}'
    Path(folder).mkdir(parents=True,exist_ok=True)

    tau = lambda t: th.exp(-t/T)*.99

    def plot_from_txt(file,name):
        df = pd.read_csv(file,sep=';',dtype=np.float64)
        df = df[[c for c in df.columns if 'KL' in c]]
        df.plot()
        plt.savefig(f'{name}.png')


    def callback(alg, state,write_file,dir):
        if state.n % check_iter == 0:
            if state.n==0:
                try:
                    os.remove(write_file)
                except:
                    pass

                with open(write_file, "a") as myfile:
                    for i in marginals[:-1]:
                        myfile.write(f'TV_{i};KL_{i};')
                    i = marginals[-1]
                    myfile.write(f'TV_{i};KL_{i}\n')

            for i in marginals:
                s = state.x_out[:,i]
                s = s[:,th.newaxis]
                sample_hist = torchist.histogramdd(s, edges = direct_sample_bins[i])
                KL = util.KL(sample_hist,direct_sample_hist[i],direct_sample_bins[i])
                TV = util.TV(sample_hist,direct_sample_hist[i],direct_sample_bins[i])

                with open(write_file, "a") as myfile:
                    if i<len(marginals)-1:
                        myfile.write(f'{TV};{KL};')
                    else:
                        myfile.write(f'{TV};{KL}\n')

        if state.n in save_sample:
            Path(dir).mkdir(exist_ok=True,parents=True)
            th.save(state.x_in, f'{dir}/sample_iter_{state.n}')

        return


    if methods['diffusion'] == True:
        times = [0]
        while times[-1]<tmax and len(times)<maxit:
            tau_val = tau(th.tensor([times[-1]])).to(device)
            sigma_max_sq = (1-tau_val)*sigmas.max()**2 + tau_val
            sigma_min_sq = (1-tau_val)*sigmas.min()**2 + tau_val
            L=1/sigma_min_sq
            a = 1/sigma_max_sq
            step = a/L**2
            times.append(times[-1]+step)
        times = th.tensor(times)#.to(device)
        taus = tau(times)
        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=gmm.score)
        
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_diffusion.txt',dir=f'{folder}/diffusion_samples')
        
        print('Diffusion')
        sample = sampler(x_init = x_init, callback_fn = callback_)

        plot_from_txt(f'{folder}/err_diffusion.txt',f'{folder}/err_diffusion')

    if methods['ULA'] == True:
        times = th.Tensor(np.arange(0,tmax,step.cpu()))
        times = times[0:maxit]
        print('ULA')
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_ULA.txt',dir=f'{folder}/ULA_samples')
        taus_ = times*0.
        sampler = algo.GeneralAnnealing(times=times,taus=taus_,
                nabla_f=gmm.score)
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_ULA.txt',f'{folder}/err_ULA')

    if methods['tempering'] == True:
        times = [0]
        while times[-1]<tmax and len(times)<maxit:
            tau_val = tau(th.tensor([times[-1]])).to(device)
            L=(1-tau_val)*1/sigmas.min()**2 + tau_val
            a = (1-tau_val)*1/sigmas.max()**2 + tau_val
            step = a/L**2
            times.append(times[-1]+step)
        times = th.tensor(times)#.to(device)
        taus = tau(times)
        print('Tempering')
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_tempering.txt',dir=f'{folder}/tempering_samples')
        def nabla_f_tempered(x,tau):
            return (1-tau)*gmm.score(x/np.sqrt(1-tau),0) - tau*x
        
        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=nabla_f_tempered)
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_tempering.txt',f'{folder}/err_tempering')


    if methods['dilation'] == True:
        times = [0]
        while times[-1]<tmax and len(times)<maxit:
            tau_val = tau(th.tensor([times[-1]])).to(device)
            L=1/sigmas.min()**2 / (1-tau_val)
            a = 1/sigmas.max()**2 / (1-tau_val)
            step = a/L**2
            times.append(times[-1]+step)
        times = th.tensor(times)#.to(device)
        taus = tau(times)
        print('dilation')

        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_dilation.txt',dir=f'{folder}/dilation_samples')
        def nabla_f_dilation(x,tau):
            return np.sqrt(1-tau)*gmm.score(x/np.sqrt(1-tau),0)
        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=nabla_f_dilation)
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_dilation.txt',f'{folder}/err_dilation')


    if methods['DAZ'] == True:
        tau = lambda t: daz_scale*th.exp(-t/T)*.99
        times = [0]
        while times[-1]<tmax and len(times)<maxit:
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
        print('DAZ')

        def nabla_f_DAZ(x,tau):
            nabla_f = lambda y: -gmm.score(y,0)
            f = lambda y: -th.log(gmm(y,0))

            # prox = util.APGD_prox(nabla_f=nabla_f, f=f, t=tau, u=x, x_init=x, num_iter=100, J=1, beta=0.9, gamma=1.5, tol=1e-7,L=100)
            prox, grad = util.APGD_prox_multi_init(nabla_f=nabla_f, f=f, t=tau, u=x, x_init=means, num_iter=50, tol=1e-7,L=L)
            grad = -grad
            return grad
        
        taus_ = taus
        sampler = algo.GeneralAnnealing(times=times,taus=taus_,
                nabla_f=nabla_f_DAZ)
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_daz_scale{daz_scale}.txt',dir=f'{folder}/daz_scale{daz_scale}_samples')
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_daz_scale{daz_scale}.txt',f'{folder}/err_daz_scale{daz_scale}')
