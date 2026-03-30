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

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
folder_ = f'results/gmm_1d/unif_steps/'
Path(folder_).mkdir(parents=True,exist_ok=True)

methods = {
            'ULA': True,
            'dilation':True,
            'tempering':True,
            'diffusion':True,
            'DAZ':True
}

N = 5000   # number of samples
N_gt = 5000   # number of samples gt
check_iter = 100
show_plot = False
save_sample = [i*1000 for i in range(10)] + [i*10000 for i in range(10)]
tmax=150
x_init = th.zeros([N,1]).to(device)-1

means = th.Tensor([[-2],[0],[2]]).to(device)
sigmas=th.Tensor([[0.2],[0.1],[0.3]]).to(device)
L=1/sigmas.min()**2
a = 1/sigmas.max()**2
step = a/L**2
mixture_weights = th.Tensor([0.3,0.4,0.3]).to(device)
gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)
# ground truth samples
sample = gmm.sample(N_gt,tau=0.0)
th.save(sample,f'{folder_}/gt_sample')
direct_sample_hist, direct_sample_bins = th.histogram(sample.cpu(),bins=200,density=True)
direct_sample_hist = direct_sample_hist.to(device)
direct_sample_bins = direct_sample_bins.to(device)
x_vals = (direct_sample_bins[1:]+direct_sample_bins[:-1])/2

gt_hist = th.zeros(direct_sample_hist.shape).to(device)#.to('cpu')
for i in range(len(direct_sample_bins)-1):
    x_left = direct_sample_bins[i]
    x_right = direct_sample_bins[i+1]
    gt_hist[i] = 0.5*(gmm(x_left,tau=0) + gmm(x_right,tau=0))

for T in [.1, 1., 2.,10.]:
    print(f'T={T}')
    folder = f'{folder_}T_{T}'

    tau = lambda t: th.exp(-t/T)*.99
    Path(folder).mkdir(parents=True,exist_ok=True)


    def plot_from_txt(file,name):
        df = pd.read_csv(file,sep=';',dtype=np.float64)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('TV')
        ln1 = ax1.plot(df[df.columns[0]],color='blue',label='TV')
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.set_ylabel('KL')
        ln2 = ax2.plot(df[df.columns[1]],color='red',label='KL')
        lns = ln1+ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f'{name}.png')
        plt.close('all')

    try:
        os.remove('KL.txt')
    except:
        pass

    def callback(alg, state,write_file,dir):
        if state.n % check_iter == 0:
            # sample_hist, sample_bins = th.histogram(state.x_out.cpu(),density=True,bins = direct_sample_bins)
            sample_hist = torchist.histogramdd(state.x_out, edges = direct_sample_bins)
            KL = util.KL(sample_hist,gt_hist,direct_sample_bins)
            TV = util.TV(sample_hist,gt_hist,direct_sample_bins)

            if show_plot:
                plt.plot(x_vals, sample_hist)
                plt.plot(x_vals, gt_hist)
                plt.title(f'Time: {state.t}. Iter: {state.n}. KL {KL}')
                plt.show()

            if state.n==0:
                try:
                    os.remove(write_file)
                except:
                    pass
            with open(write_file, "a") as myfile:
                if state.n==0:
                    myfile.write(f'TV;KL\n')
                myfile.write(f'{TV};{KL}\n')
        
        if state.n % 10000 == 0:
            print(state.n)

        if state.n in save_sample:
            Path(dir).mkdir(exist_ok=True)
            th.save(state.x_out, f'{dir}/sample_iter_{state.n}')

        return
    
    L=1/sigmas.min()**2
    a = 1/sigmas.max()**2
    step = a/L**2
    times = th.Tensor(np.arange(0,tmax,step.cpu()))
    taus = tau(times)
    

    if methods['diffusion'] == True:
        print('Diffusion')

        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=gmm.score)
        
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_diffusion.txt',dir=f'{folder}/diffusion_samples')

        sample = sampler(x_init = x_init, callback_fn = callback_)

        plot_from_txt(f'{folder}/err_diffusion.txt',f'{folder}/err_diffusion')

    if methods['ULA'] == True:
        print('ULA')
        taus_ = times*0.
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_ULA.txt',dir=f'{folder}/ULA_samples')
        sampler = algo.GeneralAnnealing(times=times,taus=taus_,
                nabla_f=gmm.score)
        sample = sampler(x_init = x_init, callback_fn = callback_)

        plot_from_txt(f'{folder}/err_ULA.txt',f'{folder}/err_ULA')

    if methods['tempering'] == True:
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
        print('Dilation')

        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_dilation.txt',dir=f'{folder}/dilation_samples')
        def nabla_f_dilation(x,tau):
            return np.sqrt(1-tau)*gmm.score(x/np.sqrt(1-tau),0)
        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=nabla_f_dilation)
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_dilation.txt',f'{folder}/err_dilation')


    if methods['DAZ'] == True:
        print('DAZ')

        def nabla_f_DAZ(x,tau):
            nabla_f = lambda y: -gmm.score(y,0)
            f = lambda y: -th.log(gmm(y,0))

            # prox = util.APGD_prox(nabla_f=nabla_f, f=f, t=tau, u=x, x_init=x, num_iter=100, J=1, beta=0.9, gamma=1.5, tol=1e-7,L=100)
            prox, grad = util.APGD_prox_multi_init(nabla_f=nabla_f, f=f, t=tau, u=x, x_init=means, num_iter=40, tol=1e-7,L=L)
            grad = -grad
            return grad
        
        taus_ = taus
        sampler = algo.GeneralAnnealing(times=times,taus=taus_,
                nabla_f=nabla_f_DAZ)
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_daz.txt',dir=f'{folder}/daz_samples')
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_daz.txt',f'{folder}/err_daz')


