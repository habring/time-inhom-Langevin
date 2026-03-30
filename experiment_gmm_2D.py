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

# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "Helvetica",
#     "font.size": 20
# })

N = 5000   # number of samples
N_gt = 5000 # number of GT samples for comparison
check_iter = 10
show_plot = False
save_sample = [i*1000 for i in range(10)] + [i*10000 for i in range(10)]
x_init = th.zeros([N,2]).to(device)-1
tmax=100


methods = {
            'ULA': True,
            'dilation':True,
            'tempering':True,
            'diffusion':True,
            'DAZ':True
}

means = th.tensor([[0.,0],
                   [2,0],
                   [0,2],
                   [2,2]]).to(device)

sigmas = th.tensor([[0.2,0.2],
                   [0.1,0.2],
                   [0.3,0.1],
                   [.1,.1]]).to(device)

L=1/sigmas.min()**2
a = 1/sigmas.max()**2
step = a/L**2
print(f'step: {step}')
mixture_weights = th.Tensor([0.2,0.4,0.2,.2]).to(device)
gmm = pot.GMM_diffusion(means = means, sigmas=sigmas, mixture_weights=mixture_weights)


for T in [.1,1.,2.,10.]:
    print(f'T={T}')
    folder = f'results/gmm_2d/T_{T}'
    Path(folder).mkdir(exist_ok=True,parents=True)
    # ground truth samples
    sample = gmm.sample(N_gt,tau=0.0)
    th.save(sample,f'{folder}/gt_sample')

    tau = lambda t: th.exp(-t/T)*.99

    direct_sample_hist, direct_sample_bins = th.histogramdd(sample.cpu(),bins=[200,202],density=True)
    direct_sample_bins = [direct_sample_bins[0].to(device),direct_sample_bins[1].to(device)]
    x_vals = (direct_sample_bins[0][1:]+direct_sample_bins[0][:-1])/2
    y_vals = (direct_sample_bins[1][1:]+direct_sample_bins[1][:-1])/2
    xmin = x_vals.min()
    xmax = x_vals.max()
    ymin = y_vals.min()
    ymax = y_vals.max()

    xx,yy = th.meshgrid(x_vals,y_vals)
    coordinates = th.cat([xx[...,None],yy[...,None]],dim=-1)
    coordinates = coordinates.view((coordinates.shape[:-1].numel(),coordinates.shape[-1])).to(device)
    gt_hist = gmm(coordinates,tau=0).view(xx.shape)#.to('cpu')


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


    try:
        os.remove('KL.txt')
    except:
        pass


    def callback(alg, state,write_file,dir):
        if state.n % check_iter == 0:
            # sample_hist, sample_bins = th.histogramdd(state.x_out,density=True,bins = direct_sample_bins)
            sample_hist = torchist.histogramdd(state.x_out, edges = direct_sample_bins)

            KL = util.KL2D(sample_hist,gt_hist,direct_sample_bins)
            TV = util.TV2D(sample_hist,gt_hist,direct_sample_bins)

            if show_plot:
                fig,ax = plt.subplots(2,1,figsize=(20,10))
                ax[0].imshow(sample_hist.cpu())
                ax[1].imshow(gt_hist.cpu())
                plt.title(f'Time: {state.t}. KL {KL}')
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

        if state.n in save_sample:
            Path(dir).mkdir(exist_ok=True,parents=True)
            th.save(state.x_out, f'{dir}/sample_iter_{state.n}')

        return


    if methods['diffusion'] == True:
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
        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=gmm.score)
        
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_diffusion.txt',dir=f'{folder}/diffusion_samples')
        print('Diffusion')
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_diffusion.txt',f'{folder}/err_diffusion')

    if methods['ULA'] == True:
        times = th.Tensor(np.arange(0,tmax,step.cpu()))
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_ULA.txt',dir=f'{folder}/ULA_samples')
        taus_ = times*0.
        sampler = algo.GeneralAnnealing(times=times,taus=taus_,
                nabla_f=gmm.score)
        print('ULA')
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_ULA.txt',f'{folder}/err_ULA')

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
        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_tempering.txt',dir=f'{folder}/tempering_samples')
        def nabla_f_tempered(x,tau):
            return (1-tau)*gmm.score(x/np.sqrt(1-tau),0) - tau*x
        
        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=nabla_f_tempered)
        print('Tempering')
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_tempering.txt',f'{folder}/err_tempering')


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

        def callback_(algo,state):
            return callback(alg=algo,state=state, write_file=f'{folder}/err_dilation.txt',dir=f'{folder}/dilation_samples')
        def nabla_f_dilation(x,tau):
            return np.sqrt(1-tau)*gmm.score(x/np.sqrt(1-tau),0)
        sampler = algo.GeneralAnnealing(times=times,taus=taus,
                nabla_f=nabla_f_dilation)
        print('dilation')
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_dilation.txt',f'{folder}/err_dilation')


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
            print(step)
        times = th.tensor(times)#.to(device)
        taus = tau(times)

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
        print('DAZ')
        sample = sampler(x_init = x_init, callback_fn = callback_)
        plot_from_txt(f'{folder}/err_daz.txt',f'{folder}/err_daz')
