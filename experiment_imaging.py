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

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.physics import Downsampling
from deepinv.utils import load_example
from tqdm import tqdm  # to visualize progress

num_runs = 1

with th.no_grad():


    for run in range(num_runs):

        # generate random means
        th.manual_seed(0)
        device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        print(device)

        folder_ = f'results/imaging/inpainting'
        Path(folder_).mkdir(parents=True,exist_ok=True)

        # plt.rcParams.update({
        #     "text.usetex": False,
        #     "font.family": "Helvetica",
        #     "font.size": 20,
        #     'text.latex.preamble': r'\usepackage{amsfonts}'
        # })
        rate = 0.8
        img_size = 1024
        x_true = load_example("celeba_example.jpg", img_size=img_size).requires_grad_(False).to(device)
        physics = Downsampling(filter = "gaussian", factor=4).to(device)
        x_true = physics(x_true)
        img_size = x_true.shape[0]

        try:
            mask = th.load(f'{folder_}/mask09.th')
            assert mask.shape[-1] == img_size
        except:
            mask = th.rand_like(x_true[:,0,...]).to(device)[None,...]
            mask = 1.0*(mask>rate)
            th.save(mask,f'{folder_}/mask09.th')
        lam = 100
        burnin = 100
        x_true = (x_true-x_true.min()) / (x_true.max()-x_true.min())
        x = x_true.clone().requires_grad_(False).to(device)
        x_init = x_true*mask # th.randn_like(x).to(device).requires_grad_(False).to(device)


        maxit = 30000
        check_iter = 0
        save_sample = np.arange(0,maxit,10)
        show_plot=False

        methods = {
                    'ULA': True,
                    'dilation':True,
                    'tempering':True,
                    'diffusion':True,
        }

        def callback(alg, state,write_file,dir):
            if check_iter>0 and state.n % check_iter==0:
                fig,ax = plt.subplots(1,4,figsize = (20,7))
                x_plt = (state.x_in-state.x_in.min()) / (state.x_in.max() - state.x_in.min())
                mean_plt = (state.running_mean-state.running_mean.min()) / (state.running_mean.max() - state.running_mean.min())
                ax[0].imshow(th.permute((mask*x_true).squeeze(),(1,2,0)).cpu())
                ax[1].imshow(th.permute(x_plt.squeeze(),(1,2,0)).cpu())
                ax[2].imshow(th.permute(mean_plt.squeeze(),(1,2,0)).cpu())
                ax[3].imshow(th.permute(x_true.squeeze(),(1,2,0)).cpu())
                plt.title(f'tau = {state.tau}, iter = {state.n}')
                plt.show()

            if state.n in save_sample:
                Path(dir).mkdir(exist_ok=True,parents=True)
                th.save(state.x_in, f'{dir}/sample_iter_{state.n}.th')
            return
        
        model = pot.DSM_score()

        sigma_final = th.tensor(0.01)
        step = 1e-3
        times = th.Tensor(np.arange(0,step*maxit,step))

        folder_ = f'{folder_}/step_{step}'

        if methods['ULA'] == True:
            print('ULA')
            def callback_(algo,state):
                return callback(alg=algo,state=state, write_file=f'{folder_}/err_ULA.txt',dir=f'{folder_}/ULA_samples')

            def nabla_f(x,tau):
                return model.score(x,sigma_final) - lam*mask*(x-x_true)
            
            taus = times*0. + sigma_final
            sampler = algo.GeneralAnnealing(times=times,taus=taus,
                    nabla_f=nabla_f)
            sample = sampler(x_init = x_init, callback_fn = callback_)

        for T in [100*step,200*step,500*step,1000*step,2000*step]:
            folder = f'{folder_}/T_{T}'
            Path(folder).mkdir(parents=True,exist_ok=True)

            tau = lambda t: sigma_final + th.exp(-t/T)*(1-sigma_final-1e-3)

            if methods['diffusion'] == True:
                taus = tau(times)
                
                def callback_(algo,state):
                    return callback(alg=algo,state=state, write_file=f'{folder}/err_diffusion.txt',dir=f'{folder}/diffusion_samples')
                
                def nabla_f(x,tau):
                    return model.score(x,tau) - lam*mask*(x-x_true)
                
                sampler = algo.GeneralAnnealing(times=times,taus=taus,
                        nabla_f=nabla_f,burnin=burnin)
                
                print('Diffusion')
                sample = sampler(x_init = x_init, callback_fn = callback_)
                print('Done')


            if methods['tempering'] == True:
                taus = tau(times)
                print('Tempering')
                def callback_(algo,state):
                    return callback(alg=algo,state=state, write_file=f'{folder}/err_tempering.txt',dir=f'{folder}/tempering_samples')
                def nabla_f(x,tau):
                    return (1-tau)*model.score(x/np.sqrt(1-tau),sigma_final) - tau*x - lam*mask*(x-x_true)
                
                sampler = algo.GeneralAnnealing(times=times,taus=taus,
                        nabla_f=nabla_f)
                sample = sampler(x_init = x_init, callback_fn = callback_)


            if methods['dilation'] == True:
                times = th.tensor(times)
                taus = tau(times)
                print('dilation')

                def callback_(algo,state):
                    return callback(alg=algo,state=state, write_file=f'{folder}/err_dilation.txt',dir=f'{folder}/dilation_samples')
                def nabla_f(x,tau):
                    return np.sqrt(1-tau)*model.score(x/np.sqrt(1-tau),sigma_final) - lam*mask*(x-x_true) 
                
                sampler = algo.GeneralAnnealing(times=times,taus=taus,
                        nabla_f=nabla_f)
                sample = sampler(x_init = x_init, callback_fn = callback_)