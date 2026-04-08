import math
import torch as th
import torch.nn as nn
import numpy as np
import deepinv as dinv


device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


# GMM diffusion density; for tau=0, basic GMM; for tau>0, the density of VP diffusion initialized at the GMM. GMM components can have any diagonal covariance (otherwise diffusion not explicit)
class GMM_diffusion(nn.Module):
    def __init__(self, means: th.Tensor, sigmas: th.Tensor, mixture_weights: th.Tensor) -> None:
        super().__init__()

        self.d = means.shape[-1]
        self.means = means # shape: num of means, dimension
        self.sigmas = sigmas # shape: num of means, dimension
        self.weights = mixture_weights

    def forward(self, x: th.Tensor, tau: th.float64) -> th.Tensor:
        density = 0
        for k in range(len(self.weights)):
            mean = np.sqrt(1-tau)*self.means[k,...]
            var = (1-tau)*self.sigmas[k,...]**2 + tau
            diff = x - mean
            mahalanobis = th.exp(-0.5 * (diff**2/var).sum(dim=-1))
            mahalanobis *= self.weights[k]/th.sqrt((2*np.pi)**self.d*th.prod(var))
            density += mahalanobis
        return density
    
    def score(self, x: th.Tensor, tau: th.float64) -> th.Tensor:
        density = 0
        inner_score = 0

        for k in range(len(self.weights)):
            mean = np.sqrt(1-tau)*self.means[k]
            var = (1-tau)*self.sigmas[k,...]**2 + tau
            diff = x - mean
            mahalanobis = th.exp(-0.5 * (diff**2/var).sum(dim=-1))
            mahalanobis *= self.weights[k]/th.sqrt((2*np.pi)**self.d*th.prod(var))
            density += mahalanobis
            inner_score -= mahalanobis[...,None]*diff/var

        return inner_score/(th.maximum(density,th.ones_like(density)*1e-9)[...,None])
    

    def sample(self, n: int,tau: th.float64) -> th.Tensor:

        k = th.multinomial(self.weights,n, replacement=True)

        # sample from the component
        mean = self.means[k]
        std = self.sigmas[k]

        sample = std*th.randn(n,self.d,dtype=self.means.dtype).to(self.means[0].device) + mean

        return sample


class DSM_score(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = dinv.models.DiffUNet(large_model=False).to(device)
    
    def score(self, x: th.Tensor, tau: th.float64) -> th.Tensor:
        
        # diffusion model acts on images in [0,1]
        x_denoised = self.model(x,th.sqrt(tau/(1-tau)))
        score = - (x-x_denoised)/tau
        
        return score
    

def test_score_gmm(fun, score, n):
    
    d = fun.d
    x = th.randn(n,d)
    dir = th.randn(n,d)
    dir /= np.sqrt((dir**2).sum())

    t = 1e-4

    v1 = (th.log(fun(x+t*dir,0)) - th.log(fun(x,0)))/t
    v2 = (score(x,0)*dir).sum()

    print('Error')
    print(((v1-v2)**2).sum())



class GMM(nn.Module):
    def __init__(self, means: th.Tensor, covs: list, mixture_weights: th.Tensor) -> None:
        super().__init__()

        self.d = len(means[0])
        self.means = nn.Parameter(means)
        self.cov_trils = []
        for c in covs:
            self.cov_trils.append(th.linalg.cholesky(c))
            self.dets.append(th.prod(th.diag(self.cov_trils[-1])))**2


    def forward(self, x: th.Tensor) -> th.Tensor:
        density = 0
        for k in range(self.means):
            diff = x - self.mean[k]
            mahalanobis = th.exp(-0.5 * (diff * th.cholesky_solve(diff, self.cov_trils[k])).sum(-1))
            mahalanobis *= self.weights[k]/th.sqrt((2*np.pi)**self.d*self.dets(k))
            density += mahalanobis

        return density