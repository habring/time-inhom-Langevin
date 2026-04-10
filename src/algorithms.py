import math
import torch as th
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


class DAZ(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        n: int
        k: int
        x_in: th.Tensor
        x_out: th.Tensor
        t_n: th.Tensor
        tau_n: th.Tensor
        df: th.Tensor
        pg: th.Tensor
        dF: th.Tensor
        z: th.Tensor

    def __init__(
        self,
        N: int,
        K: int,
        ts: th.Tensor,
        taus: th.Tensor,
        nabla_f: Callable,
        prox_g: Callable,
        zero_mean: bool = False,
    ) -> None:
        super().__init__()

        assert N > 0 and K > 0
        assert ts.ndim == 1 and ts.numel() == N
        assert taus.ndim == 1 and taus.numel() == N
        assert callable(nabla_f)
        assert callable(prox_g)
        assert isinstance(zero_mean, bool)

        self.N = N
        self.K = K
        self.ts = nn.Parameter(ts, requires_grad=False)
        self.taus = nn.Parameter(taus, requires_grad=False)
        self.nabla_f = nabla_f
        self.prox_g = prox_g
        self.zero_mean = zero_mean

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x = x_init.clone()

        for n, (t_n, tau_n) in enumerate(zip(self.ts, self.taus)):
            for k in range(self.K):
                z = th.randn_like(x)

                df = self.nabla_f(x)
                pg = self.prox_g(x, t_n)
                dF = df + (x - pg) / t_n

                x_ = x - tau_n * dF + th.sqrt(2 * tau_n) * z
                if self.zero_mean:
                    x_ -= x_.mean(dim=-1, keepdim=True)

                s = self.State(
                    n=n,
                    k=k,
                    x_in=x,
                    x_out=x_,
                    t_n=t_n,
                    tau_n=tau_n,
                    df=df,
                    pg=pg,
                    dF=dF,
                    z=z,
                )
                callback_fn(self, s)

                x = x_

        return x


class ULA(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        k: int
        x_in: th.Tensor
        x_out: th.Tensor
        tau: th.Tensor
        df: th.Tensor
        z: th.Tensor

    def __init__(
        self,
        K: int,
        tau: float,
        nabla_f: Callable,
        zero_mean: bool = False,
    ) -> None:
        super().__init__()

        assert K > 0
        assert tau > 0.0
        assert callable(nabla_f)
        assert isinstance(zero_mean, bool)

        self.K = K
        self.tau = tau
        self.nabla_f = nabla_f
        self.zero_mean = zero_mean

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x = x_init.clone()

        taus = self.tau * x_init.new_ones(self.K)

        for k, tau in enumerate(taus):
            z = th.randn_like(x)
            df = self.nabla_f(x)
            x_ = x - tau * df + th.sqrt(2 * tau) * z

            if self.zero_mean:
                x_ -= x_.mean(dim=-1, keepdim=True)

            s = self.State(k=k, x_in=x, x_out=x_, tau=tau, df=df, z=z)
            callback_fn(self, s)

            x = x_

        return x


import numpy as np
class GeneralAnnealing(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        n: int
        x_in: th.Tensor
        x_out: th.Tensor
        tau: th.Tensor
        t: th.Tensor
        df: th.Tensor
        z: th.Tensor
        running_mean: th.Tensor

    def __init__(
        self,
        times: th.Tensor,
        taus: th.Tensor,
        nabla_f: Callable,
        zero_mean: bool = False,
        burnin: int=0,
        reset=np.inf,
    ) -> None:
        super().__init__()

        self.times = nn.Parameter(times, requires_grad=False)
        self.taus = nn.Parameter(taus, requires_grad=False)
        self.nabla_f = nabla_f
        self.zero_mean = zero_mean
        self.burnin = burnin
        self.reset = reset

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x = x_init.clone()
        running_mean = th.zeros_like(x)
        last_reset = self.burnin

        for n, time in enumerate(self.times[:-1]):
            z = th.randn_like(x)
            df = self.nabla_f(x,self.taus[n])
            dt = (self.times[n+1]-self.times[n])
            x_ = x + dt * df + th.sqrt(2 * dt) * z
            if self.zero_mean:
                x_ -= x_.mean(dim=-1, keepdim=True)


            if n>self.burnin:
                running_mean = (running_mean*(n-last_reset-1) + x_) / (n-last_reset)
                if n-last_reset == self.reset:
                    last_reset = n
                    running_mean *= 0

            s = self.State(n=n, x_in=x, x_out=x_, tau=self.taus[n+1], t=self.times[n+1], df=df, z=z,running_mean=running_mean)
            callback_fn(self, s)
            x = x_.clone()



class AL(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        n: int
        k: int
        x_in: th.Tensor
        x_out: th.Tensor
        tau_n: th.Tensor
        df: th.Tensor
        z: th.Tensor

    def __init__(
        self,
        N: int,
        K: int,
        taus: th.Tensor,
        nabla_f: Callable,
        zero_mean: bool = False,
    ) -> None:
        super().__init__()

        assert N > 0 and K > 0
        assert taus.ndim == 1 and taus.numel() == N
        assert callable(nabla_f)
        assert isinstance(zero_mean, bool)

        self.N = N
        self.K = K
        self.taus = nn.Parameter(taus, requires_grad=False)
        self.nabla_f = nabla_f
        self.zero_mean = zero_mean

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x = x_init.clone()

        for n, tau_n in enumerate(self.taus):
            for k in range(self.K):
                z = th.randn_like(x)
                df = self.nabla_f(x)
                x_ = x - tau_n * df + th.sqrt(2 * tau_n) * z

                if self.zero_mean:
                    x_ -= x_.mean(dim=-1, keepdim=True)

                s = self.State(n=n, k=k, x_in=x, x_out=x_, tau_n=tau_n, df=df, z=z)
                callback_fn(self, s)

                x = x_.clone()

        return x


class GeomTemperedLangevin(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        k: int
        x_in: th.Tensor
        x_out: th.Tensor
        tau_k: th.Tensor
        lam_k: th.Tensor
        dU: th.Tensor
        dV: th.Tensor
        grad: th.Tensor
        z: th.Tensor

    def __init__(
        self,
        K: int,
        taus: th.Tensor,
        lambdas: th.Tensor,
        nabla_U: Callable,
        nabla_V: Callable,
    ) -> None:
        super().__init__()

        assert K > 0
        assert taus.ndim == 1 and taus.numel() == K
        assert lambdas.ndim == 1 and lambdas.numel() == K
        assert callable(nabla_U) and callable(nabla_V)

        self.K = K
        self.taus = nn.Parameter(taus, requires_grad=False)
        self.lambdas = nn.Parameter(lambdas, requires_grad=False)
        self.nabla_U = nabla_U
        self.nabla_V = nabla_V

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x = x_init.clone()

        for k, (tau_k, lam_k) in enumerate(zip(self.taus, self.lambdas)):
            z = th.randn_like(x)
            dU, dV = self.nabla_U(x), self.nabla_V(x)
            grad = (1 - lam_k) * dU + lam_k * dV
            x_ = x - tau_k * grad + th.sqrt(2 * tau_k) * z

            s = self.State(
                k=k,
                x_in=x,
                x_out=x_,
                tau_k=tau_k,
                lam_k=lam_k,
                dU=dU,
                dV=dV,
                grad=grad,
                z=z,
            )
            callback_fn(self, s)

            x = x_.clone()

        return x


class MYULA(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        k: int
        x_in: th.Tensor
        x_out: th.Tensor
        t: th.Tensor
        tau: th.Tensor
        df: th.Tensor
        pg: th.Tensor
        dF: th.Tensor
        z: th.Tensor

    def __init__(
        self,
        K: int,
        t: float,
        tau: float,
        nabla_f: Callable,
        prox_g: Callable,
        zero_mean: bool = False,
    ) -> None:
        super().__init__()

        assert K > 0 and t > 0.0 and tau > 0.0
        assert callable(nabla_f) and callable(prox_g)
        assert isinstance(zero_mean, bool)

        self.K = K
        self.t = t
        self.tau = tau
        self.nabla_f = nabla_f
        self.prox_g = prox_g
        self.zero_mean = zero_mean

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x = x_init.clone()

        ts = self.t * x_init.new_ones(self.K)
        taus = self.tau * x_init.new_ones(self.K)

        for k, (t, tau) in enumerate(zip(ts, taus)):
            z = th.randn_like(x)

            df = self.nabla_f(x)
            pg = self.prox_g(x, t)
            dF = df + (x - pg) / t

            x_ = x - tau * dF + th.sqrt(2 * tau) * z

            if self.zero_mean:
                x_ -= x_.mean(dim=-1, keepdim=True)

            s = self.State(
                k=k, x_in=x, x_out=x_, t=t, tau=tau, df=df, pg=pg, dF=dF, z=z
            )
            callback_fn(self, s)

            x = x_

        return x


class SKROCK(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        k: int
        s: int
        x_old: th.Tensor
        x_in: th.Tensor
        x_half: th.Tensor
        x_out: th.Tensor
        t: th.Tensor
        tau: th.Tensor
        mu_s: float
        nu_s: float
        k_s: float

    def __init__(
        self,
        K: int,
        nabla_f: Callable,
        prox_g: Callable,
        t: float,
        S: int,
        L_f: float,
        eta: float = 0.05,
        zero_mean: bool = False,
    ) -> None:
        super().__init__()

        assert K > 0
        assert callable(nabla_f)
        assert callable(prox_g)
        assert t > 0.0
        assert 3 <= S <= 15
        assert L_f >= 0.0
        assert eta > 0.0
        assert isinstance(zero_mean, bool)

        self.K = K
        self.nabla_f = nabla_f
        self.prox_g = prox_g
        self.t = t
        self.S = S
        self.L_f = L_f
        self.eta = eta
        self.zero_mean = zero_mean

        self.l_s = (S - 0.5) ** 2 * (2 - 4 / 3 * eta) - 1.5

        self.o_0 = 1 + eta / S**2
        self.o_1 = self._cheby(self.o_0, S) / self._d_cheby(self.o_0, S)

        self.mu_1 = self.o_1 / self.o_0
        self.nu_1 = 0.5 * S * self.o_1
        self.k_1 = S * self.o_1 / self.o_0

        self.tau = 0.9 * self.l_s / (L_f + 1 / t)

    def _cheby(self, x: float, s: int) -> float:
        if s == 0:
            return 1.0

        if s == 1:
            return x

        return 2 * x * self._cheby(x, s - 1) - self._cheby(x, s - 2)

    def _d_cheby(self, x: float, s: int) -> float:
        if s == 0:
            return 0.0

        if s == 1:
            return 1.0

        return (
            2 * self._cheby(x, s - 1)
            + 2 * x * self._d_cheby(x, s - 1)
            - self._d_cheby(x, s - 2)
        )

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x_old = x_init.clone()

        taus = self.tau * x_init.new_ones(self.K)
        ts = self.t * x_init.new_ones(self.K)

        for k, (tau, t) in enumerate(zip(taus, ts)):
            z = th.randn_like(x_old)

            x_half = x_old + self.nu_1 * math.sqrt(2 * tau) * z
            df = self.nabla_f(x_half)
            pg = self.prox_g(x_half, t)
            dF = df + (x_half - pg) / t

            x = x_old - self.mu_1 * tau * dF + self.k_1 * math.sqrt(2 * tau) * z

            for s in range(2, self.S + 1):
                mu_s = (
                    2
                    * self.o_1
                    * self._cheby(self.o_0, s - 1)
                    / self._cheby(self.o_0, s)
                )
                nu_s = (
                    2
                    * self.o_0
                    * self._cheby(self.o_0, s - 1)
                    / self._cheby(self.o_0, s)
                )

                k_s = 1 - nu_s

                df = self.nabla_f(x)
                pg = self.prox_g(x, t)
                dF = df + (x - pg) / t

                x_ = -mu_s * tau * dF + nu_s * x + k_s * x_old

                if self.zero_mean:
                    x_ -= x_.mean(dim=-1, keepdim=True)

                state = self.State(
                    k=k,
                    s=s,
                    x_old=x_old,
                    x_in=x,
                    x_half=x_half,
                    x_out=x_,
                    t=t,
                    tau=tau,
                    mu_s=mu_s,
                    nu_s=nu_s,
                    k_s=k_s,
                )
                callback_fn(self, state)

                x_old = x
                x = x_

            x_old = x

        return x


class DAZSKROCK(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        n: int
        x_in: th.Tensor
        x_out: th.Tensor
        t_n: th.Tensor

    def __init__(
        self,
        N: int,
        K: int,
        S: int,
        nabla_f: Callable,
        prox_g: Callable,
        ts: th.Tensor,
        L_f: float,
        eta: float = 0.05,
        zero_mean: bool = False,
    ) -> None:
        super().__init__()

        assert N > 0 and K > 0
        assert ts.ndim == 1 and ts.numel() == N
        assert S >= 3 and S <= 15
        assert callable(nabla_f)
        assert callable(prox_g)
        assert L_f >= 0.0
        assert eta > 0.0
        assert isinstance(zero_mean, bool)

        self.N = N
        self.K = K
        self.S = S
        self.ts = nn.Parameter(ts, requires_grad=False)
        self.nabla_f = nabla_f
        self.prox_g = prox_g
        self.L_f = L_f
        self.eta = eta
        self.zero_mean = zero_mean

    def forward(
        self, x_init: th.Tensor, callback_fn: lambda cfg, state: None
    ) -> th.Tensor:
        x = x_init.clone()

        for n, t_n in enumerate(self.ts):
            skrock = SKROCK(
                self.K,
                self.nabla_f,
                self.prox_g,
                t_n.item(),
                self.S,
                self.L_f,
                self.eta,
                self.zero_mean,
            ).to(x.device)

            x_ = skrock(x)

            s = self.State(n=n, x_in=x, x_out=x_, t_n=t_n)
            callback_fn(self, s)

            x = x_

        return x


class APGD(nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class State:
        k: int
        theta: float
        x_old: th.Tensor
        x_cur: th.Tensor
        x_bar: th.Tensor
        L: th.Tensor
        df: th.Tensor
        x_half: th.Tensor
        x_new: th.Tensor

    def __init__(
        self,
        K: int,
        f: Callable,
        nabla_f: Callable,
        g: Callable,
        prox_g: Callable,
        L_init: float = 1.0,
        theta_fn: Optional[Callable] = lambda k: k / (k + 3),
        backtrack: bool = False,
        J: int = 20,
        alpha: float = 1.01,
        beta: float = 0.9,
        gamma: float = 2.0,
        L_min: float = 1e-12,
        L_max: float = 1e12,
        early_stopping: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        assert K > 0
        assert callable(f) and callable(nabla_f)
        assert callable(g) and callable(prox_g)
        assert L_init > 0.0
        assert callable(theta_fn)
        assert isinstance(backtrack, bool)

        if backtrack:
            assert J > 0
            assert alpha > 1.0
            assert 0.0 < beta < 1.0
            assert gamma > 1.0
            assert L_min > 0.0
            assert L_max >= L_min

        assert isinstance(early_stopping, bool)
        if early_stopping:
            assert eps > 0.0

        self.K = K
        self.f = f
        self.nabla_f = nabla_f
        self.g = g
        self.prox_g = prox_g
        self.L_init = L_init
        self.theta_fn = theta_fn

        self.backtrack = backtrack
        self.J = J
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.L_min = L_min
        self.L_max = L_max

        self.early_stopping = early_stopping
        self.eps = eps

    def forward(
        self, x_init: th.Tensor, callback_fn: Callable = lambda cfg, state: None
    ) -> th.Tensor:
        x_cur = x_init.clone()
        x_old = x_init.clone()

        L = self.L_init * x_init.new_ones(x_init.shape[0]).view(
            -1, *[1] * (x_init.dim() - 1)
        )

        for k in range(self.K):
            theta = self.theta_fn(k)
            x_bar = x_cur + theta * (x_cur - x_old)

            if self.backtrack:
                df = self.nabla_f(x_bar)
                f = self.f(x_bar)
                for _ in range(self.J):
                    x_half = x_bar - df / L
                    x_new = self.prox_g(x_half, 1 / L)
                    f_new = self.f(x_new)

                    dx = x_new - x_bar

                    q = (
                        f
                        + (df * dx).flatten(start_dim=1).sum(-1).view_as(L)
                        + 0.5 * L * dx.pow(2).flatten(start_dim=1).sum(-1).view_as(L)
                    )

                    if th.any(crit := f_new > self.alpha * q):
                        L = th.where(crit, (self.gamma * L).clamp_max(self.L_max), L)
                    else:
                        L = (self.beta * L).clamp_min(self.L_min)
                        break
            else:
                df = self.nabla_f(x_bar)
                x_half = x_bar - df / L
                x_new = self.prox_g(x_half, 1 / L)

            if self.early_stopping:
                diff_x = (
                    (x_new - x_cur).pow(2).flatten(start_dim=1).sum(-1).sqrt().mean()
                )
                norm_x = x_cur.pow(2).flatten(start_dim=1).sum(-1).sqrt().mean()

                if th.all(diff_x / norm_x < self.eps):
                    break

            s = self.State(
                k=k,
                theta=theta,
                x_old=x_old,
                x_cur=x_cur,
                x_bar=x_bar,
                L=L,
                df=df,
                x_half=x_half,
                x_new=x_new,
            )
            callback_fn(self, s)

            x_old = x_cur.clone()
            x_cur = x_new.clone()

        return x_cur
