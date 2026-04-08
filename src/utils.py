from pathlib import Path
import tempfile
from cycler import cycler
from matplotlib import ticker
from matplotlib.colors import LogNorm
import torch as th
import torch.nn as nn
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Self, Union
from PIL.Image import Image
# from pathos.multiprocessing import ProcessingPool as Pool
import subprocess

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

# proximal mapping of x\mapsto a*|x-z|^2/2
def prox_sq_l2(x, z=0,t=1,a=1):
    return (t*a*z + x)/(1+t*a)

def APGD_prox(nabla_f: callable, f: callable, t: th.float64, u:th.Tensor, x_init: th.Tensor, num_iter: int=200, J: int=50, beta: th.float64=0.8, gamma: th.float64=1.5, tol: th.float64=1e-7,L: th.float64=1) -> th.Tensor:
    x = x_init.clone()
    x_old = x_init.clone()
    x_old = x.clone()
    x_new = x.clone()
    for k in range(num_iter):
        x_ = x+(x-x_old)/np.sqrt(2)

        nabla_f_ = t*nabla_f(x_)
        x_tmp = x_ - nabla_f_/L
        x_new = prox_sq_l2(x_tmp,u,t=1/L,a=1)
        grad = t*nabla_f(x_new) + (x_new - u)

        if th.abs(grad).sum() < tol:
            print(grad)
            break
        
        x_old = x
        x = x_new
    
    return x_new

def APGD_prox_multi_init(nabla_f: callable, f: callable, t: th.float64, u:th.Tensor, x_init: th.Tensor, num_iter: int=200, tol: th.float64=1e-7,L: th.float64=1) -> th.Tensor:
    num_init = x_init.shape[0]+1
    num_samples = u.shape[0]
    uu = th.repeat_interleave(u,num_init,dim=0)
    x_init = th.cat([x_init,th.zeros([1,*x_init.shape[1:]]).to(x_init.device)])
    x = th.cat([x_init]*num_samples,dim=0)
    x[::num_init,...] = u[:]
    x_old = x.clone()
    x_new = x.clone()
    for k in range(num_iter):
        x_ = x+(x-x_old)/np.sqrt(2)

        nabla_f_ = t*nabla_f(x_)
        x_tmp = x_ - nabla_f_/L
        x_new = prox_sq_l2(x_tmp,uu,t=1/L,a=1)
        grad = t*nabla_f(x_new) + (x_new - uu)
        
        if th.abs(grad).sum() < tol:
            print(grad)
            break
        
        x_old = x
        x = x_new
    
    f_vals = t*f(x) + 0.5*((x - uu)**2).sum(dim=-1)
    f_vals = f_vals.squeeze()
    min_indices = f_vals.view(-1,num_init).argmin(dim=1)
    ran = th.arange(0,len(f_vals),num_init).to(x.device)
    x = x[ran+min_indices]

    return x, nabla_f(x)


_sentinel = object()
def KL(P,Q,bins=_sentinel):
    # assert th.abs(((th.abs(Q)<1e-9)*P)).sum()<1e-6
    # make sure P<<Q:

    P_normalized = th.zeros(P.shape).to(P.device)
    Q_normalized = th.zeros(P.shape).to(Q.device)
    if bins is not _sentinel:
        dx = bins[1:]-bins[:-1]
        P_normalized = P * dx
        Q_normalized = Q * dx
            
    # make sure P<<Q
    Q_normalized += 1e-9


    P_normalized /= P_normalized.sum()
    Q_normalized /= Q_normalized.sum()

    return (th.log((th.maximum(P_normalized,1e-9*th.ones_like(P))/Q_normalized))*P_normalized).sum()

_sentinel = object()
def TV(P,Q,bins=_sentinel):
    # assert th.abs(((th.abs(Q)<1e-9)*P)).sum()<1e-6
    # make sure P<<Q:

    P_normalized = th.zeros(P.shape).to(P.device)
    Q_normalized = th.zeros(P.shape).to(Q.device)
    if bins is not _sentinel:
        dx = bins[1:]-bins[:-1]
        P_normalized = P * dx
        Q_normalized = Q * dx
            

    P_normalized /= P_normalized.sum()
    Q_normalized /= Q_normalized.sum()

    return (th.abs(P_normalized-Q_normalized)).sum()



def KL2D(P,Q,bins=_sentinel):
    # assert th.abs(((th.abs(Q)<1e-9)*P)).sum()<1e-6
    # make sure P<<Q:

    P_normalized = th.zeros(P.shape).to(P.device)
    Q_normalized = th.zeros(P.shape).to(Q.device)
    if bins is not _sentinel:
        dx = bins[0][1:]-bins[0][:-1]
        dy = bins[1][1:]-bins[1][:-1]
        dxdy = th.outer(dx,dy)
        P_normalized = P * dxdy
        Q_normalized = Q * dxdy

            
    # make sure P<<Q
    Q_normalized += 1e-9

    P_normalized /= P_normalized.sum()
    Q_normalized /= Q_normalized.sum()

    return (th.log((th.maximum(P_normalized,1e-9*th.ones_like(P))/Q_normalized))*P_normalized).sum()

_sentinel = object()
def TV2D(P,Q,bins=_sentinel):
    # assert th.abs(((th.abs(Q)<1e-9)*P)).sum()<1e-6
    # make sure P<<Q:

    P_normalized = th.zeros(P.shape).to(P.device)
    Q_normalized = th.zeros(P.shape).to(Q.device)
    if bins is not _sentinel:
        dx = bins[0][1:]-bins[0][:-1]
        dy = bins[1][1:]-bins[1][:-1]
        dxdy = th.outer(dx,dy)
        P_normalized = P * dxdy
        Q_normalized = Q * dxdy
            

    P_normalized /= P_normalized.sum()
    Q_normalized /= Q_normalized.sum()

    return (th.abs(P_normalized-Q_normalized)).sum()






def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@th.no_grad()
def tensor_info(
    x: Union[
        th.Tensor,
        list[th.Tensor],
        tuple[str, th.Tensor],
        list[tuple[str, th.Tensor]],
    ],
    precision: int = 6,
) -> str:
    if isinstance(x, th.Tensor):
        x = [("0001", x)]

    elif isinstance(x, tuple):
        if len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], th.Tensor):
            x = [x]
        else:
            raise ValueError(f"Invalid tuple format: {x}")

    elif isinstance(x, list):
        if len(x) == 0:
            return ""

        first = x[0]

        if isinstance(first, th.Tensor):
            if not all(isinstance(t, th.Tensor) for t in x):
                raise ValueError(f"Invalid list of tensors format: {x}")

            x = [(f"{i + 1:04d}", t) for i, t in enumerate(x)]

        if isinstance(first, tuple):
            if not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], th.Tensor)
                for item in x
            ):
                raise ValueError(f"Invalid list of tuples format: {x}")

    else:
        raise ValueError(f"Unsupported input type: {type(x)}")

    def sign(val: float) -> str:
        return "+" if val >= 0 else "-"

    def stats(x: th.Tensor) -> list[float]:
        x = x.double()

        quantiles = th.quantile(x, th.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to(x))
        quantiles = [q.item() for q in quantiles]

        iqr = quantiles[3] - quantiles[1]

        if x.numel() <= 1:
            mean = th.mean(x)
            std = th.tensor(0.0)
        else:
            std, mean = th.std_mean(x)

        return *quantiles, iqr, mean.item(), std.item()

    headers = [
        "name",
        "shape",
        "dtype",
        "device",
        "req. ∇",
        "min",
        "q=.25",
        "median",
        "q=.75",
        "max",
        "iqr",
        "μ",
        "σ",
    ]
    col_widths = [max(len(header), 6) for header in headers]

    rows = []

    for name, t in x:
        shape = str(tuple(t.shape))
        dtype = str(t.dtype)
        device = str(t.device)
        requires_grad = "+" if t.requires_grad else "-"

        if t.numel() == 0 or t.dtype == th.bool:
            min_str = "-"
            q25_str = "-"
            q50_str = "-"
            q75_str = "-"
            max_str = "-"
            iqr_str = "-"
            mean_str = "-"
            std_str = "-"

        elif t.is_complex():
            r_min, r_q25, r_q50, r_q75, r_max, r_iqr, r_mean, r_std = stats(t.real)
            i_min, i_q25, i_q50, i_q75, i_max, i_iqr, i_mean, i_std = stats(t.imag)

            c_prec = max(2, precision // 3)
            min_str = f"{r_min:.{c_prec}g}"
            min_str += f"{sign(i_min)}j{abs(i_min):.{c_prec}g}"

            q25_str = f"{r_q25:.{c_prec}g}"
            q25_str += f"{sign(i_q25)}j{abs(i_q25):.{c_prec}g}"

            q50_str = f"{r_q50:.{c_prec}g}"
            q50_str += f"{sign(i_q50)}j{abs(i_q50):.{c_prec}g}"

            q75_str = f"{r_q75:.{c_prec}g}"
            q75_str += f"{sign(i_q75)}j{abs(i_q75):.{c_prec}g}"

            max_str = f"{r_max:.{c_prec}g}"
            max_str += f"{sign(i_max)}j{abs(i_max):.{c_prec}g}"

            iqr_str = f"{r_iqr:.{c_prec}g}"
            iqr_str += f"{sign(i_iqr)}j{abs(i_iqr):.{c_prec}g}"

            mean_str = f"{r_mean:.{c_prec}g}"
            mean_str += f"{sign(i_mean)}j{abs(i_mean):.{c_prec}g}"

            std_str = f"{r_std:.{c_prec}g}"
            std_str += f"{sign(i_std)}j{abs(i_std):.{c_prec}g}"
        else:
            minimum, q25, q50, q75, maximum, iqr, mean, std = stats(t)
            min_str = f"{minimum:.{precision}g}"
            q25_str = f"{q25:.{precision}g}"
            q50_str = f"{q50:.{precision}g}"
            q75_str = f"{q75:.{precision}g}"
            max_str = f"{maximum:.{precision}g}"
            iqr_str = f"{iqr:.{precision}g}"
            mean_str = f"{mean:.{precision}g}"
            std_str = f"{std:.{precision}g}"

        row = [
            name,
            shape,
            dtype,
            device,
            requires_grad,
            min_str,
            q25_str,
            q50_str,
            q75_str,
            max_str,
            iqr_str,
            mean_str,
            std_str,
        ]
        rows.append(row)
        col_widths = [max(len(str(val)), w) for val, w in zip(row, col_widths)]

    header_str = " | ".join(header.center(w) for header, w in zip(headers, col_widths))

    final_rows = []
    final_rows.append("=" * len(header_str))
    final_rows.append(header_str)
    final_rows.append("-" * len(header_str))

    for row in rows:
        final_rows.append(
            " | ".join(str(val).rjust(w) for val, w in zip(row, col_widths))
        )

    final_rows.append("=" * len(header_str))

    return "\n".join(final_rows)


def print_tensor_info(x: list[th.Tensor], precision: int = 6) -> None:
    print(tensor_info(x, precision))


class Timer:
    def __init__(self) -> None:
        self.start_time = 0
        self.dt = 0.0

    def __enter__(self) -> Self:
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, *args) -> None:
        self.dt = time.perf_counter_ns() - self.start_time


def to_np(x: th.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def batch_to_images(
    x: th.Tensor,
    clamp: bool = True,
    scale: bool = False,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> list[Image]:
    s = T.Lambda(lambda t: (t - t.min()) / (t.max() - t.min()))
    cl = T.Lambda(lambda t: t.clamp(vmin, vmax))

    grayscale = x.shape[1] == 1
    c = T.Lambda(lambda t: plt.get_cmap(cmap)(t.squeeze()))

    transform = T.Compose(
        [
            cl if clamp else T.Identity(),
            s if scale else T.Identity(),
            c if grayscale else T.Identity(),
            T.ToPILImage(),
        ]
    )

    imgs = [transform(t) for t in x.detach().cpu()]
    return imgs


def contours(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    log_scaling: bool = False,
    outline: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    if ax is None:
        ax = plt.gca()

    if log_scaling:
        vmin, vmax = kwargs.pop("vmin", None), kwargs.pop("vmax", None)

    im = ax.imshow(
        z,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        norm=LogNorm(vmin=vmin, vmax=vmax) if log_scaling else None,
        **kwargs,
    )

    if outline is not None:
        ax.contour(
            x,
            y,
            z,
            colors=outline,
            locator=ticker.LogLocator() if log_scaling else None,
            linewidths=0.5,
            linestyles="solid",
        )

    return im


# def create_video(frame_fn, frames, output_path, fps, W, H, dpi):
#     with tempfile.TemporaryDirectory(dir=Path("/dev/shm")) as dir:

#         def visualize_and_save(frame):
#             fig = frame_fn(frame)
#             # fig.set_size_inches(W, H)
#             # fig.set_dpi(dpi)
#             fig.savefig(
#                 f"{dir}/frame_{frame[0]:06d}.png",
#                 dpi=300,
#                 bbox_inches="tight",
#                 # pad_inches=0.0,
#             )
#             plt.close(fig)

#         pool = Pool()
#         pool.map(visualize_and_save, frames)
#         pool.close()
#         pool.join()

#         subprocess.run(
#             [
#                 "ffmpeg",
#                 "-y",
#                 "-framerate",
#                 str(fps),
#                 "-i",
#                 f"{dir}/frame_%06d.png",
#                 "-vf",
#                 "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=1920:-2",
#                 "-c:v",
#                 "libx264",
#                 "-preset",
#                 "slow",
#                 "-crf",
#                 "18",
#                 "-pix_fmt",
#                 "yuv420p",
#                 output_path,
#             ]
#         )


METHODS = [
    "ila",
    "ula",
    "ses",
    "em",
    "oba",
    "baoab",
    "nila",
    "ilav2",
    "nilav2",
    "lm",
    "cheng",
    "abo",
    "aob",
    "bao",
    "boa",
    "oab",
    "aboba",
    "aobab",
    "boaob",
    "oabab",
    "obabo",
    "hmc",
    "skrock",
    "ground_truth",
]

METHOD_LABELS = {
    "ila": "ILA",
    "nila": "NILA",
    "ilav2": "ILAv2",
    "nilav2": "NILAv2",
    "ula": "ULA",
    "lm": "LM",
    "cheng": "Cheng",
    "ses": "SES",
    "em": "EM",
    "abo": "ABO",
    "aob": "AOB",
    "bao": "BAO",
    "boa": "BOA",
    "oab": "OAB",
    "oba": "OBA",
    "aboba": "ABOBA",
    "aobab": "AOBAB",
    "baoab": "BAOAB",
    "boaob": "BOAOB",
    "oabab": "OABAB",
    "obabo": "OBABO",
    "hmc": "HMC",
    "skrock": "SK-ROCK",
    "ground_truth": "GT",
}

COLORS = [
    "#0072B2",
    "#009E73",
    "#FF7F00",
    "#E41A1C",
    "#E69F00",
    "#984EA3",
    "#A6761D",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
    "#E41A1C",
    "#4DAF4A",
    "#377EB8",
    "#FFFFB3",
    "#8DD3C7",
    "#BEBADA",
    "#FB8072",
    "#80B1D3",
    "#FDB462",
    "#999999",
    "#B3DE69",
    "#FCCDE5",
    "#BC80BD",
    "#000000",
]

METHOD_COLORS = {}
for METHOD, COLOR in zip(METHODS, COLORS):
    METHOD_COLORS[METHOD] = COLOR


CM = 1 / 2.54

HALF_FIG_WIDTH = 7.5 * CM
HALF_FIG_HEIGHT = HALF_FIG_WIDTH
HALF_FIGSIZE = (HALF_FIG_WIDTH, HALF_FIG_HEIGHT)

FULL_FIG_WIDTH = 15 * CM
FULL_FIG_HEIGHT = 0.5 * FULL_FIG_WIDTH
FULL_FIGSIZE = (FULL_FIG_WIDTH, FULL_FIG_HEIGHT)

ALONE_FIG_WIDTH = 12 * CM
ALONE_FIG_HEIGHT = 0.5 * FULL_FIG_WIDTH
ALONE_FIGSIZE = (ALONE_FIG_WIDTH, ALONE_FIG_HEIGHT)

THIRD_FIG_WIDTH = 5 * CM
THIRD_FIG_HEIGHT = THIRD_FIG_WIDTH
THIRD_FIGSIZE = (THIRD_FIG_WIDTH, THIRD_FIG_HEIGHT)


def setup_mpl():
    plt.style.use(
        "https://gist.githubusercontent.com/alfa-98/0804703f872b7df6d853fdb17637ba09/raw/alfa-98.mplstyle"
    )
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = "small"
    plt.rcParams["axes.labelsize"] = "small"
    plt.rcParams["xtick.labelsize"] = "small"
    plt.rcParams["ytick.labelsize"] = "small"
    plt.rcParams["figure.dpi"] = 300

    plt.rcParams["legend.handlelength"] = 1
    plt.rcParams["legend.handleheight"] = 0.5
    plt.rcParams["legend.labelspacing"] = 0.5
    plt.rcParams["legend.handletextpad"] = 0.8
    plt.rcParams["legend.fontsize"] = "x-small"

    plt.rcParams["image.cmap"] = "plasma"

    plt.rc("axes", prop_cycle=cycler(color=COLORS))
