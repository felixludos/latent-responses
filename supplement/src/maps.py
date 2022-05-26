from typing import Union, Optional, TypeVar, Generic, Dict, List

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions as distrib
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from .responses import Autoencoder
from .utils import plot_imgs


def push_forward(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], fn_name: str,
                 num_samples: Optional[int] = None, batch_size: int = 128, shuffle: bool = False,
                 device: Optional[str] = None, pbar: Optional[tqdm] = None) -> torch.Tensor:
    if num_samples is None:
        num_samples = len(data)
    else:
        num_samples = min(num_samples, len(data))
    batches = iter((torch.randperm(len(data)) if shuffle else torch.arange(len(data))).split(batch_size))

    outs = []
    N = 0

    fn = getattr(autoencoder, fn_name)

    if pbar is not None:
        pbar = pbar(total=num_samples)
    while N < num_samples:
        batch = next(batches)
        X = data[batch]
        if isinstance(X, tuple):
            X = X[0]
        out = fn(X.to(device))
        if isinstance(out, distrib.Distribution):
            out = out.mean
        outs.append(out.cpu())
        N += len(batch)
        if pbar is not None:
            pbar.update(len(batch))

    if pbar is not None:
        pbar.close()
    return torch.cat(outs)



def collect_posterior_means(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], shuffle=True, **kwargs):
    return push_forward(autoencoder, data, 'encode', shuffle=shuffle, **kwargs)


def collect_reconstructions(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], **kwargs):
    return push_forward(autoencoder, data, 'reconstruct', **kwargs)


def collect_responses(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], **kwargs):
    return push_forward(autoencoder, data, 'response', shuffle=False, **kwargs)



def generate_2d_latent_map(xidx, yidx, base, n=64, r=2, h=None, w=None, extent=None):
    if extent is None:
        extent = [-r, r, -r, r]
    xmin, xmax, ymin, ymax = extent
    if h is None:
        h = n
    if w is None:
        w = n

    cx, cy = torch.meshgrid(torch.linspace(xmin, xmax, h), torch.linspace(ymin, ymax, w))
    cx = cx.reshape(-1)
    cy = cy.reshape(-1)

    vecs = base.cpu().view(1, -1).expand(len(cx), -1).contiguous()
    vecs[:, xidx] = cx
    vecs[:, yidx] = cy
    return vecs.view(h, w, -1)



def collect_response_map(autoencoder: Autoencoder, zmap: torch.Tensor, batch_size: int = 128,
                 device: Optional[str] = None, pbar: Optional[tqdm] = None) -> torch.Tensor:
    '''
    Compute the response map for a given set of inputs `X`.
    :param autoencoder: used for computing the response (decode + encode)
    :param zmap: input latent sample map (H, W, latent_dim)
    :param batch_size: max batch size
    :param device: optional device
    :param pbar: optional progress bar
    :return: response map (H, W, latent_dim)
    '''

    H, W, _ = zmap.shape
    responses = collect_responses(autoencoder, zmap.view(H*W, -1), batch_size=batch_size, device=device, pbar=pbar)
    return responses.view(H, W, -1)


def response_map_2d(autoencoder: Autoencoder, xidx, yidx, base=None, n=64, r=2, batch_size: int = 128,
                    device: Optional[str] = None, pbar: Optional[tqdm] = None, **kwargs):

    if base is None:
        base = autoencoder.sample_prior(1)

    zmap = generate_2d_latent_map(xidx, yidx, base, n=n, r=r, **kwargs)
    rmap = collect_response_map(autoencoder, zmap, batch_size, device, pbar)
    umap = rmap.sub(zmap)[..., [xidx, yidx]]
    return umap



def compute_mean_curvature_2d(umap):
    nmap = umap.div(umap.norm(dim=-1, keepdim=True))
    cmap = - 0.5 * compute_divergence_2d(nmap)
    return cmap



def compute_divergence_2d(deltas: torch.Tensor) -> torch.Tensor:
    '''
    Numerically estimates the divergence by finite differencing of a given 2D grid.

    :param deltas: grid of 2D vectors (H, W, 2)
    :return: divergences (H, W)
    '''
    deltas = deltas.cpu().detach().numpy()
    divx, divy = np.gradient(deltas, axis=[0,1])
    divM = divx[...,0] + divy[...,1]
    return torch.as_tensor(divM)



def plot_map(grid, rescale=True, fgax=None, aspect=None, cmap='viridis', colorbar=False, **kwargs):
    if fgax is not None:
        fg, ax = fgax
        plt.sca(ax)
    else:
        ax = None

    if aspect is None:
        aspect = grid.size(1) / grid.size(0)

    axvals = np.linspace(grid.min().item(), grid.max().item(), 9)
    if rescale:
        axvals = np.concatenate([np.linspace(grid.min().item(), 0, 5), np.linspace(0, grid.max().item(), 5)[1:]])
        grid = rescale_map_for_viz(grid.clone())

    out = plt.imshow(grid.detach().cpu().numpy().T[::-1], cmap=cmap, aspect=aspect, **kwargs)
    if colorbar:
        cbar = plt.colorbar(ax=ax)
        axlbls = [f'{v.item():.2f}' for v in axvals]
        cbar.set_ticks(np.linspace(-1, 1, 9))
        cbar.set_ticklabels(axlbls)
    return out


def plot_posterior_2d(Z, xidx=None, yidx=None, fgax=None, bins=32, r=2, extent=None,
                      cmap='Reds', interpolation = "gaussian", **kwargs):
    if extent is None:
        extent = [-r, r, -r, r]

    if isinstance(Z, distrib.Distribution):
        Z = Z.sample()
    if Z.size(1) > 2:
        assert xidx is not None and yidx is not None, 'Must specify xidx and yidx'
        Z = Z[..., [xidx, yidx]]
    assert Z.size(1) == 2, 'Z must have 2 dimensions'

    if fgax is not None:
        fg, ax = fgax
        plt.sca(ax)

    hist, *other = np.histogram2d(*Z.cpu().t().numpy(), bins=bins, normed=True,
                                  range = np.array([[extent[0], extent[1]], [extent[2], extent[3]]]))
                                  # range=torch.stack([mn, mx]).t().cpu().numpy())
    return plt.imshow(hist.T[::-1], cmap=cmap, interpolation=interpolation, extent=extent, **kwargs)


def plot_recs(autoencoder: Autoencoder, xidx, yidx, base=None, n=64, r=2, extent=None, batch_size: int = 128,
                    device: Optional[str] = None, pbar: Optional[tqdm] = None, **kwargs):
    if extent is None:
        extent = [-r, r, -r, r]
    zgrid = generate_2d_latent_map(xidx, yidx, base=base, n=n, r=r, extent=extent)
    H, W, _ = zgrid.size()

    recs = push_forward(autoencoder, zgrid.view(H*W, -1), 'decode', device=device, batch_size=batch_size, pbar=pbar)
    _, c, h, w = recs.size()

    recs = torch.as_tensor(recs.view(H, W, c, h, w).permute(1,0,2,3,4).detach().cpu().numpy()[::-1].copy())
    return plot_imgs(recs.view(H*W, c, h, w), **kwargs)



def rescale_map_for_viz(dmap):
    vmap = dmap.clone()
    sel = vmap > 0
    if sel.sum() > 0:
        vmap[sel] /= vmap[sel].max().abs()
    sel = vmap < 0
    if sel.sum()>0:
        vmap[sel] /= vmap[sel].min().abs()
    return vmap






