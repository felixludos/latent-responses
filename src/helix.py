import sys, os, time, shutil, random
from typing import Union
import numpy as np
import torch
from torch import nn
from torch import distributions as distrib
from torch.nn import functional as F

from .utils import make_MLP
from .responses import Autoencoder



def helix_labels(N: int, strands: Union[None, int] = None, gen: torch.Generator = None) -> torch.Tensor:
	if strands is None:
		strands = 2
	z = torch.rand(N, generator=gen).mul(2).sub(1)
	n = torch.randint(strands, size=(N,), generator=gen)
	return torch.stack([z, n], -1)



def helix_observations(Y: Union[int, torch.Tensor], noise_std: Union[None, float] = 0.1, strands: Union[None, int] = 2,
                       w: float = 1., Rx: float = 1., Ry: float = 1., Rz: float = 1.,
                       gen: torch.Generator = None) -> torch.Tensor:
	if isinstance(Y, int):
		Y = helix_labels(Y, strands=strands, gen=gen)
	if strands is None:
		strands = Y.size(-1)
	z = Y.narrow(-1,0,1)
	n = Y.narrow(-1,1,1)
	theta = z.mul(w).add(n.div(strands)*2).mul(np.pi)
	amp = torch.tensor([Rx, Ry, Rz]).float().to(Y.device)
	X = amp.unsqueeze(0) * torch.cat([theta.cos(), theta.sin(), z],-1)
	if noise_std is not None:
		X += noise_std * torch.randn_like(X)
	return X



_standard_normal = distrib.Normal(0,1)



class ToyVAE(Autoencoder, nn.Module):
	def __init__(self, obs_dim, latent_dim, beta=1., hidden=[], nonlin=nn.ELU, soft_sigma=False,
	             criterion=nn.MSELoss(reduction='none'), **kwargs):
		super().__init__(latent_dim=latent_dim, **kwargs)
		self.obs_dim = obs_dim
		self.beta = beta
		self._soft_sigma = soft_sigma

		self.encoder = make_MLP(obs_dim, latent_dim if beta is None else latent_dim*2, hidden, nonlin=nonlin)
		self.decoder = make_MLP(latent_dim, obs_dim, hidden, nonlin=nonlin)

		self.criterion = criterion


	def forward(self, x):
		z = self.encode(x)
		rec = self.decode(z)
		info = {'latent': z, 'reconstruction': rec}

		rec_loss = self.criterion(rec, x).sum()
		info['rec_loss'] = rec_loss
		loss = rec_loss#.mean(0)

		if self.beta is not None:
			# kl_loss = distrib.kl_divergence(z, _standard_normal).sum()#.sum(dim=-1)
			mu, sigma = z.mean, z.stddev
			kl_loss = (mu.pow(2) - sigma.log() + sigma - 1).sum() / 2
			loss += self.beta * kl_loss#.mean(0)
			info['kl_loss'] = kl_loss
		return loss, info


	def encode(self, x):
		z = self.encoder(x)
		if self.beta is not None:
			mu, logsigma = z.chunk(2,dim=1)
			z = distrib.Normal(mu, F.softplus(logsigma) if self._soft_sigma else logsigma.exp())
		return z


	def decode(self, z):
		if isinstance(z, distrib.Normal):
			z = z.rsample() if self.training else z.mean
		return self.decoder(z)



















