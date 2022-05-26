import sys, os, time, shutil, random
from typing import Union, Optional, Type, List, Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch import distributions as distrib
from torch.nn import functional as F

from .utils import make_MLP
from .responses import Autoencoder

Observation = torch.Tensor
Latent = Union[torch.Tensor, distrib.Distribution]



def helix_labels(N: int, strands: Union[None, int] = None, gen: torch.Generator = None) -> torch.Tensor:
	'''
	Generate N labels a N-helix dataset. The labels are 2D: the first dimension is the height, and the
	second is the helix number.

	:param N: number of observations to generate
	:param strands: number of helices (evenly spaced)
	:param gen: torch.Generator to use
	:return: labels shape (N, 2)
	'''
	if strands is None:
		strands = 2
	z = torch.rand(N, generator=gen).mul(2).sub(1)
	n = torch.randint(strands, size=(N,), generator=gen)
	return torch.stack([z, n], -1)



def helix_observations(Y: Union[int, torch.Tensor], noise_std: Union[None, float] = 0.1, strands: Union[None, int] = 2,
                       w: float = 1., Rx: float = 1., Ry: float = 1., Rz: float = 1.,
                       gen: torch.Generator = None) -> torch.Tensor:
	'''
	Generate N observations from a N-helix dataset. Observations are 3D with the helices evenly spaced centered at
	the origin, and extending along the z-direction.
	:param Y: labels shape (N, 2) (or number of samples to generate)
	:param noise_std: standard deviation of the additive gaussian noise
	:param strands: number of helices (evenly spaced)
	:param w: frequency of the helix
	:param Rx: x-axis radius
	:param Ry: y-axis radius
	:param Rz: z-axis extent
	:param gen: torch.Generator to use
	:return: observations shape (N, 3)
	'''
	if isinstance(Y, int):
		Y = helix_labels(Y, strands=strands, gen=gen)
	z = Y.narrow(-1,0,1)
	n = Y.narrow(-1,1,1)
	if strands is None:
		strands = n.max().item() + 1
	theta = z.mul(w).add(n.div(strands)*2).mul(np.pi)
	amp = torch.tensor([Rx, Ry, Rz]).float().to(Y.device)
	X = amp.unsqueeze(0) * torch.cat([theta.cos(), theta.sin(), z],-1)
	if noise_std is not None:
		X += noise_std * torch.randn_like(X)
	return X



class ToyVAE(Autoencoder, nn.Module):
	'''VAE to use for the helix toy dataset'''
	def __init__(self, obs_dim: int, latent_dim: int, beta: Optional[float] = 1., hidden: Optional[List[int]] =[],
	             nonlin: Optional[Type[nn.Module]] = nn.ELU, soft_sigma: Optional[bool] = False,
	             criterion: Optional[nn.Module] = nn.MSELoss(reduction='none'), **kwargs):
		super().__init__(latent_dim=latent_dim, **kwargs)
		self.obs_dim = obs_dim
		self.beta = beta
		self._soft_sigma = soft_sigma

		self.encoder = make_MLP(obs_dim, latent_dim if beta is None else latent_dim*2, hidden, nonlin=nonlin)
		self.decoder = make_MLP(latent_dim, obs_dim, hidden, nonlin=nonlin)

		self.criterion = criterion


	def forward(self, x: Observation) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		z = self.encode(x)
		rec = self.decode(z)
		info = {'latent': z, 'reconstruction': rec}

		rec_loss = self.criterion(rec, x).sum()
		info['rec_loss'] = rec_loss
		loss = rec_loss

		if self.beta is not None:
			mu, sigma = z.mean, z.stddev
			kl_loss = (mu.pow(2) - sigma.log() + sigma - 1).sum() / 2
			loss += self.beta * kl_loss
			info['kl_loss'] = kl_loss
		return loss, info


	def encode(self, x: Observation) -> Latent:
		z = self.encoder(x)
		if self.beta is not None:
			mu, logsigma = z.chunk(2,dim=1)
			z = distrib.Normal(mu, F.softplus(logsigma) if self._soft_sigma else logsigma.exp())
		return z


	def decode(self, z: Latent) -> Observation:
		if isinstance(z, distrib.Normal):
			z = z.rsample() if self.training else z.mean
		return self.decoder(z)



















