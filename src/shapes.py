import subprocess
from typing import Union, Optional, Type, List, Dict, Tuple
from pathlib import Path
import h5py as hf
import numpy as np
import torch
from torch import nn
from torch import distributions as distrib
from torch.nn import functional as F

from .utils import make_MLP
from .responses import Autoencoder, FactorDataset

Observation = torch.Tensor
Latent = Union[torch.Tensor, distrib.Distribution]



class Shapes3D(FactorDataset):
	def __init__(self, root: str = '.', download: bool = False, **kwargs):
		super().__init__(factor_order = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'],
		                 factor_sizes = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
		                               'scale': 8, 'shape': 4, 'orientation': 15},
		                 **kwargs)
		self.path = Path(root) / '3dshapes.h5'

		if not self.path.exists():
			if download:
				self.download(self.path)
			else:
				raise FileNotFoundError(f'{str(self.path)} not found (try setting download=True)')

		with hf.File(self.path, 'r') as f:
			images = f['images'][()]
			labels = f['labels'][()]
		self.images = torch.from_numpy(images).permute(0,3,1,2)
		self.labels = torch.from_numpy(labels)


	def format_images(self, imgs: torch.Tensor) -> torch.Tensor:
		'''Rescale images to be floats in [0,1].'''
		return imgs.float().div(255)
	
	
	def images_given_index(self, inds: torch.Tensor) -> torch.Tensor:
		'''Given a list of indices, return the corresponding images.'''
		return self.format_images(super().images_given_index(inds))


	_source_url = 'gs://3d-shapes/3dshapes.h5'
	@classmethod
	def download(cls, destination: Union[str, Path]) -> None:
		'''
		Download the dataset to the given destination from Google storage using `gsutil`.
		(total download size: ~255MB)
		'''
		print('Downloading 3D shapes dataset... ')
		try:
			subprocess.run(['gsutil', 'cp', cls._source_url, str(destination)])
		except:
			print('Failed to download 3D shapes dataset. (Try running `pip install gsutil`)')
			raise
		print('Done.')


	def __len__(self):
		return len(self.images)


	def __getitem__(self, item: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
		'''Returns the image and label at the given index.'''
		return self.format_images(self.images[item]), self.labels[item]



class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
	             pool: Optional[str] = None, unpool: Optional[str] = None, pool_size: int = 2,
	             nonlin: Optional[Type[nn.Module]] = nn.ELU, norm: Optional[str] = 'batch', **kwargs):
		super().__init__()
		self.unpool = None if unpool is None else nn.Upsample(scale_factor=pool_size, mode=unpool)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
		self.pool = None if pool is None \
			else (nn.MaxPool2d(pool_size) if pool == 'max' else nn.AvgPool2d(pool_size))
		if norm == 'batch':
			self.norm = nn.BatchNorm2d(out_channels)
		elif norm == 'instance':
			self.norm = nn.InstanceNorm2d(out_channels)
		elif norm == 'group':
			self.norm = nn.GroupNorm(8, out_channels)
		else:
			self.norm = None
		self.nonlin = nonlin()


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		c = self.unpool(x) if self.unpool is not None else x
		c = self.conv(c)
		if self.pool is not None:
			c = self.pool(c)
		if self.norm is not None:
			c = self.norm(c)
		if self.nonlin is not None:
			c = self.nonlin(c)
		return c



class ShapesVAE(Autoencoder, nn.Module):
	def __init__(self, latent_dim: int = 24, soft_sigma: bool = False, **kwargs):
		super().__init__(latent_dim=latent_dim, **kwargs)
		self._soft_sigma = soft_sigma

		self.encoder = nn.Sequential(
			ConvBlock( 3, 64, 5, 1, 2, pool='max', norm='group', nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, pool='max', norm='group', nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, pool='max', norm='group', nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, pool='max', norm='group', nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, pool='max', norm='group', nonlin=nn.ELU),
			make_MLP((64, 2, 2), latent_dim*2, [256, 128], nonlin=nn.ELU),
		)

		self.decoder = nn.Sequential(
			make_MLP(latent_dim, (64, 2, 2), [128, 256], nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, unpool='bilinear', norm='group', nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, unpool='bilinear', norm='group', nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, unpool='bilinear', norm='group', nonlin=nn.ELU),
			ConvBlock(64, 64, 3, 1, 1, unpool='bilinear', norm='group', nonlin=nn.ELU),
			ConvBlock(64,  3, 3, 1, 1, unpool='bilinear', norm=None, nonlin=nn.Sigmoid),
		)


	def encode(self, x: Observation) -> Latent:
		z = self.encoder(x)
		mu, logsigma = z.chunk(2, dim=1)
		z = distrib.Normal(mu, F.softplus(logsigma) if self._soft_sigma else logsigma.exp())
		return z


	def decode(self, z: Latent) -> Observation:
		if isinstance(z, distrib.Normal):
			z = z.rsample() if self.training else z.mean
		return self.decoder(z)











