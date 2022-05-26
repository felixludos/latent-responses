from typing import Union, Optional, TypeVar, Generic, Dict, List
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions as distrib
from torch.utils.data import Dataset

Observation = torch.Tensor # B, C, H, W
Latent = Union[torch.Tensor, distrib.Distribution] # B, D



class Autoencoder:
	def __init__(self, latent_dim: int, **kwargs):
		super().__init__(**kwargs)
		self.latent_dim = latent_dim


	def sample(self, N: int = 1, gen: Optional[torch.Generator] = None) -> Observation:
		return self.decode(self.sample_prior(N, gen=gen))


	def sample_prior(self, N: int = 1, gen: Optional[torch.Generator] = None) -> Latent:
		return torch.randn(N, self.latent_dim, generator=gen)


	def encode(self, x: Observation) -> Latent:
		raise NotImplementedError


	def decode(self, z: Latent) -> Observation:
		raise NotImplementedError


	def reconstruct(self, x: Observation) -> Observation:
		return self.decode(self.encode(x))


	def response(self, z: Latent) -> Latent:
		if isinstance(z, distrib.Distribution):
			z = z.rsample()
		return self.encode(self.decode(z))


	def select_indices(self, N: int, num_samples: int, force_different: bool = False,
	                   gen: torch.Generator = None) -> torch.Tensor:
		'''
		Samples N indices [0, num_samples-1]. If N > num_samples, with replacement, otherwise without.

		:param N: number of indices to sample
		:param num_samples: max index + 1
		:param force_different: force each index to be different than it's position in the list
		:param gen: generator to use for sampling
		:return: indices
		'''
		if N > num_samples:
			picks = torch.randint(num_samples, size=(N,), generator=gen)
		else:
			picks = torch.randperm(num_samples, generator=gen)[:N]
			if force_different:
				bad = torch.where(picks.sub(torch.arange(N)).eq(0))[0]
				shifted = (bad + 1) % len(picks)
				picks[bad], picks[shifted] = picks[shifted], picks[bad]
		return picks


	def intervene_(self, z: Latent, sel: Union[int, List[int]], # TODO: make this differentiable
	              values: Union[Latent, torch.Tensor, float] = None,
	              shuffle: bool = True, force_different: bool = False,
	              gen: torch.Generator = None) -> Latent:
		'''
		Intervene in the latent samples z (in place) by replacing the selected indices with the given values.

		:param z: initial latent samples
		:param sel: which index/indices to intervene in
		:param values: intervention options (sampled from prior if not provided)
		:param shuffle: shuffle the intervention options
		:param force_different: force the intervention to be a different index than the original sample
		:param gen: generator to use for sampling
		:return: inplace intervened samples
		'''
		if isinstance(sel, int):
			sel = [sel]

		if values is None:
			values = self.sample_prior(len(z), gen=gen)[:, sel]
		else:
			values = torch.as_tensor(values)
			if len(values.size()) == 0:
				values = values.expand(len(sel))
			if len(values.size()) == 1:
				values = values.unsqueeze(1)
			if values.size(1) > len(sel):
				assert values.size(1) == z.size(1), 'interventions should have the same dims as latent or len(sel)'
				values = values[:, sel]
			else:
				values = values.expand(-1, len(sel))
		values = values.to(z.device)

		assert shuffle or len(values) == len(z), \
			f'should have an intervention for every sample: {len(values)} vs {len(z)}'
		if shuffle:
			values = values[self.select_indices(len(z), len(values), force_different=force_different, gen=gen)]

		z[:, sel] = values
		return z


	def intervene(self, z: Latent, sel: Union[int, List[int]],
	              values: Union[Latent, torch.Tensor, float] = None,
	              gen: torch.Generator = None, **kwargs) -> Latent:
		'''Same as intervene_ but not in place'''
		return self.intervene_(z.clone(), sel=sel, values=values, gen=gen, **kwargs)



def response_mat(autoencoder: Autoencoder, samples: Optional[Latent] = None,
                 interventions: Optional[Union[Dict[int,Latent],Latent]] = None,
                 num_samples: Optional[int] = 100, force_different: bool = False,
                 device: Optional[Union[torch.device,str]] = None, pbar: Optional[tqdm] = None,
                 gen: Optional[torch.Generator] = None, ) -> torch.Tensor:
	'''
	Compute the latent response matrix for the given autoencoder. Rows correspond to the latent dimension that is
	intervened, while values (generally in [0, 1]) are the average response in the latent dimension
	corresponding to the column.

	:param autoencoder: needed for encoding and decoding, interventions, and sampling (from the prior)
	:param samples: unintervened latent samples to start from (uses prior samples if not provided)
	:param interventions: intervention options for each latent dimension (uses prior samples if not provided)
	:param num_samples: number of samples to use (sampled from `samples` or prior)
	(defaults to `samples` if not provided)
	:param force_different: force the interventions to be from a different index than the original samples
	(only has an effect if both samples and interventions are provided)
	:param device: device to use for the computation
	:param pbar: optional progress bar
	:param gen: generator to use for sampling
	:return: (latent_dim, latent_dim) response matrix
	'''
	if interventions is None:
		interventions = {}
	elif isinstance(interventions, torch.Tensor):
		interventions = {i: interventions[:, i] for i in range(interventions.size(1))}

	assert num_samples is not None or samples is not None, 'must specify either samples or num_samples'

	rows = []
	itr = range(autoencoder.latent_dim)
	if pbar is not None:
		itr = pbar(itr, total=autoencoder.latent_dim)
	for j in itr:
		if pbar is not None:
			itr.set_description(f'Response matrix')
		
		options = interventions.get(j)
		
		if num_samples is None:
			z = samples
		elif samples is None:
			z = autoencoder.sample_prior(num_samples, gen=gen)
		else:
			inds = autoencoder.select_indices(num_samples, len(samples), gen=gen)
			if force_different and len(options) == len(samples):
				shift = torch.randint(len(options)-1, size=(num_samples,), generator=gen)
				options = options[(inds + shift + 1) % len(options)]
			
			z = samples[inds]
		
		if device is not None:
			z = z.to(device)
			if options is not None:
				options = options.to(device)
		
		z_int = autoencoder.intervene(z, j, options, gen=gen, shuffle=not force_different)

		s = autoencoder.response(z)
		if isinstance(s, distrib.Distribution):
			s = s.mean
		s_int = autoencoder.response(z_int)
		if isinstance(s_int, distrib.Distribution):
			s_int = s_int.mean

		rows.append( s_int.sub(s).pow(2).mean(0).sqrt() )
	return torch.stack(rows)



class FactorDataset(Dataset):
	def __init__(self, factor_order: List[str], factor_sizes: Dict[str,int], **kwargs):
		super().__init__(**kwargs)
		self.factor_order = factor_order
		self.factor_sizes = factor_sizes
		self._factor_steps = torch.as_tensor(np.array([self.factor_sizes[f] for f in self.factor_order] + [1])
		                                     [::-1].cumprod()[-2::-1].copy()).long()
		self._factor_nums = torch.as_tensor([self.factor_sizes[f] for f in self.factor_order]).float().unsqueeze(0)

		self.images = None


	def images_given_index(self, inds: torch.Tensor) -> torch.Tensor:
		'''
		Load the image(s) corresponding to the given label index(es)

		:param inds: (N, 6) tensor of label indices
		:return: corresponding images
		'''
		return self.images[inds @ self._factor_steps]


	def sample_inds(self, n: int = 1, gen: Optional[torch.Generator] = None) -> torch.Tensor:
		'''
		Sample `n` indices from the dataset

		:param n: number of samples
		:param gen: generator to use for sampling
		:return: (n, 6) tensor of label indices
		'''
		return torch.rand(n, 6, generator=gen).mul(self._factor_nums).long()


	def factor_traversal_codes(self, factor: Union[str, int], base: Optional[torch.Tensor] = None,
	                           gen: Optional[torch.Generator] = None):
		'''
		Generate a sequence of codes corresponding to a traversal for the given `factor` while all other factors are
		fixed (specified by `base`, or sampled uniformly)

		:param factor: factor to traverse
		:param base: other factors to use as a base (defaults to sampled uniformly), tensor shape (6)
		:param gen: generator to use for sampling
		:return: (k, 6) tensor of label indices corresponding to the traversal
		'''
		if base is None:
			base = self.sample_inds(1, gen=gen)[0]
		if isinstance(factor, int):
			factor = self.factor_order[factor]

		i = self.factor_order.index(factor)
		v = torch.arange(self.factor_sizes[factor])

		codes = base.unsqueeze(0).expand(v.size(0), -1).clone()
		codes[:, i] = v
		return codes


	def factor_traversal_images(self, factor: Union[str, int], base: Optional[torch.Tensor] = None,
	                           gen: Optional[torch.Generator] = None):
		'''
		Generate a sequence of images corresponding to a traversal for the given `factor` while all other factors are
		fixed (specified by `base`, or sampled uniformly)
		:param factor: factor to traverse
		:param base: other factors to use as a base (defaults to sampled uniformly), tensor shape (6)
		:param gen: generator to use for sampling
		:return: (k, 3, 64, 64) tensor of images corresponding to the traversal
		'''
		codes = self.factor_traversal_codes(factor, base=base, gen=gen)
		return self.images_given_index(codes)



def conditioned_response_mat(autoencoder: Autoencoder, dataset: FactorDataset, num_traversals: int = 50,
                             batch_size: int = 32, reduction: Optional[Union[str, None]] = 'mean',
                             device: Optional[Union[torch.device,str]] = None, pbar: Optional[tqdm] = None,
                             gen: Optional[torch.Generator] = None):
	'''
	Compute the response matrix for a given dataset, conditioned on the traversal of a given factor

	:param autoencoder: needed for encoding and decoding, interventions, and sampling (from the prior)
	:param dataset: must enable traversing a single true factor at a time (see FactorDataset)
	:param num_traversals: number of traversals to use for each factor (recommended to be at least 10)
	:param batch_size: number of interventions to use for each traversal (recommended to be at least 32)
	:param reduction: reduction to apply along the number of traversals (defaults to mean)
	:param device: device to use for tensors
	:param gen: generator to use for sampling
	:param pbar: optional progress bar to use
	:return: (num_factors, num_traversals, latent_dim) conditioned response matrix if reduction is None,
	otherwise (num_factors, latent_dim)
	'''

	rows = []
	for factor in dataset.factor_order:
		itr = range(num_traversals)
		if pbar is not None:
			itr = pbar(itr, total=num_traversals, desc=str(factor))
		diags = []
		for j in itr:
			group = dataset.factor_traversal_images(factor, gen=gen)
			group = autoencoder.encode(group.to(device))
			if isinstance(group, distrib.Distribution):
				group = group.mean
			group = group.cpu()

			mat = response_mat(autoencoder, samples=group, interventions=group, num_samples=batch_size, device=device,
			                   force_different=True, gen=gen)
			diags.append(mat.diag())
		rows.append(torch.stack(diags))
	rows = torch.stack(rows)
	
	if reduction == 'mean':
		rows = rows.mean(1)
	return rows



def score_from_conditioned_response_mat(cond_mat: torch.Tensor) -> float:
	'''Aggregates the conditioned response matrix into a scalar score -> Causal Disentanglement Score'''
	return cond_mat.max(0)[0].sum().div(cond_mat.sum()).item()



def causal_disentanglement_score(autoencoder: Autoencoder, dataset: FactorDataset, num_traversals: int = 50,
                                 device: Optional[Union[torch.device,str]] = None, **kwargs):
	'''Computes the Causal Disentanglement Score by first computing the conditioned response matrix'''
	cond_mat = conditioned_response_mat(autoencoder, dataset, num_traversals=num_traversals, device=device, **kwargs)
	return score_from_conditioned_response_mat(cond_mat)





	



