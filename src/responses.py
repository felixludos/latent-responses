from typing import Union, Optional, TypeVar, Generic, Dict, List
import omnifig as fig
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions as distrib
from torch.utils.data import Dataset

from omnilearn import util

import tsalib
B, C, H, W, D = tsalib.dim_vars('Batch(b) Channels(c) Height(h) Width(w) Latent(d)', exists_ok=True)
Observation = B, C, H, W
# Latent = (B,D)
Latent = Union[torch.Tensor, distrib.Distribution] # B, D
# Latent = torch.Tensor


class Autoencoder:
	def __init__(self, latent_dim, **kwargs):
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

	def select_indices(self, N: int, num_samples: int, # selects N samples from num_samples
	           force_different: bool = False, gen: torch.Generator = None) -> torch.Tensor:
		if N < num_samples:
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
		if isinstance(sel, int):
			sel = [sel]

		if values is None:
			values = self.sample_prior(len(z), gen=gen)[:, sel]
		else:
			values = torch.as_tensor(values).to(z.device)
			if len(values.size()) == 0:
				values = values.expand(len(sel))
			if len(values.size()) == 1:
				values = values.unsqueeze(1)
			if values.size(1) > len(sel):
				assert values.size(1) == z.size(1), 'interventions should have the same dims as latent or len(sel)'
				values = values[:, sel]
			else:
				values = values.expand(-1, len(sel))

		assert shuffle or len(values) == len(z), \
			f'should have an intervention for every sample: {len(values)} vs {len(z)}'
		if shuffle:
			values = values[self.select_indices(len(z), len(values), force_different=force_different, gen=gen)]

		z[:, sel] = values
		return z

	def intervene(self, z: Latent, sel: Union[int, List[int]],
	              values: Union[Latent, torch.Tensor, float] = None,
	              gen: torch.Generator = None, **kwargs) -> Latent:
		return self.intervene_(z.clone(), sel=sel, values=values, gen=gen, **kwargs)


# def _response_mat()


def response_mat(autoencoder: Autoencoder, samples: Optional[Latent] = None,
                 interventions: Optional[Union[Dict[int,Latent],Latent]] = None,
                 num_samples: Optional[int] = 100, force_different: bool = False,
                 gen: Optional[torch.Generator] = None, pbar: Optional[tqdm] = None) -> torch.Tensor:
	if interventions is None:
		interventions = {}
	elif isinstance(interventions, torch.Tensor):
		interventions = {i: interventions[:, i] for i in range(interventions.size(1))}

	rows = []
	itr = range(autoencoder.latent_dim)
	if pbar is not None:
		itr = pbar(itr, total=autoencoder.latent_dim)
	for j in itr:
		if pbar is not None:
			itr.set_description(f'Response matrix')
		if num_samples is None:
			z = samples
		else:
			z = autoencoder.sample_prior(num_samples, gen=gen) if samples is None \
				else samples[autoencoder.select_indices(num_samples, len(samples),
				                                        force_different=force_different, gen=gen)]
		z_int = autoencoder.intervene(z, j, interventions.get(j), gen=gen)

		s = autoencoder.response(z)
		if isinstance(s, distrib.Distribution):
			s = s.mean
		s_int = autoencoder.response(z_int)
		if isinstance(s, distrib.Distribution):
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


	def images_given_index(self, inds):
		return self.images[inds @ self._factor_steps]


	def sample_inds(self, n=1, gen=None):
		return torch.rand(n, 6, generator=gen).mul(self._factor_nums).long()


	def factor_traversal_codes(self, factor, base=None, gen=None):
		if base is None:
			base = self.sample_inds(1, gen=gen)[0]
		if isinstance(factor, int):
			factor = self.factor_order[factor]

		i = self.factor_order.index(factor)
		v = torch.arange(self.factor_sizes[factor])

		codes = base.unsqueeze(0).expand(v.size(0), -1).clone()
		codes[:, i] = v
		return codes


	def factor_traversal_images(self, factor, base=None, gen=None):
		codes = self.factor_traversal_codes(factor, base=base, gen=gen)
		return self.images_given_index(codes)





def conditioned_response_mat(autoencoder: Autoencoder, dataset: FactorDataset, num_traversals: int = 50,
                             gen: Optional[torch.Generator] = None, pbar: Optional[tqdm] = None):


	for factor in dataset.factor_order:
		itr = range(num_traversals)
		if pbar is not None:
			itr = pbar(itr, total=num_traversals, desc=f'Factor {factor}')

		for j in itr:
			group = dataset.factor_traversal_images(factor, gen=gen)

			mat = response_mat(autoencoder, samples=group, interventions=group, num_samples=len(group),



	pass




# from full interventions

def sample_full_interventions(sampler, num_groups=50, device='cuda', pbar=None):
	
	D = len(sampler)
	
	factors = []
	
	itr = range(D)
	if pbar is not None:
		itr = pbar(itr, total=D)
		itr.set_description('Sampling interventions')
	else:
		print('Sampling interventions')
	for idx in itr:
		groups = [sampler.full_intervention(idx) for _ in range(num_groups)]
	
		full = torch.stack(groups).to(device)
		
		factors.append(full)
	
	return factors


def response_mat(Q, encode, decode, n_interv=None, scales=None,
                 force_different=False,
                 max_batch_size=None, device=None):
	if scales is not None:
		raise NotImplementedError
	
	@torch.no_grad()
	def response_function(q):
		if device is not None:
			q = q.to(device)
		r = encode(decode(q))
		if isinstance(r, distrib.Normal):
			r = r.loc
		return r
	
	B, D = Q.shape
	
	if n_interv is None:
		n_interv = B
	if max_batch_size is None:
		max_batch_size = n_interv
	
	resps = []
	for i, qi in enumerate(Q.t()):  # Opy(D) (n_interv is parallelized)
		order = torch.randperm(B)
		iorder = order.clone()
		if force_different:
			iorder[1:] = order[:-1]
			iorder[0] = order[-1]
		qsel = slice(0,n_interv) if force_different else torch.randint(B, size=(n_interv,))
		isel = slice(0,n_interv) if force_different else torch.randint(B, size=(n_interv,))
		q = Q[order[qsel]]
		dq = q.clone()
		dq[:, i] = qi[iorder[isel]]
		z = util.process_in_batches(response_function, q, batch_size=max_batch_size)
		dz = util.process_in_batches(response_function, dq, batch_size=max_batch_size)
		resps.append(dz.sub(z).pow(2).mean(0).sqrt())
	return torch.stack(resps)


def conditioned_reponses(encode, decode, factor_samples, resp_kwargs={}, include_q=False,
					pbar=None, factor_names=None):
	'''
	:param encode:
	:param decode:
	:param factor_samples: list with K elements (one for each factor of variation), each element hast
	N sets of full interventions
	:param resp_kwargs:
	:param include_q:
	:param pbar:
	:param factor_names:
	:return:
	'''

	Fs = []
	allQs = [] if include_q else None
	
	def _encode(x):
		q = encode(x)
		if isinstance(q, util.Distribution):
			q = q.bsample()
		return q
	
	for i, groups in enumerate(factor_samples):
		
		N, G, C, *other = groups.size()
		
		with torch.no_grad():
			
			Q = util.process_in_batches(_encode, groups.view(N*G, C, *other), batch_size=64)
			# Q = encode(groups.view(N*G, C, *other))
			# if isinstance(Q, util.Distribution):
			# 	Q = Q.bsample()
			Qs = Q.view(N, G, -1)
			if allQs is not None:
				allQs.append(Qs)
			
			Ms = []
			
			todo = zip(groups, Qs)
			if pbar is not None:
				todo = pbar(todo, total=len(groups))
				if factor_names is not None:
					todo.set_description(factor_names[i])
			
			for group, q in todo:
				Ms.append(response_mat(q, encode, decode, force_different=True, **resp_kwargs).cpu())
		
		Fs.append(torch.stack(Ms))
	
	out = [torch.stack(Fs)]
	if include_q:
		out.append(allQs)
	return out















#
# def compute_response(Q, encode, decode, include_q2=False, mag=None,
#                      force_different=False, skip_shuffle=False):
# 	N, D = Q.size()
#
# 	hyb = []
# 	resp = []
#
# 	q2s = None
# 	Q2 = [] if include_q2 else None
#
# 	if force_different and not skip_shuffle:
# 		Q = Q[torch.randperm(len(Q))]
#
# 	if mag is not None and isinstance(mag, (int, float)):
# 		mag = [mag] * D
#
# 	for idx in range(D):
#
# 		V = Q[:, idx]
# 		if mag is not None:
# 			m = mag[idx]
# 			U = V + m * (-1) ** torch.randint(2, size=(len(V),), device=V.device)
# 		elif force_different:
# 			U = V.clone()
# 			U[:-1] = V[1:]
# 			U[-1] = V[0]
# 		else:
# 			U = V[torch.randperm(len(V))]
# 		loader = DataLoader(TensorDataset(Q, U), batch_size=100, drop_last=False)
#
# 		H, Y = [], []
#
# 		for q, u in loader:
#
# 			with torch.no_grad():
# 				h = q.clone()
# 				h[:, idx] = u
#
# 				H.append(h)
# 				y = encode(decode(h))
# 				if isinstance(y, distrib.Distribution):
# 					y = y.loc
# 				Y.append(y)
#
# 				if Q2 is not None:
# 					q2 = encode(decode(q))
# 					if isinstance(q2, distrib.Distribution):
# 						q2 = q2.loc
# 					Q2.append(q2)
#
# 		hyb.append(torch.cat(H))
# 		resp.append(torch.cat(Y))
#
# 		if Q2 is not None:
# 			q2s = torch.cat(Q2)
# 			Q2 = None
#
# 	out = [torch.stack(hyb), torch.stack(resp)]
# 	if q2s is not None:
# 		out.append(q2s)
#
# 	return out
#
#
# def response_mat(Q, encode, decode, scales=None, dist_type='rms', mag=None, **resp_kwargs):
# 	if isinstance(mag, (float, int)) and scales is not None:
# 		mag = mag * scales
#
# 	H, Y, Q2 = compute_response(Q, encode, decode, include_q2=True, mag=mag, **resp_kwargs)
#
# 	R = Y - Q2.unsqueeze(0)
#
# 	if scales is not None:
# 		R /= scales.view(1, 1, -1)
#
# 	if dist_type == 'rms':
# 		R = R.pow(2).mean(1).sqrt()
# 	elif dist_type == 'sqr':
# 		R = R.pow(2).mean(1)
# 	elif dist_type == 'abs':
# 		R = R.abs().mean(1)
# 	elif dist_type == 'l1':
# 		R = R.abs().sum(1)
# 	elif dist_type == 'l2':
# 		R = R.pow(2).sum(1).sqrt()
#
# 	return R
#



	



