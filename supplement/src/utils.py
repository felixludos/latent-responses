import numpy as np
import torch
from torch import nn


def make_MLP(din, dout, hidden=None, nonlin=nn.ELU, output_nonlin=None, bias=True, output_bias=None):
	'''
	:param din: int
	:param dout: int
	:param hidden: ordered list of int - each element corresponds to a FC layer with that width (empty means network is not deep)
	:param nonlin: str - choose from options found in get_nonlinearity(), applied after each intermediate layer
	:param output_nonlin: str - nonlinearity to be applied after the last (output) layer
	:return: an nn.Sequential instance with the corresponding layers
	'''

	if hidden is None:
		hidden = []

	if output_bias is None:
		output_bias = bias

	flatten = False
	reshape = None

	if isinstance(din, (tuple, list)):
		flatten = True
		din = int(np.product(din))
	if isinstance(dout, (tuple, list)):
		reshape = dout
		dout = int(np.product(dout))

	nonlins = [nonlin] * len(hidden) + [output_nonlin]
	biases = [bias] * len(hidden) + [output_bias]
	hidden = din, *hidden, dout

	layers = []
	if flatten:
		layers.append(nn.Flatten())

	for in_dim, out_dim, nonlin, bias in zip(hidden, hidden[1:], nonlins, biases):
		layer = nn.Linear(in_dim, out_dim, bias=bias)
		layers.append(layer)
		if nonlin is not None:
			layers.append(nonlin())

	if reshape is not None:
		layers.append(Reshaper(reshape))


	net = nn.Sequential(*layers)

	net.din, net.dout = din, dout
	return net



class Reshaper(nn.Module): # by default flattens
	def __init__(self, dout=(-1,)):
		super().__init__()

		self.dout = dout


	def extra_repr(self):
		return f'out={self.dout}'


	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.dout)



import matplotlib.pyplot as plt
from matplotlib.figure import figaspect



def factors(n): # has duplicates, starts from the extremes and ends with the middle
	return (x for tup in ([i, n//i]
				for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)



def calc_tiling(N, H=None, W=None, prefer_tall=False):

	if H is not None and W is None:
		W = N//H
	if W is not None and H is None:
		H = N//W

	if H is not None and W is not None and N == H*W:
		return H, W

	H,W = tuple(factors(N))[-2:] # most middle 2 factors

	if H > W or prefer_tall:
		H, W = W, H

	return H, W



def plot_imgs(imgs, H=None, W=None,
              figsize=None, scale=1,
              reverse_rows=False, grdlines=False,
              channel_first=None,
              imgroot=None, params={},
              savepath=None, dpi=96, autoclose=True, savescale=1,
              adjust={}, border=0., between=0.01):
	if isinstance(imgs, torch.Tensor):
		imgs = imgs.detach().cpu().squeeze(0).numpy()
	elif isinstance(imgs, (list, tuple)):
		imgs = [img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img for img in imgs]

	if isinstance(imgs, np.ndarray):
		shape = imgs.shape

		if channel_first is None \
				and shape[0 if len(shape) == 3 else 1] in {1, 3, 4} and shape[-1] not in {1, 3, 4}:
			channel_first = True

		if len(shape) == 2 or (len(shape) == 3 and ((shape[0] in {1, 3, 4} and channel_first)
		                                            or (shape[-1] in {1, 3, 4} and not channel_first))):
			imgs = [imgs]
		elif len(shape) == 4:
			if channel_first:
				imgs = imgs.transpose(0, 2, 3, 1)
				channel_first = False
		else:
			raise Exception(f'unknown shape: {shape}')

	imgs = [img.transpose(1, 2, 0).squeeze() if channel_first and len(img.shape) == 3 else img.squeeze() for img in
	        imgs]

	iH, iW = imgs[0].shape[:2]

	H, W = calc_tiling(len(imgs), H=H, W=W)

	fH, fW = iH * H, iW * W

	aw = None
	if figsize is None:
		aw, ah = figaspect(fH / fW)
		aw, ah = scale * aw, scale * ah
		figsize = aw, ah

	fg, axes = plt.subplots(H, W, figsize=figsize)
	axes = [axes] if len(imgs) == 1 else list(axes.flat)

	for ax, img in zip(axes, imgs):
		plt.sca(ax)
		if reverse_rows:
			img = img[::-1]
		plt.imshow(img, **params)
		if grdlines:
			plt.plot([0, iW], [iH / 2, iH / 2], c='r', lw=.5, ls='--')
			plt.plot([iW / 2, iW / 2], [0, iH], c='r', lw=.5, ls='--')
			plt.xlim(0, iW)
			plt.ylim(0, iH)

		plt.axis('off')

	if adjust is not None:

		base = dict(wspace=between, hspace=between,
		            left=border, right=1 - border, bottom=border, top=1 - border)
		base.update(adjust)
		plt.subplots_adjust(**base)

	if savepath is not None:
		plt.savefig(savepath, dpi=savescale * (dpi if aw is None else fW / aw))
		if autoclose:
			plt.close()
			return

	return fg, axes


def plot_mat(M, val_fmt=None, figax=None, figsize=None, figside=0.7,
             edgeeps=0.03, text_kwargs=dict(), **kwargs):
	if figax is None:
		figax = plt.subplots(figsize=figsize)
	fg, ax = figax

	plt.sca(ax)
	if isinstance(M, torch.Tensor):
		M = M.cpu().detach().numpy()
	if len(M.shape) == 1:
		M = M.reshape(1, -1)
	H, W = M.shape
	if figsize is None:
		fg.set_size_inches(figside * W + 0.5, figside * H + 0.5)
	plt.matshow(M, False, **kwargs)
	plt.yticks(np.arange(H), np.arange(H))
	plt.xticks(np.arange(W), np.arange(W))
	plt.subplots_adjust(edgeeps, edgeeps, 1 - edgeeps, 1 - edgeeps)
	if val_fmt is not None:
		if isinstance(val_fmt, int):
			val_fmt = f'.{val_fmt}g'
		if isinstance(val_fmt, str):
			val_fmt = '{:' + val_fmt + '}'
			fmt = lambda x: val_fmt.format(x)
		else:
			fmt = val_fmt

		if 'va' not in text_kwargs:
			text_kwargs['va'] = 'center'
		if 'ha' not in text_kwargs:
			text_kwargs['ha'] = 'center'
		for (i, j), z in np.ndenumerate(M):
			ax.text(j, i, fmt(z), **text_kwargs)
	return fg, ax



