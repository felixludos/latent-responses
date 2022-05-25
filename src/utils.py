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






