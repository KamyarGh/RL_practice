import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F


def FCNet(layer_dims):
	layers = []
	for i in xrange(len(layer_dims)-2):
		layers += [
			nn.Linear(layer_dims[i], layer_dims[i+1]),
			nn.BatchNorm1d(layer_dims[i+1]),
			nn.ReLU()
		]
	layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

	return nn.Sequential(*layers)


if __name__ == '__main__':
	fc_net = FCNet(100, 6)
	obs = Variable(torch.randn(1,100))

	print(fc_net)

	import time
	N = 100
	start = time.time()
	for i in xrange(N):
		fc_net.forward(obs)
	end = time.time()

	print((start - end) / N)
