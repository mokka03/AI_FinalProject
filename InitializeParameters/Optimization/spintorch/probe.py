import torch
import skimage

from .utils import to_tensor


class WaveProbe(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		# Need to be int64
		self.register_buffer('x', to_tensor(x, dtype=torch.int64))
		self.register_buffer('y', to_tensor(y, dtype=torch.int64))

	def forward(self, x):
		return x[:,1, self.x, self.y]

	def plot(self, ax, color='k'):
		marker, = ax.plot(self.x.cpu().numpy(), self.y.cpu().numpy(), '.', markeredgecolor='none', markerfacecolor=color, markeredgewidth=0.3, markersize=2,alpha=0.5)
		return marker


class WaveIntensityProbe(WaveProbe):
	def __init__(self, x, y):
		super().__init__(x, y)

	def forward(self, x):
		return super().forward(x).pow(2)

class WaveIntensityProbeDisk(WaveIntensityProbe):
	def __init__(self, x, y, r):
		x, y = skimage.draw.circle(x, y, r)
		super().__init__(x, y)

	def forward(self, x):
		return super().forward(x).sum().unsqueeze(0)
