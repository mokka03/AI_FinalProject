import skimage
import torch
from matplotlib.patches import Rectangle
from .utils import to_tensor


class WaveSource(torch.nn.Module):
    def __init__(self, x, y, dim=0):
        super().__init__()

        # These need to be longs for advanced indexing to work
        self.register_buffer('x', to_tensor(x, dtype=torch.int64))
        self.register_buffer('y', to_tensor(y, dtype=torch.int64))
        self.register_buffer('dim', to_tensor(dim, dtype=torch.int32))

    def forward(self, B, Bt):
        B[0, self.x, self.y] = Bt[0, 0]
        return B

    def plot(self, ax, color='r'):
        marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', markeredgecolor=color, markerfacecolor='none', markeredgewidth=1.0, markersize=4)
        return marker


class WaveLineSource(WaveSource):
    def __init__(self, r0, c0, r1, c1, dim=0):
        x, y = skimage.draw.line(r0, c0, r1, c1)

        self.r0 = r0
        self.c0 = c0
        self.r1 = r1
        self.c1 = c1
        super().__init__(x, y, dim)

    def plot(self, ax, color='r'):
        line, = ax.plot([self.r0, self.r1], [self.c0, self.c1], 'g-', alpha=0.9   , lw=2)
        return line
    
class WaveRectangleSource(WaveSource):
    def __init__(self, r0, c0, r1, c1, dim=0):
        x, y = skimage.draw.rectangle((r0, c0), end=(r1, c1))

        self.r0 = r0
        self.c0 = c0
        self.r1 = r1
        self.c1 = c1
        super().__init__(x, y, dim)

    def plot(self, ax, color='r'):
        rect = Rectangle(((self.r0, self.c0)), self.r1-self.r0, self.c1-self.c0,
                         ec='None', fc='b', alpha=0.6)
        ax.add_patch(rect)
        return rect
