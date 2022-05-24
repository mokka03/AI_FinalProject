from typing import Tuple
from numba import cuda
import torch
from torch import rfft, irfft, stack, sum
from .utils import to_tensor
from .demag import demag_tensor_fft, complex_multiply
from numpy import pi
from scipy.io import loadmat



class WaveGeometry(torch.nn.Module):
    def __init__(self, dim: Tuple, dx: float, dy: float, dz: float, B0: float, Ms: float):
        super().__init__()

        self.dim = dim
        self.register_buffer("dx", to_tensor(dx))
        self.register_buffer("dy", to_tensor(dy))
        self.register_buffer("dz", to_tensor(dz))
        self.register_buffer("B0", to_tensor(B0))
        self.register_buffer("Ms", to_tensor(Ms))

    def forward(self):
        raise NotImplementedError


class WaveGeometryFreeForm(WaveGeometry):
    def __init__(self, dim: Tuple, dx: float, dy: float, dz: float, B0: float, B1: float, Ms: float):

        super().__init__(dim, dx, dy, dz, B0, Ms)

        self.rho = torch.nn.Parameter(torch.zeros(dim))
        self.register_buffer("B", torch.zeros((3,)+dim))
        self.register_buffer("B1", to_tensor(B1))
        self.B[2,] = self.B0
        
    def forward(self):
        self.B = torch.zeros_like(self.B)
        self.B[2,] = self.B1*self.rho + self.B0
        return self.B



class WaveGeometryMs(WaveGeometry):   # Eredeti
    def __init__(self, dim: Tuple, dx: float, dy: float, dz: float, Ms: float, B0: float):

        super().__init__(dim, dx, dy, dz, B0, Ms)

        # self.rho = torch.nn.Parameter(torch.ones((dim[0]-40,dim[1]-0)))
        self.rho = torch.nn.Parameter(torch.ones((308,308)))
        self.register_buffer("Msat", torch.zeros(dim))
        self.register_buffer("B0", to_tensor(B0))
        self.register_buffer("B", torch.zeros((3,)+dim))
        self.B[2,] = self.B0
        
    def forward(self):
        rho_pad = torch.ones_like(self.Msat)*142780
        # rho_pad[20:self.dim[0]-20,0:self.dim[1]-0] = self.rho
        rho_pad[6:314,166:474] = self.rho * self.Ms
        self.Msat = rho_pad
        return self.Msat

class WaveGeometryMsBinary(WaveGeometry):
    def __init__(self, dim: Tuple, dx: float, dy: float, dz: float, Ms: float, B0: float):

        super().__init__(dim, dx, dy, dz, B0, Ms)

        # self.rho = torch.nn.Parameter(torch.ones((dim[0]-40,dim[1]-0)))
        self.rho = torch.nn.Parameter(torch.ones((dim[0]-20,dim[1]-20)))
        self.register_buffer("Msat", torch.zeros(dim))
        self.register_buffer("B0", to_tensor(B0))
        self.register_buffer("B", torch.zeros((3,)+dim))
        self.B[2,] = self.B0
        
    def forward(self):
        self.rho = (torch.tanh(self.rho)+1)/2
        rho_pad = torch.ones_like(self.Msat)
        # rho_pad[20:self.dim[0]-20,0:self.dim[1]-0] = self.rho
        rho_pad[10:self.dim[0]-10,10:self.dim[1]-10] = self.rho
        self.Msat = self.Ms*rho_pad
        return self.Msat


class WaveGeometryArray(WaveGeometry):
    def __init__(self, rho, dim: Tuple, dx: float, dy: float, dz: float, B0: float,
                 r0: int, dr: int, dm: int, z_off: int, rx: int, ry: int,
                 Ms_CoPt: float, beta: float = 100.0):

        super().__init__(dim, dx, dy, dz, B0)
        self.r0 = r0
        self.dr = dr
        self.rx = rx
        self.ry = ry
        self.dm = dm
        self.z_off = z_off
        self.register_buffer("beta", to_tensor(beta))
        self.register_buffer("Ms_CoPt", to_tensor(Ms_CoPt))
        self.rho = torch.nn.Parameter(to_tensor(rho))
        self.convolver = torch.nn.Conv2d(3, 3, self.dm, padding=(self.dm//2), 
                                    groups=3, bias=False).to(self.beta.device)
        self.convolver.weight.requires_grad = False
        
        for i in range(3):
            self.convolver.weight[i, 0, ] = torch.ones((dm, dm), device=self.beta.device)
        
        self.register_buffer("B", torch.zeros((3,)+dim))
        self.B[2,] += self.B0

    def forward(self):
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr, rx, ry = self.r0, self.dr, self.rx, self.ry

        rho = torch.tanh(self.rho*self.beta)   
        m_rho = torch.zeros((1, 3, ) + self.dim, device=self.dx.device)
        m_rho[0, 2, r0:r0+rx*dr:dr, r0:r0+ry*dr:dr] = rho

        m_rho_ = self.convolver(m_rho)[:,:,0:nx,0:ny]
        m_ = torch.nn.functional.pad(m_rho_.unsqueeze(4), (0, nz, 0, ny, 0, nx))
        m_fft = rfft(m_, 3, onesided=False)
        Kx_fft, Ky_fft, Kz_fft = demag_tensor_fft(nx, ny, nz, float(self.dx), float(self.dy), float(self.dz),
                                                  self.B0.device, int(self.z_off))

        B_demag = irfft(stack(
            [sum(complex_multiply(Kx_fft, m_fft),1),
             sum(complex_multiply(Ky_fft, m_fft),1),
             sum(complex_multiply(Kz_fft, m_fft),1)], 1),3,onesided=False)

        self.B = B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]*self.Ms_CoPt*mu0
        self.B[2,] += self.B0
        return self.B
