import numpy as np
from numpy.core.numeric import ones
import torch as tr
import torch.nn as nn
from .utils import nan2num, to_tensor
from skimage.draw import rectangle_perimeter
from .demag import demag_tensor_fft_2D, complex_multiply
from torch.utils.checkpoint import checkpoint
from .geom import WaveGeometryFreeForm, WaveGeometryMs, WaveGeometryMsBinary
from scipy.io import savemat


class WaveCell(tr.nn.Module):
    def __init__(self, geometry, dt: float, Ms: float, gamma_LL: float,
                 alpha: float, A_exch: float, sources, probes=[]):

        super().__init__()

        # Set values
        self.relax = False
        self.register_buffer("dt", to_tensor(dt))                                   # timestep (s)
        self.register_buffer("gamma_LL", to_tensor(gamma_LL))                       # gyromagnetic ratio (rad/Ts)
        self.register_buffer("alpha", to_tensor(alpha))                             # damping coefficient ()
        self.register_buffer("A_exch", to_tensor(A_exch))                           # exchange coefficient (J/m)
        self.geom = geometry
        self.register_buffer("Msat_copy", tr.ones(self.geom.dim))
        self.register_buffer("ones_", tr.ones(self.geom.dim))





        self.LAPLACE = nn.Conv2d(3, 3, 3, groups=3, padding=1,
                                 padding_mode='replicate',
                                 bias=False).to(self.dt.device)
        self.LAPLACE.weight.requires_grad = False

        for i in range(3):
            self.LAPLACE.weight[i, 0, :, :] = tr.tensor([[0.0, 1.0/self.geom.dx**2, 0.0],
                                                         [1.0/self.geom.dy**2, -2.0/self.geom.dx**2 - 2.0/self.geom.dy**2, 1.0/self.geom.dy**2],
                                                         [0.0, 1.0/self.geom.dx**2, 0.0]])

        A = alpha*tr.ones((1, 3,) + self.geom.dim)
        N = 10
        alpha_max = 0.5
        for i in range(N):
            x, y = rectangle_perimeter((i+1, 0), (self.geom.dim[0]-i-2,
                                                    self.geom.dim[1]-2))
            A[:, :, x, y] = (1-i/N)**2*(alpha_max-alpha) + alpha
        A[:, :, N:self.geom.dim[0]-N-1, :] = alpha
        self.register_buffer("Alpha", A)

        Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft = demag_tensor_fft_2D(
            int(self.geom.dim[0]), int(self.geom.dim[1]),
            float(self.geom.dx), float(self.geom.dy), float(self.geom.dz), self.dt.device)
        self.register_buffer("Kxx_fft", Kxx_fft)
        self.register_buffer("Kyy_fft", Kyy_fft)
        self.register_buffer("Kzz_fft", Kzz_fft)
        self.register_buffer("Kxy_fft", Kxy_fft)
        self.register_buffer("Kxz_fft", Kxz_fft)
        self.register_buffer("Kyz_fft", Kyz_fft)

        if type(sources) is list:
            self.sources = tr.nn.ModuleList(sources)
        else:
            self.sources = tr.nn.ModuleList([sources])

        if type(probes) is list:
            self.probes = tr.nn.ModuleList(probes)
        else:
            self.probes = tr.nn.ModuleList([probes])

    # def parameters(self, recursive=True):
    #     for param in self.geom.parameters():
    #         yield param

    def forward(self, signal, output_fields=False):
        if isinstance(self.geom, WaveGeometryMs) or isinstance(self.geom, WaveGeometryMsBinary):
            M = tr.zeros((1, 3,) + self.geom.dim).to(self.dt.device)
            M[:, 2,] =  self.geom()
            M.to(self.dt.device)
            self.Msat_copy = self.geom.Msat.clone().detach()
            B_ext = self.geom.B

        elif isinstance(self.geom, WaveGeometryFreeForm):
            B_ext = self.geom()
            self.Msat_copy = self.ones_*self.geom.Ms.clone().detach()
            M = tr.zeros((1, 3,) + self.geom.dim).to(self.dt.device)
            M[:, 2,] = self.Msat_copy  # set magnetization in z direction
            M.to(self.dt.device)

        B_ext_0 = B_ext
        # Damp waves from magnetization change
        self.relax = True
        self.Nr=20
        for t in range(200//self.Nr):
            M = checkpoint(self.n_step_relax, M, B_ext)
        self.relax = False
        M0 = M
        y_all = []
        self.N = 50
        t1 = 0
        for tn, st in enumerate(signal.chunk(self.N, dim=1)):
            y2, M = checkpoint(self.n_step, M, B_ext, M0, B_ext_0, st, 
                            tr.tensor(tn,device=self.dt.device),
                            tr.tensor(output_fields,device=self.dt.device))
            # y2 = self.n_step(m, B_ext, m0, B_ext_0, st, tr.tensor(tn,device=self.dt.device))
            y_all.append(y2)
        y = tr.cat(y_all, dim=1)
        return y

    def B_demag_2D(self, M):
        nx, ny = self.geom.dim[0], self.geom.dim[1]
        

        M_ = tr.nn.functional.pad(M, (0, ny, 0, nx))
        M_fft = tr.rfft(M_, 2, onesided=False)

        B_demag = tr.irfft(tr.stack(
            [tr.sum(complex_multiply(tr.stack([self.Kxx_fft, self.Kxy_fft],1), M_fft[:,0:2,]),1),
             tr.sum(complex_multiply(tr.stack([self.Kxy_fft, self.Kyy_fft],1), M_fft[:,0:2,]),1),
             complex_multiply(self.Kzz_fft, M_fft[:,2,])], 1),2,onesided=False)

        return np.pi*4e-7*B_demag[...,nx-1:2*nx-1,ny-1:2*ny-1]
        
    def B_exch(self, M):
        return nan2num(2*self.A_exch/(self.Msat_copy**2) * self.LAPLACE(M))

    def B_eff(self, M, B_ext):
        return B_ext + self.B_exch(M) + self.B_demag_2D(M)

    def euler_step_LLG(self, M, B_ext):
        return self.normalize(M + (self.gamma_LL*self.dt) * self.torque_LLG(M, B_ext))

    def n_step_relax(self, M, B_ext):
        for n in range(self.Nr):
            M = checkpoint(self.rk4_step_LLG, M, B_ext)
        return M

    def n_step(self, M, B_ext, M0, B_ext_0, signal,tn,output_fields=False):
        y_all = []
        # Loop through time
        t2 = 0
        for tt, st in enumerate(signal.chunk(signal.size(1), dim=1)):
            # Inject source(s)
            B_ext = B_ext.clone()
            for si, src in enumerate(self.sources):
                B_ext[src.dim, src.x, src.y] = B_ext_0[src.dim, src.x, src.y] + st[0,0,si]
#                 B_ext = src(B_ext, st)

            # Propagate the fields
            # Checkpointing saves memory at the expense of some computing time
            M = checkpoint(self.rk4_step_LLG, M, B_ext)

            if len(self.probes) > 0 and not output_fields:
                # Measure probe(s)
                probe_values = []
                for probe in self.probes:
                    probe_values.append(probe(M-M0))
                if tn >= 0:
                    y_all.append(tr.stack(probe_values, dim=-1))
            else:
                # No probe, so just return the fields
                if tn >= 0*(self.N/2-1):
                    y_all.append((M-M0)[:, 1, ])
        if y_all:
            y = tr.stack(y_all, dim=1)   # Combine outputs into a single tensor
        else:
            y = tr.empty(0,device=self.dt.device)
        
        return y, M

    def rk4_step_LLG(self, M, B_ext):
        h = self.gamma_LL * self.dt  # this time unit is closer to 1
        k1 = self.torque_LLG(M, B_ext)
        k2 = self.torque_LLG(M + h*k1/2, B_ext)
        k3 = self.torque_LLG(M + h*k2/2, B_ext)
        k4 = self.torque_LLG(M + h*k3, B_ext)
        return (M + h/6 * ((k1 + 2*k2) + (2*k3 + k4)))

    def torque_LLG(self, M, B_ext):
        m_x_Beff = tr.cross(M, self.B_eff(M, B_ext))
        if self.relax:
            return -(1 / (1 + 0.5**2) * (m_x_Beff + 0.5*tr.cross(nan2num(M/self.Msat_copy), m_x_Beff)))
        else:
            return -(1 / (1 + self.Alpha**2) * (m_x_Beff + self.Alpha*tr.cross(nan2num(M/self.Msat_copy), m_x_Beff)))

    def normalize(self, M):
        norm = tr.sqrt(M[:, 0, ]**2 + M[:, 1, ]**2 + M[:, 2, ]**2)
        return nan2num(M/(norm.repeat(1, 3, 1, 1))*self.Msat_copy)