"""
Created on Mon May 11 21:29:36 2020

@author: Adam Papp
based on:
Ru Zhu, Accelerate micromagnetic simulations with GPU programming in MATLAB
https://arxiv.org/ftp/arxiv/papers/1501/1501.07293.pdf
"""

from numpy import pi, log, arctan, zeros, sqrt, stack
from numba import jit
import torch as tr
from torch.fft import fft2, ifft2, fftn, ifftn

@jit(nopython=True)
def demag_tensor(nx, ny, nz, dx, dy, dz, z_off=0):
    """
    Calculates the demagnetization tensor.
    Numba is used to accelerate the calculation.
    Inputs: nx, ny, nz number of cells in x/y/z, dx cellsize
    Outputs: demag tensor elements (numpy.array)
    """
    # Initialization of demagnetization tensor
    Kxx = zeros((nx*2, ny*2, nz*2))
    Kyy = zeros((nx*2, ny*2, nz*2))
    Kzz = zeros((nx*2, ny*2, nz*2))
    Kxy = zeros((nx*2, ny*2, nz*2))
    Kxz = zeros((nx*2, ny*2, nz*2))
    Kyz = zeros((nx*2, ny*2, nz*2))

    for K in range(-nz+1+z_off, nz+z_off):
        for J in range(-ny+1, ny):
            for I in range(-nx+1, nx):
                L, M, N = (I+nx-1), (J+ny-1), (K+nz-1-z_off)  # non-negative indices
                for i in (-0.5, 0.5):
                    for j in (-0.5, 0.5):
                        for k in (-0.5, 0.5):
                            sgn = (-1)**(i+j+k+1.5)/(4*pi)
                            r = sqrt(((I+i)*dx)**2 + ((J+j)*dy)**2 + ((K+k)*dz)**2)
                            Kxx[L, M, N] += sgn * arctan((K+k)*(J+j)*dz*dy/(r*(I+i)*dx))
                            Kyy[L, M, N] += sgn * arctan((I+i)*(K+k)*dx*dz/(r*(J+j)*dy))
                            Kzz[L, M, N] += sgn * arctan((J+j)*(I+i)*dy*dx/(r*(K+k)*dz))
                            Kxy[L, M, N] -= sgn * log(abs((K+k)*dz + r))
                            Kxz[L, M, N] -= sgn * log(abs((J+j)*dy + r))
                            Kyz[L, M, N] -= sgn * log(abs((I+i)*dx + r))

    return Kxx, Kyy, Kzz, Kxy, Kxz, Kyz 


@jit(nopython=True)
def demag_tensor_2D(nx, ny, dx, dy, dz):
    """
    Calculates the demagnetization tensor for 2D problems.
    Numba is used to accelerate the calculation.
    Inputs: nx, ny, nz number of cells in x/y/z, dx cellsize
    Outputs: demag tensor elements (numpy.array)
    """
    # Initialization of demagnetization tensor
    Kxx = zeros((nx*2, ny*2))
    Kyy = zeros((nx*2, ny*2))
    Kzz = zeros((nx*2, ny*2))
    Kxy = zeros((nx*2, ny*2))
    Kxz = zeros((nx*2, ny*2))
    Kyz = zeros((nx*2, ny*2))
    K = 0
    for J in range(-ny+1, ny):
        for I in range(-nx+1, nx):
            L, M = (I+nx-1), (J+ny-1)  # non-negative indices
            for i in (-0.5, 0.5):
                for j in (-0.5, 0.5):
                    for k in (-0.5, 0.5):
                        sgn = (-1)**(i+j+k+1.5)/(4*pi)
                        r = sqrt(((I+i)*dx)**2 + ((J+j)*dy)**2 + ((K+k)*dz)**2)
                        Kxx[L, M] += sgn * arctan((K+k)*(J+j)*dz*dy/(r*(I+i)*dx))
                        Kyy[L, M] += sgn * arctan((I+i)*(K+k)*dx*dz/(r*(J+j)*dy))
                        Kzz[L, M] += sgn * arctan((J+j)*(I+i)*dy*dx/(r*(K+k)*dz))
                        Kxy[L, M] -= sgn * log(abs((K+k)*dz + r))
                        Kxz[L, M] -= sgn * log(abs((J+j)*dy + r))
                        Kyz[L, M] -= sgn * log(abs((I+i)*dx + r))

    return Kxx, Kyy, Kzz, Kxy, Kxz, Kyz


def demag_tensor_fft(nx, ny, nz, dx, dy, dz, dev, z_off=0):
    """
    Returns demagnetization kernel in Fourier space.
    Inputs: nx, ny, nz number of cells in x/y/z, dx, dy, dz cellsize
    Outputs: demag tensor elements stacked (torch.tensor)
    """
    Kxx, Kyy, Kzz, Kxy, Kxz, Kyz = demag_tensor(nx, ny, nz, dx, dy, dz, z_off)

    Kx_fft = tr.view_as_real(fftn(tr.tensor(stack((Kxx[:,:,:], Kxy[:,:,:], Kxz[:,:,:]),0),
                                     device=dev, dtype=tr.float32).unsqueeze(0), dim=3))
    Ky_fft = tr.view_as_real(fftn(tr.tensor(stack((Kxy[:,:,:], Kyy[:,:,:], Kyz[:,:,:]),0),
                                     device=dev, dtype=tr.float32).unsqueeze(0), dim=3))
    Kz_fft = tr.view_as_real(fftn(tr.tensor(stack((Kxz[:,:,:], Kyz[:,:,:], Kzz[:,:,:]),0),
                                     device=dev, dtype=tr.float32).unsqueeze(0), dim=3))
    return Kx_fft, Ky_fft, Kz_fft


def demag_tensor_fft_2D(nx, ny, dx, dy, dz, dev):
    """
    Returns demagnetization kernel in Fourier space.
    Symmetries in 2D: Kyx=Kxy, Kxz=Kzx=Kyz=Kzy=0
    Inputs: nx, ny, nz number of cells in x/y/z, dx cellsize
    Outputs: demag tensor elements (exploiting symmetry) (torch.tensor)
    """
    Kxx, Kyy, Kzz, Kxy, Kxz, Kyz = demag_tensor_2D(nx, ny, dx, dy, dz)

    Kxx_fft = tr.view_as_real(fft2(tr.tensor(Kxx, device=dev, dtype=tr.float32
                                ).unsqueeze(0)))
    Kyy_fft = tr.view_as_real(fft2(tr.tensor(Kyy, device=dev, dtype=tr.float32
                                ).unsqueeze(0)))
    Kzz_fft = tr.view_as_real(fft2(tr.tensor(Kzz, device=dev, dtype=tr.float32
                                ).unsqueeze(0)))
    Kxy_fft = tr.view_as_real(fft2(tr.tensor(Kxy, device=dev, dtype=tr.float32
                                ).unsqueeze(0)))
    Kxz_fft = tr.view_as_real(fft2(tr.tensor(Kxz, device=dev, dtype=tr.float32
                                ).unsqueeze(0)))
    Kyz_fft = tr.view_as_real(fft2(tr.tensor(Kyz, device=dev, dtype=tr.float32
                                ).unsqueeze(0)))
    
    return Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft
#cella x magnesezettsegehogy hat az x,re; y, y-ra, z z-re, x, y-ra:
#szimmetria miatt a tobbi kiesik

def complex_multiply(C1, C2):
    # Complex multiplication by hand (a+ib)(c+id)=(ac-bd)+i((a+b)(c+d)-(ac+bd))
    # Last dim is complex (given by torch.rfft )
    AC_BD = C1*C2   # ac and bd
    M_i = C1.sum(-1)*C2.sum(-1) - AC_BD.sum(-1)
    M_r = AC_BD.select(-1, 0) - AC_BD.select(-1, 1) 
    return tr.stack([M_r, M_i], -1)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from utils import tic, toc, stat_cuda
    from scipy.io import savemat

    mpl.rcParams['figure.dpi'] = 600    # Produce decent-quality figures

    # number of cells on x/y/z direction
    nx = 200
    ny = 200
    nz = 1
    dx = 25
    dy = 25
    dz = 25
    Ms = 140e3
    mu0 = 4*pi*1e-7

    dev = tr.device('cuda')  # 'cuda' or 'cpu'

    stat_cuda('start')
    t0 = t1 = tic()



    """Calculate demag tensor 2D"""
    Kxx, Kyy, Kzz, Kxy, Kxz, Kyz = demag_tensor_2D(nx, ny, dx, dy, dz)
    # print("demag_tensor done")
    # t1 = toc(t0, t1)

    Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft = demag_tensor_fft_2D(nx, ny, dx, dy, dz, dev)

    stat_cuda("demag_tensor_fft_2D done")
    t1 = toc(t0, t1)

    """Calculate demag field"""
    # Magnetization
    m = tr.zeros((1, 3, nx, ny), device=dev)
    m[:, 2, :, :] = 1  # set magnetization in z direction
    m_ = tr.nn.functional.pad(m, (0, ny, 0, nx))
    m_fft = tr.view_as_real(fft2(m_))

    stat_cuda("M initialization done")
    t1 = toc(t0, t1)

    KxM = complex_multiply(tr.stack([Kxx_fft, Kxy_fft],1), m_fft[:,0:2,])#temp variable
    Bx_demag = tr.view_as_real(ifft2(tr.view_as_complex(
        tr.sum(KxM,1))))[0,nx-1:2*nx-1,ny-1:2*ny-1,0]*Ms*mu0

    KxM = complex_multiply(tr.stack([Kxy_fft, Kyy_fft],1), m_fft[:,0:2,])
    By_demag = tr.view_as_real(ifft2(tr.view_as_complex(
        tr.sum(KxM,1))))[0,nx-1:2*nx-1,ny-1:2*ny-1,0]*Ms*mu0

    KxM = complex_multiply(Kzz_fft, m_fft[:,2,])
    Bz_demag = tr.view_as_real(ifft2(tr.view_as_complex(
        KxM)))[0,nx-1:2*nx-1,ny-1:2*ny-1,0]*Ms*mu0


    # # Complex multiplication by hand (a+ib)(c+id)=(ac-bd)+i((a+b)(c+d)-(ac+bd))
    # # Last dim is complex (given by torch.rfft )
    # AC_BD = tr.stack([Kxx_fft, Kxy_fft],1)*m_fft[:,0:2,]
    # KxM_i = tr.stack([Kxx_fft, Kxy_fft],1).sum(4)*m_fft[:,0:2,].sum(4) - AC_BD.sum(4)
    # KxM_r = AC_BD[:, :, :, :, 0] - AC_BD[:, :, :, :, 1]
    # KxM = tr.stack([KxM_r, KxM_i], 4)
    # Bx_demag = tr.irfft(tr.sum(KxM,1),2,onesided=False
    #                     )[0,nx-1:2*nx-1,ny-1:2*ny-1]*Ms*mu0

    # AC_BD = tr.stack([Kxy_fft, Kyy_fft],1)*m_fft[:,0:2,]
    # KxM_i = tr.stack([Kxy_fft, Kyy_fft],1).sum(4)*m_fft[:,0:2,].sum(4) - AC_BD.sum(4)
    # KxM_r = AC_BD[:, :, :, :, 0] - AC_BD[:, :, :, :, 1]
    # KxM = tr.stack([KxM_r, KxM_i], 4)
    # By_demag = tr.irfft(tr.sum(KxM,1),2,onesided=False
    #                     )[0,nx-1:2*nx-1,ny-1:2*ny-1]*Ms*mu0

    # AC_BD = Kzz_fft*m_fft[:,2,]
    # KxM_i = Kzz_fft.sum(3)*m_fft[:,2,].sum(3) - AC_BD.sum(3)
    # KxM_r = AC_BD[:, :, :, 0] - AC_BD[:, :, :, 1]
    # KxM = tr.stack([KxM_r, KxM_i], 3)
    # Bz_demag = tr.irfft(KxM,2,onesided=False
    #                     )[0,nx-1:2*nx-1,ny-1:2*ny-1]*Ms*mu0

    stat_cuda("iFFT done")
    t1 = toc(t0, t1)

    print("Plotting..")

    fig_x, ax_x = plt.subplots()
    hx = ax_x.imshow(Bx_demag[:,:].squeeze().cpu().numpy())
    fig_x.colorbar(hx, ax=ax_x)

    fig_y, ax_y = plt.subplots()
    hy = ax_y.imshow(By_demag[:,:].squeeze().cpu().numpy())
    fig_y.colorbar(hy, ax=ax_y)

    fig_z, ax_z = plt.subplots()
    hz = ax_z.imshow(Bz_demag[:, :].squeeze().cpu().numpy())
    fig_z.colorbar(hz, ax=ax_z)
    plt.show()

#save kernel and demag
    savemat("spintorch_kernel.mat",{
    "spintorch_kernel_xx": Kxx,
    "spintorch_kernel_xy": Kxy,
    "spintorch_kernel_yy": Kyy,
    "spintorch_kernel_zz": Kzz}
    )

    savemat("spintorch_demag.mat",{
    "spintorch_demag_x": Bx_demag[:, :].squeeze().cpu().numpy(),
    "spintorch_demag_y": By_demag[:, :].squeeze().cpu().numpy(),
    "spintorch_demag_z": Bz_demag[:, :].squeeze().cpu().numpy()}
    )







    # """Calculate demag tensor 3D"""
    # # Kxx, Kyy, Kzz, Kxy, Kxz, Kyz = demag_tensor(nx, ny, nz, dx)
    # # print("demag_tensor done")
    # # t1 = toc(t0, t1)
    
    # Kx_fft, Ky_fft, Kz_fft = demag_tensor_fft(nx, ny, nz, dx, dev)
    # stat_cuda("demag_tensor_fft done")
    # t1 = toc(t0, t1)

    # """Calculate demag field"""
    # # Magnetization
    # m = tr.zeros((1, 3, nx, ny, nz), device=dev)
    # m[:, 2, :, :, 0] = 1  # set magnetization in z direction
    # m_ = tr.nn.functional.pad(m, (0, nz, 0, ny, 0, nx))
    # m_fft = tr.rfft(m_, 3, onesided=False)

    # stat_cuda("M initialization done")
    # t1 = toc(t0, t1)

    # # Complex multiplication by hand (a+ib)(c+id)=(ac-bd)+i((a+b)(c+d)-(ac+bd))
    # # Last dim is complex (given by torch.rfft )
    # AC_BD = Kx_fft*m_fft
    # KxM_i = Kx_fft.sum(5)*m_fft.sum(5) - AC_BD.sum(5)
    # KxM_r = AC_BD[:, :, :, :, :, 0] - AC_BD[:, :, :, :, :, 1]
    # KxM = tr.stack([KxM_r, KxM_i], 5)
    # Bx_demag = tr.irfft(tr.sum(KxM,1,True),3,onesided=False
    #                     )[0,0,nx-1:2*nx-1,ny-1:2*ny-1,0]*Ms*mu0

    # AC_BD = Ky_fft*m_fft
    # KxM_i = Ky_fft.sum(5)*m_fft.sum(5) - AC_BD.sum(5)
    # KxM_r = AC_BD[:, :, :, :, :, 0] - AC_BD[:, :, :, :, :, 1]
    # KxM = tr.stack([KxM_r, KxM_i], 5)
    # By_demag = tr.irfft(tr.sum(KxM,1,True),3,onesided=False
    #                     )[0,0,nx-1:2*nx-1,ny-1:2*ny-1,0]*Ms*mu0

    # AC_BD = Kz_fft*m_fft
    # KxM_i = Kz_fft.sum(5)*m_fft.sum(5) - AC_BD.sum(5)
    # KxM_r = AC_BD[:, :, :, :, :, 0] - AC_BD[:, :, :, :, :, 1]
    # KxM = tr.stack([KxM_r, KxM_i], 5)
    # Bz_demag = tr.irfft(tr.sum(KxM,1,True),3,onesided=False
    #                     )[0,0,nx-1:2*nx-1,nx-1:2*ny-1,0]*Ms*mu0

    # stat_cuda("iFFT done")
    # t1 = toc(t0, t1)

    # print("Plotting..")

    # fig_x, ax_x = plt.subplots()
    # hx = ax_x.imshow(Bx_demag.squeeze().cpu().numpy())
    # fig_x.colorbar(hx, ax=ax_x)

    # fig_y, ax_y = plt.subplots()
    # hy = ax_y.imshow(By_demag.squeeze().cpu().numpy())
    # fig_y.colorbar(hy, ax=ax_y)

    # fig_z, ax_z = plt.subplots()
    # hz = ax_z.imshow(Bz_demag[:, :].squeeze().cpu().numpy())
    # fig_z.colorbar(hz, ax=ax_z)
