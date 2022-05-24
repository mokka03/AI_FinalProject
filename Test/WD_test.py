"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from spintorch.utils import tic, toc, stat_cuda
from scipy.io import savemat


mpl.use('Agg') # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

t0 = t1 = tic()


"""Parameters"""
dx = 156.25e-9      # discretization (m)
dy = 156.25e-9      # discretization (m)
dz = 69e-9      # discretization (m)
nx = 320        # size x    (cells)
ny = 640        # size y    (cells)

Ms = 1.43725e5      # saturation magnetization (A/m)
B0 = 283.4e-3     # bias field (T)
Bt = 1e-3       # excitation field amplitude (T)
alpha = 7.9e-4    # damping coefficient ()
gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
A_exch = 3.65e-12       # exchange coefficient (J/m)

dt = 20e-12     # timestep (s)
# f1 = 3e9        # source frequency 1 (Hz)
# f2 = 2.977e9    # source frequency 2 (Hz)
timesteps = 5000 # number of timesteps for wave propagation

'''Directories'''
basedir = 'WD_testBinary_closeOutput/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)

out_list = []
f_values = []

TestFrequencies = [2954, 2965, 2977, 2988, 3000, 3011, 3023]
for f_MHz in TestFrequencies:
    f1 = f_MHz*1e6
    '''Geometry, sources, probes, model definitions'''
    # geom = spintorch.WaveGeometryFreeForm((nx, ny), dx, dy, dz, B0, B1, Ms)
    geom = spintorch.WaveGeometryMs((nx, ny), dx, dy, dz, Ms, B0)
    src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=1)
    probes = []
    Np = 31  # number of probes
    for p in range(Np):
        probes.append(spintorch.WaveIntensityProbeDisk(nx-40, int(ny*(p+1)/(Np+1)), 5))
    model = spintorch.WaveCell(geom, dt, Ms, gamma_LL, alpha, A_exch, src, probes)

    dev = torch.device('cuda')  # 'cuda' or 'cpu'
    print('Running on', dev)
    model.to(dev)   # sending model to GPU/CPU


    '''Define the source signal'''
    t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
    X = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude

    INPUTS = X  # here we could cat multiple inputs
    OUTPUTS = torch.tensor([int(Np/2)]).to(dev) # desired output


    '''Plot wave propagation'''
    print("Wave propagation for plotting")

    epoch = 0
    with torch.no_grad():
        u_field = model(INPUTS, output_fields=True)
        # stat_cuda('after plot propagating')
        
        t1 = toc(t0, t1)

        m = u_field[0,timesteps-1,]
        m = m / model.geom.Msat
        savemat("m_Binary_%dMHz.mat" % (f_MHz), {"m": m.to(torch.device("cpu")).numpy().transpose()})
        
        # timesteps = u_field.size(1)
        Nx, Ny = 2, 2
        times = np.ceil(np.linspace(timesteps/Nx/Ny, timesteps, num=Nx*Ny))-1

        if f_MHz == 3500:            
            spintorch.plot.geometry(model, cbar=True, saveplot=True, epoch=epoch, plotdir=plotdir)
        
        spintorch.plot.field_snapshot(model, u_field, [timesteps-1],label=False)
        plt.gcf().savefig(plotdir+'field_snapshot_%dMHz.png' % (f_MHz))

        u = spintorch.utils.normalize_power(model(INPUTS).sum(dim=1))    
        spintorch.plot.plot_output(u[0,], OUTPUTS[0]+1, f_MHz, "MHz", plotdir)
        outs = u[0,].detach().cpu().numpy()
        # out = outs[int(Np/2)]
        out_list.append(outs)
        f_values.append(f_MHz)
    stat_cuda('befor del')
    del u_field
    torch.cuda.empty_cache()
    stat_cuda('after del')
    print('---------------------------------------')


out_array = np.array(out_list)
# f_array = np.array(f_values)
# f_out = np.zeros((2,f_array.shape[0]))
# f_out[0,] = f_array
# f_out[1,] = out_array

# savemat("f_out.mat", {"f_out": f_out})
savemat("outsBinary.mat", {"outs": out_array})