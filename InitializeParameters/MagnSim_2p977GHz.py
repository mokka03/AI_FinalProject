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
dt = 20e-12     # timestep (s)
dx = 156.25e-9      # discretization (m)
dy = 156.25e-9      # discretization (m)
dz = 69e-9      # discretization (m)
B1 = 50e-3      # training field multiplier (T)
B0 = 283.4e-3     # bias field (T)
Bt = 1e-3       # excitation field (T)
# Ms = 1.43725e5      # saturation magnetization (A/m)
Ms = 1.4278e5
alpha = 7.9e-4    # damping coefficient ()
gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
A_exch = 3.65e-12       # exchange coefficient (J/m)

f1 = 2.977e9         # source frequency (Hz)
timesteps = 5000 # number of timesteps for wave propagation

nx = 320        # size x
ny = 640        # size y

'''Directories'''
basedir = 'MagnSim_f2/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
# geom = spintorch.WaveGeometryFreeForm((nx, ny), dx, dy, dz, B0, B1, Ms)
geom = spintorch.WaveGeometryMs((nx, ny), dx, dy, dz, Ms, B0)
src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=1)
probes = []
Np = 21  # number of probes
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

'''Define optimizer and criterion'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss(reduction='sum')


'''Load checkpoint'''
epoch_init = -1 # select previous checkpoint (-1 = don't use checkpoint)
epoch = epoch_init
if epoch_init>=0:
    checkpoint = torch.load('/home/maule/Spintorch_FIB/YIG_Dubs2/models/focus_YIGdubs2_50umx50um/' + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []
    


'''Plot wave propagation'''
print("Wave propagation for plotting")


with torch.no_grad():
    u_field = model(INPUTS, output_fields=True)
    stat_cuda('after plot propagating')
    
    t1 = toc(t0, t1)
    
    # timesteps = u_field.size(1)
    Nx, Ny = 2, 2
    times = np.ceil(np.linspace(timesteps/Nx/Ny, timesteps, num=Nx*Ny))-1

    spintorch.plot.field_snapshot(model, u_field, times, Ny=Ny)
    plt.gcf().savefig(plotdir+'field_4snapshots_epoch%d.png' % (epoch))
    
    spintorch.plot.total_field(model, u_field,cbar=False)
    plt.gcf().savefig(plotdir+'total_field_epoch%d.png' % (epoch))
    
    spintorch.plot.field_snapshot(model, u_field, [timesteps-1],label=False)
    plt.gcf().savefig(plotdir+'field_snapshot_epoch%d.png' % (epoch))
    
    spintorch.plot.geometry(model, cbar=True, saveplot=True, epoch=epoch, plotdir=plotdir)


    # Msat = model.geom.Msat.detach()
    # savemat("Msat_YIGdubs2_50umx50um.mat", {"Msat": Msat.to(torch.device("cpu")).numpy().transpose()})
    # m = u_field[0,4999,]
    u_field_ = u_field[0,timesteps-1,]
    savemat("m_s_50umx100um_2p977GHz.mat", {"m": u_field_.to(torch.device("cpu")).numpy().transpose()})
    # print('--------------------------------------------------')
    # print(torch.min(Msat))
    # print(torch.max(Msat))