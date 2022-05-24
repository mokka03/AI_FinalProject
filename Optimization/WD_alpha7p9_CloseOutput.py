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
alpha = 7e-4    # damping coefficient ()
gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
A_exch = 3.65e-12       # exchange coefficient (J/m)

dt = 20e-12     # timestep (s)
f1 = 3e9        # source frequency 1 (Hz)
f2 = 2.977e9    # source frequency 2 (Hz)
timesteps = 5000 # number of timesteps for wave propagation

'''Directories'''
basedir = 'WD_alpha7p9_CloseOutput/'
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
Np = 31  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-40, int(ny*(p+1)/(Np+1)), 5))
model = spintorch.WaveCell(geom, dt, Ms, gamma_LL, alpha, A_exch, src, probes)

dev = torch.device('cuda')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU


'''Define the source signal'''
## 2 input frequencies
t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
X1 = Bt*torch.sin(2*np.pi*f1*t)
X2 = Bt*torch.sin(2*np.pi*f2*t)

INPUTS = torch.cat((X1, X2))  # here we could cat multiple inputs
print(INPUTS.shape)
OUTPUTS = torch.tensor([14,16]).to(dev) # desired output


'''Define optimizer and criterion'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.000172)
criterion = torch.nn.CrossEntropyLoss(reduction='sum')


'''Load checkpoint'''
epoch_init = -1 # select previous checkpoint (-1 = don't use checkpoint)
epoch = epoch_init
if epoch_init>=0:
    checkpoint = torch.load(savedir + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []
    
'''Train the network'''
for epoch in range(epoch_init+1, 20):
    loss_batch = 0
    for i in range(0, INPUTS.size()[0]):
        def closure():
            optimizer.zero_grad()
            u = spintorch.utils.normalize_power(model(INPUTS[i:i+1]).sum(dim=1))
            spintorch.plot.plot_output(u[0,], OUTPUTS[i]+1, epoch, plotdir)
            loss = criterion(u,OUTPUTS[i:i+1])
            stat_cuda('after criterion')
            loss.backward()
            return loss
    
        loss = optimizer.step(closure)
        print("Epoch: %d -- Loss: %.6f" % (epoch, loss))
        t1 = toc(t0, t1)
        loss_batch += loss.item()
        
        spintorch.plot.geometry(model, cbar=True, saveplot=True,
                                    epoch=epoch, plotdir=plotdir)
        
    loss_iter.append(loss_batch/INPUTS.size()[0])  # store loss values
    spintorch.plot.plot_loss(loss_iter, plotdir)
    
    stat_cuda('after training')
        
    '''Save model checkpoint'''
    torch.save({
                'epoch': epoch,
                'loss_iter': loss_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, savedir + 'model_e%d.pt' % (epoch))


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


    Msat = model.geom.Msat.detach()
    savemat("Msat_s_WD_alpha7p9_CloseOutput.mat", {"Msat": Msat.to(torch.device("cpu")).numpy().transpose()})
    m = u_field[0,timesteps-1,]
    savemat("m_s_WD_alpha7p9_CloseOutput.mat", {"m": m.to(torch.device("cpu")).numpy().transpose()})
    print('--------------------------------------------------')
    print(torch.min(Msat))
    print(torch.max(Msat))
    # u = spintorch.utils.normalize_power(model(INPUTS).sum(dim=1))
    # print(u[0,])