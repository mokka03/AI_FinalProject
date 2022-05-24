import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import spintorch
from .geom import WaveGeometry, WaveGeometryFreeForm, WaveGeometryMs
from .cell import WaveCell

import warnings
warnings.filterwarnings("ignore")

bbox_white = {'boxstyle': 'round,pad=0.3',
              'fc': 'white',
              'ec': 'none',
              'alpha': 0.75}

color_dim = {'light': '#cccccc',
             'dark': '#555555'}


color_highlight = '#a1d99b'

def plot_loss(loss_iter, plotdir):
    fig = plt.figure()
    plt.plot(loss_iter, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy loss")
    fig.savefig(plotdir+'cross_entropy_loss.png')
    
def plot_output(u, p, epoch, plotdir):
    fig = plt.figure()
    plt.bar(range(1,1+u.size()[0]), u.detach().cpu().squeeze(), color='k')
    plt.xlabel("output number")
    plt.ylabel("output (normalized)")
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'output_epoch%d_X%d.png' % (epoch, p))


def total_field(model, yb, ylabel=None, block=False, ax=None, fig_width=4, cbar=True, cax=None, vmin=1e-3, vmax=1.0):
    """Plot the total (time-integrated) field over the computatonal domain for a given vowel sample
    """
    with torch.no_grad():
        y_tot = torch.abs(yb)[:,0:int(yb.size(1)*0.9)].pow(2).sum(dim=1)

        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)

        Z = y_tot[0, :, :].cpu().numpy().transpose()
        Z = Z / Z.max()
        h = ax.imshow(Z, cmap=plt.cm.magma, origin="bottom", norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        if cbar:
            if cax is None:
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("top", size="5%", pad="20%")
            if vmax < 1.0:
                extend = 'both'
            else:
                extend = 'min'
            plt.colorbar(h, cax=cax, orientation='horizontal', label=r"$\sum_t{ { u_t{\left(x,y\right)} }^2 }$")

        geometry(model, ax=ax, outline=True, outline_pml=True, vowel_probe_labels=None, highlight_onehot=ylabel,
                 bg='dark', alpha=0.5,markercolor='w')

        ax.set_xticks([])
        ax.set_yticks([])

        if ax is not None:
            plt.show(block=block)


def _plot_probes(probes, ax, vowel_probe_labels=None, highlight_onehot=None, bg='light',color="k"):
    markers = []
    for i, probe in enumerate(probes):
        if highlight_onehot is None:
            color_probe = color
        else:
            color_probe = color_highlight if highlight_onehot[0, i].item() == 1 else color_dim[bg]

        marker = probe.plot(ax, color=color_probe)
        markers.append(marker)

    return markers


def _plot_sources(sources, ax, bg='light'):
    markers = []
    for i, source in enumerate(sources):
        marker = source.plot(ax)
        markers.append(marker)

    return markers


def geometry(input, ax=None, outline=False, outline_pml=True, vowel_probe_labels=None, highlight_onehot=None, bg='light',
            alpha=1.0, cbar=False, saveplot=False, epoch=0, plotdir='', markercolor='k'):
    """Plot the spatial distribution of the wave speed
    """
    lc = '#000000' if bg == 'light' else '#ffffff'

    if isinstance(input, WaveGeometry):
        input()
        geom = input
        probes = None
        source = None
    elif isinstance(input, WaveCell):
        input.geom()
        geom = input.geom
        probes = input.probes
        sources = input.sources
        A = input.Alpha[0, 0, ].squeeze()
        alph = input.alpha.cpu().numpy()
    else:
        raise ValueError("Invalid input for plot.geometry(); should be either a WaveGeometry or a WaveCell")

    B = geom.B[2,].detach().cpu().numpy().transpose()

    # Make axis if needed
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    markers = []
    if not outline:
        if isinstance(input.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            h1 = ax.imshow(Msat, origin="bottom", rasterized=True, cmap=plt.cm.Greens)
            plt.colorbar(h1, ax=ax, label='Saturation magnetization (A/m)')
        elif isinstance(input.geom, WaveGeometryFreeForm):
            h1 = ax.imshow(B*1e3, origin="bottom", rasterized=True, cmap=plt.cm.Greens)
            plt.colorbar(h1, ax=ax, label='Magnetic field (mT)')
    else:
        h  = ax.contour(B, levels=[B[0,0]-0.01], colors=['#ffffff'], linewidths=[0.75], alpha=alpha)
        h5 = ax.contour(B, levels=[B[0,0]+0.01], colors=['#000000'], linewidths=[0.75], alpha=alpha)

    if outline_pml:
        b_boundary = A.cpu().numpy().transpose()
        h2 = ax.contour(b_boundary > alph, levels=[0], colors=[lc], linestyles=['dotted'], linewidths=[0.75], alpha=alpha)

    if probes is not None:
        markers += _plot_probes(probes, ax, vowel_probe_labels=vowel_probe_labels, highlight_onehot=highlight_onehot,
                                bg=bg,color=markercolor)

    if sources is not None:
        markers += _plot_sources(sources, ax, bg=bg)

    ax.set_xticks([])
    ax.set_yticks([])

    if show:
        plt.show()
        
    if saveplot:
        fig.savefig(plotdir+'geometry_epoch%d.png' % (epoch))



def field_snapshot(model, fields, times, ylabel=None, fig_width=7, block=False,
                   axs=None, label=True, cbar=True, Ny=1, sat=1.0):
    """Plot snapshots in time of the scalar wave field
    """
    field_slices = fields[0, times, :, :]
    if isinstance(model.geom, WaveGeometryMs):
        field_slices = field_slices/model.geom.Msat
    elif isinstance(model.geom, WaveGeometryFreeForm):
        field_slices = field_slices/model.geom.Ms


    if axs is None:
        Nx = int(len(times) / Ny)
        fig, axs = plt.subplots(Ny, Nx, constrained_layout=True)

    axs = np.atleast_1d(axs)
    axs = axs.ravel()

    field_max = field_slices.max().item()
    field_min = field_slices.min().item()
    field_max = max(abs(field_min), field_max)
    for i, time in enumerate(times):
        field = field_slices[i, :, :].cpu().numpy().transpose()
        h = axs[i].imshow(field, cmap=plt.cm.RdBu, vmin=-sat * field_max, vmax=+sat * field_max, origin="bottom",
                          rasterized=True)
        geometry(model, ax=axs[i], outline=True, outline_pml=True, highlight_onehot=ylabel, bg='light')

        axs[i].set_xticks([])
        axs[i].set_yticks([])

        if label:
            axs[i].text(0.5, 0.03, "time step %d/%d" % (time+1, fields.shape[1]),
                        transform=axs[i].transAxes, ha="center",
                        va="bottom", bbox=bbox_white, fontsize='smaller')

    if cbar:
        plt.colorbar(h, ax=axs, label=r"$m_y$", shrink=0.80)

    for j in range(i + 1, len(axs)):
        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].axis('image')
        axs[j].axis('off')

    plt.show(block=block)
    return axs