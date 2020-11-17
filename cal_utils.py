"""Utility functions for the calibration notebooks of this repository"""


import numpy as np
from matplotlib import pyplot as plt


def waterfall_column(waterfalls, flags, titles, ylabel, clims=None, clabels=None, cmaps=None, \
                     ylims=None, extents=None, hspace=.1, figsize=(12,6), dpi=100):
    """Useful plotting function for the IDR2.2 memorandum"""
    if clims is None:
        clims = [None for i in range(len(waterfalls))]
    if clabels is None:
        clabels = [None for i in range(len(waterfalls))]
    if cmaps is None:
        cmaps = [None for i in range(len(waterfalls))]
    if ylims is None:
        ylims = [None for i in range(len(waterfalls))]
    if not any(isinstance(ex, list) for ex in extents):
        extents = [extents for i in range(len(waterfalls))]

    fig, axes = plt.subplots(len(waterfalls), 1, sharex=True, squeeze=True, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(hspace=hspace)
    for ax, wf, f, t, clim, clabel, cmap, ylim, ex in zip(axes, waterfalls, flags, titles,
                                                          clims, clabels, cmaps, ylims, extents):
        with np.errstate(divide='ignore', invalid='ignore'):
            im = ax.imshow(wf / ~f, aspect='auto', extent=ex, cmap=cmap)
        plt.colorbar(im, ax=ax, label=clabel, aspect=8, pad=.025)
        if ax == axes[-1]:
            ax.set_xlabel('Frequency (MHz)')
        im.set_clim(clim)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.9, t, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.show()


def plot_vis(data, flags, hd, JD, bl, vcomp, title=None, vmax=.1, figsize=(12,3), dpi=100):
    if vcomp == 'amp':
        label = 'Amplitude'
        vcalc = np.absolute
    elif vcomp == 'phase':
        label = 'Phase (Radians)'
        vcalc = np.angle
        vmax = None
    else:
        raise ValueError('Specify either {"amp", "phase"} for vcomp')
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(vcalc(data[bl]) / (~flags[bl]), aspect='auto', cmap='twilight',
               extent=[hd.freqs[0]/1e6, hd.freqs[-1]/1e6, hd.times[-1]-int(JD), hd.times[0]-int(JD)], \
               vmax=vmax)
    plt.ylabel('JD - {}'.format(int(JD)))
    plt.xlabel('Frequency (MHz)')
    plt.title(title + ': ' + str(bl))
    plt.colorbar(label=label, aspect=8, pad=.025)
    plt.show()
