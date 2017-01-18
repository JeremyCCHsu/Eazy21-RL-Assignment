# -*- coding: utf8 -*-

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

def plot(player, msg='test'):

    plt.figure(figsize=[12, 24])

    plt.subplot(321)
    plt.imshow(
        np.max(player.Qsa, 2),
        vmin=-1, vmax=1,
        extent=[1, 10, 1, 21],
        interpolation='none')
    plt.colorbar()
    plt.title('V(s) = max Q(s, a)')

    plt.subplot(322)
    best_action = np.argmax(player.Qsa, 2)
    plt.imshow(
        best_action,
        cmap='gray',
        extent=[1, 10, 21, 1],
        interpolation='none')
    plt.title('Best policy: black=hit; white=stick')

    plt.subplot(323)
    plt.imshow(
        player.Qsa[:, :, 0],
        vmin=-1, vmax=1,
        extent=[1, 10, 1, 21],
        interpolation='none')
    plt.colorbar()
    plt.title('hit')

    plt.subplot(324)
    plt.imshow(
        player.Qsa[:, :, 1],
        vmin=-1, vmax=1,
        extent=[1, 10, 21, 1],
        interpolation='none')
    plt.colorbar()
    plt.title('stick')
    
    plt.subplot(325)
    plt.imshow(
        np.log10(np.sum(player.Nsa, 2) + 1),
        extent=[1, 10, 1, 21],
        interpolation='none')
    plt.colorbar()
    plt.title('log10 N(s, a)')
    
    plt.savefig(msg + '.png')
    plt.close()
