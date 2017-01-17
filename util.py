import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

def plot(player, msg='test'):
    plt.figure(figsize=[12, 18])
    plt.subplot(221)
    plt.imshow(
        np.max(player.Qsa, 2),
        # vmin=-1, vmax=1,
        extent=[1, 10, 1, 21],
        interpolation='none')
    plt.colorbar()
    plt.title('V(s) = max Q(s, a)')
    plt.subplot(222)
    plt.imshow(
        np.log10(np.sum(player.Nsa, 2) + 1),
        extent=[1, 10, 1, 21],
        interpolation='none')
    plt.colorbar()
    plt.title('log10 N(s, a)')
    plt.subplot(223)
    plt.imshow(
        player.Qsa[:, :, 0],
        # vmin=-1, vmax=1,
        extent=[1, 10, 1, 21],
        interpolation='none')
    plt.colorbar()
    plt.title('hit')
    plt.subplot(224)
    plt.imshow(
        player.Qsa[:, :, 1],
        # vmin=-1, vmax=1,
        extent=[1, 10, 21, 1],
        interpolation='none')
    plt.title('stick')
    plt.colorbar()
    plt.savefig(msg + '.png')

    # [TODO] one fig
    best_action = np.argmax(player.Qsa, 2)
    plt.figure()
    plt.imshow(
        best_action,
        cmap='gray',
        extent=[1, 10, 21, 1],
        interpolation='none')
    plt.title('black: hit; white: stick')
    plt.savefig(msg + 'best-act.png')
