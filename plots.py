import matplotlib as mpl
from matplotlib.patches import Ellipse
import pylab as plt
import numpy as np


def plot_ellipse(pos, cov, dashed):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    ls = 'dashed' if dashed else 'solid'

    width, height = 4 * np.sqrt(vals)
    ellip = mpl.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, lw=1, fill=True, alpha=0.35,
                                linestyle=ls)

    ax.add_artist(ellip)


def x_y_z_plot(means, covs, x):

    plt.figure(1, figsize=(12, 20))
    plt.title('G-16')

    # x-y

    plt.subplot(211)
    plt.scatter(x[:, 0], x[:, 1], c='grey', s=6)

    for i in range(means.shape[0]):
        plt.scatter(means[i, 0], means[i, 1], c='r')
        plot_ellipse(means[i, 0:2], covs[i, 0:2, 0:2], True)

    plt.xlabel('X')
    plt.ylabel('Y')

    # y -z

    plt.subplot(212)
    plt.scatter(x[:, 1], x[:, 2], c='grey', s=6)

    for i in range(means.shape[0]):
        plt.scatter(means[i, 1], means[i, 2], c='r')
        plot_ellipse(means[i, 1:3], covs[i, 1:3, 1:3], True)

    plt.xlabel('Y')
    plt.ylabel('Z')

    plt.show()