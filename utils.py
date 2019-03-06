from stats import log_multivariate_normal_density as log_pdf
from hmmlearn import hmm
from scipy.special import logsumexp
import numpy as np


def init_w_hmm(n_state, obs, length):

    """

    param n_state: Integer
    param obs: 2D Array
    param length: List

    return: Tuple

    """

    model = hmm.GaussianHMM(n_components=n_state, covariance_type='diag')

    model.fit(obs, length)
    return model.startprob_, model.means_, model.covars_


def log_normalize(a, axis=None):
    """Normalizes the input array so that the exponent of the sum is 1.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    with np.errstate(under="ignore"):
        a_lse = logsumexp(a, axis)
    a -= a_lse[:, np.newaxis]



def sum_normalize(arr, axis=None):

    """

    This function normalizes the input given in arguments
    such that sum of the elements in it will be equal to 1
    param arr: A non-normalized array

    param axis: Axis along which summation is performed

    return: None, instead modifies the given input **inplace**.

    """

    a_sum = arr.sum(axis)
    if axis and arr.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(arr.shape)
        shape[axis] = 1
        a_sum.shape = shape

    arr /= a_sum


def transition_prob(ls_data, mu_a, cov_a, mu_g, cov_g):

    """

        param ls_data: sequential data we read
        param mu_a: array_like, shape (n_components, n_features)
        param cov_a: array_like, shape (n_components, n_features, n_features)
        param mu_g: array_like, shape (n_components, n_features)
        param cov_g: array_like, shape (n_components, n_features, n_features)

        return: tuple of (3D Numpy,  3D Numpy) representing transition probabilities
                for action model and goal model, respectively.

    """

    s_a, s_g = mu_a.shape[0], mu_g.shape[0]
    tr_a, tr_g = np.zeros((s_a, s_g, s_a)), np.zeros((s_a, s_g, s_g))

    for i in ls_data:
        a = log_pdf(i[0], mu_a, cov_a).argmax(axis=1)
        g = log_pdf(i[1], mu_g, cov_g).argmax(axis=1)

        for t in range(len(list(a))-1):
            tr_a[a[t], g[t], a[t+1]] += 1
            tr_g[a[t], g[t], g[t+1]] += 1

    for i in range(s_a):
        for j in range(s_g):
            tot1 = np.sum(tr_a[i, j, :])
            tot2 = np.sum(tr_g[i, j, :])
            if tot1 > 0:
                for k in range(s_a):
                    tr_a[i, j, k] = tr_a[i, j, k]/tot1
                sum_normalize(tr_a[i, j, :])
            if tot2 > 0:
                for h in range(s_g):
                    tr_g[i, j, h] = tr_g[i, j, h]/tot2
                sum_normalize(tr_g[i, j, :])

    return tr_a, tr_g


if __name__ == "__main__":

    from dbn_32 import DBN
    import pandas as pd
    import numpy as np
    import plots

    """
    Read Data
    """

    obs_a = pd.read_csv('/home/alivelab/PycharmProjects/HADM/data/g32/Action_Keyframes.csv').values
    obs_g = pd.read_csv('/home/alivelab/PycharmProjects/HADM/data/g32/Goal_Keyframes.csv').values

    demos_a = np.split(obs_a, list(range(5, 100, 5)))
    demos_g = np.split(obs_g, list(range(5, 100, 5)))

    demos = [(demos_a[i], demos_g[i]) for i in range(len(demos_a))]

    kfs = [i[0].shape[0] for i in demos]
    num_demo = len(kfs)

    """
    Initialization of DBN
    """

    nstates_a, nstates_g = 5, 5

    model = DBN(nstates_a, nstates_g, demos, obs_a, obs_g, kfs, num_demo)

    model.fit(10)

    plots.x_y_z_plot(model.means_action, model.covars_action, obs_a)
