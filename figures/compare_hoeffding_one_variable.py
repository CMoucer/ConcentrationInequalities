######################################################################################################################
# This file computes an upper bound to Hoeffding's inequality in the variational approach for two blocks of random
# variables with different means. It corresponds to Figure 1 of the paper.
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from compute_concentration.compute_concentration_classical_bounds import compute_hoeffding_bound
from compute_concentration.compute_concentration_hoeffding import compute_hoeffding_exact_univariate, compute_hoeffding_exponential

# As a function of mu if "plot_option = 'mu'", and as function of t if "plot_option = 't'".
plot_option = 'mu' # or 't'

## With the pl_option 't', we plot the bounds as functions of t
if plot_option == 't':
    mu = 0.4 # mean
    n = 1 # number of variables
    t_max = 1 - mu - 0.05
    nb = 50
    ts = np.linspace(0, t_max, nb) # possible deviation parameters

    bound_hoeffding = np.zeros(nb)
    bound_exponential = np.zeros(nb)
    bound_exact = np.zeros(nb)

    for i in range(nb):
        bound_hoeffding[i] = compute_hoeffding_bound(ts[i], n)
        bound_exponential[i] = compute_hoeffding_exponential(mu, ts[i], n)
        bound_exact[i] = compute_hoeffding_exact_univariate(mu, ts[i])

    plt.plot(ts, bound_hoeffding, color='green', label="Hoeffding\'s inequality")
    plt.plot(ts, bound_exponential, color='blue', label='separable')
    plt.plot(ts, bound_exact, color='red', label='exact')

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.xlabel('deviation t', fontsize=16)
    plt.ylabel(r'concentration bound $\rho$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.show()


## With the plot_option mu, we plot concentration as a function of the mean mu.
elif plot_option == 'mu':
    t = 0.3 # mean
    n = 1 # number of variables
    mu_max = 1 - t - 0.05
    nb = 50
    mus = np.linspace(0.05, mu_max, nb) # possible deviation parameters

    bound_hoeffding = np.zeros(nb)
    bound_exponential = np.zeros(nb)
    bound_exact = np.zeros(nb)

    for i in range(nb):
        bound_hoeffding[i] = compute_hoeffding_bound(t, n)
        bound_exponential[i] = compute_hoeffding_exponential(mus[i], t, n)
        bound_exact[i] = compute_hoeffding_exact_univariate(mus[i], t)

    plt.plot(mus, bound_hoeffding, color='green', label="Hoeffding\'s inequality")
    plt.plot(mus, bound_exponential, color='blue', label='exponential')
    plt.plot(mus, bound_exact, color='red', label='exact')

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.xlabel(r'first-order moment $\mu$', fontsize=16)
    plt.ylabel(r'concentration bound $\rho$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.show()