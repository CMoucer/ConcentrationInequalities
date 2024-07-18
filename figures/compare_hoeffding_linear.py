x######################################################################################################################
# This file computes an upper bound to Hoeffding's inequality in the linear approach.
# It corresponds to Figure 4 of the paper.
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from compute_concentration.compute_concentration_hoeffding import compute_hoeffding_exponential, compute_variational_hoeffding
from compute_concentration.compute_extremal_points_hoeffding import compute_extremal_points


ns = np.array([2, 5, 10])
colors = np.flip(plt.cm.plasma(np.linspace(0, 1, 1+len(ns))), axis=0)
mu = 0.6
t_max = 1 - mu - 0.05
nb = 30
ts = np.linspace(0, 0.2, nb)

for j in range(len(ns)):
    bound_variational = np.zeros(nb)
    bound_exponential = np.zeros(nb)
    for i in range(nb):
        bound_exponential[i] = compute_hoeffding_exponential(mu, ts[i], ns[j])
        extremal_points = compute_extremal_points(mu, ts[i], ns[j])
        bound_variational[i] = compute_variational_hoeffding(extremal_points, mu)
    plt.plot(ts, bound_exponential, '--', color=colors[j+1])
    plt.plot(ts, bound_variational, color=colors[j+1], label=f'n={ns[j]}')

plt.plot(ts, mu/(mu+ts), color='black', label='linear')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.xlabel('deviation t', fontsize=16)
plt.ylabel(r'concentration bound $\rho$', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()