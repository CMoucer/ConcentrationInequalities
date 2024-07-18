######################################################################################################################
# This file computes an upper bound to Hoeffding's inequality in the variational approach for two blocks of random
# variables with different means. It corresponds to Figure 3 of the paper.
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from compute_concentration.compute_concentration_hoeffding import compute_variational_hoeffding_two_blocks, compute_hoeffding_exponential_means
from compute_concentration.compute_extremal_points_hoeffding import compute_extremal_points_two_blocks
from compute_concentration.compute_concentration_classical_bounds import compute_hoeffding_bound
from compute_concentration.compute_extremal_points_hoeffding import preprocessing_mu, compute_mus_two_blocks


mu2 = 0.6
mu1 = 0.4
n = 10
ids = 5
#assert ids <= 2
mus = compute_mus_two_blocks(n, ids, mu1, mu2)
mus, idx = preprocessing_mu(mus)
print(mus)

nb = 30
t_max = 1 - mu2 - 0.05
ts = np.linspace(0, t_max, nb)

bound_hoeffding = np.zeros(nb)
bound_variational = np.zeros(nb)
bound_exponential = np.zeros(nb)

for i in range(nb):
    bound_hoeffding[i] = compute_hoeffding_bound(ts[i], n)
    extremal_points = compute_extremal_points_two_blocks(mus, ts[i], n, idx)
    bound_variational[i] = compute_variational_hoeffding_two_blocks(extremal_points, mus, idx)
    bound_exponential[i] = compute_hoeffding_exponential_means(mus, ts[i], n)

plt.plot(ts, bound_hoeffding, color='green', label="Hoeffding\'s inequality")
plt.plot(ts, bound_exponential, color='blue', label='exponential')
plt.plot(ts, bound_variational, color='red', label='variational')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.xlabel('deviation t', fontsize=16)
plt.ylabel(r'concentration bound $\rho$', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.savefig(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/output/hoeffding/two_blocks_comparison_mu1_{mu1}_mu2_{mu2}_n_{n}_m_{ids}.pdf', dpi=250)#, bbox_inches='tight')
plt.show()