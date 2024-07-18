import numpy as np
import matplotlib.pyplot as plt

from compute_concentration.compute_concentration_hoeffding import compute_variational_hoeffding_two_blocks, compute_hoeffding_exponential_means
from compute_concentration.compute_extremal_points_hoeffding import compute_extremal_points_two_blocks
from compute_concentration.compute_extremal_points_hoeffding import preprocessing_mu, compute_mus_two_blocks


n = 30
mu1 = 0.45
nb = 30
mu2s = np.linspace(0.15, 0.8, nb) # value of the mean for the second subgroup
t = 0.1
ids = np.arange(1, n) # size of the subgroup
ids_mesh, mu2_mesh = np.meshgrid(ids, mu2s)

compute = False
if compute:
    bounds_variational = np.zeros((len(ids), nb))
    bounds_exponential = np.zeros((len(ids), nb))

    for j in range(nb):
        print(j)
        for i in range(len(ids)):
            mus = compute_mus_two_blocks(n, ids[i], mu1, mu2s[j])
            mus, idx = preprocessing_mu(mus)
            assert t <= 1 - np.mean(mus)
            points = compute_extremal_points_two_blocks(mus, t, n, idx)
            bounds_variational[i][j] = compute_variational_hoeffding_two_blocks(points, mus, idx)
            # Compute the Hoeffding bound to compare
            bounds_exponential[i][j] = compute_hoeffding_exponential_means(mus, t, n)

    np.savetxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/two_blocks_bounds_variational_n_{n}_mu1_{mu1}.txt', bounds_variational, delimiter=',')
    np.savetxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/two_blocks_bounds_exponential_n_{n}_mu1_{mu1}.txt', bounds_exponential, delimiter=',')
else:
    bounds_variational = np.genfromtxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/two_blocks_bounds_variational_n_{n}_mu1_{mu1}.txt', delimiter=',')
    bounds_exponential = np.genfromtxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/two_blocks_bounds_exponential_n_{n}_mu1_{mu1}.txt', delimiter=',')

h = plt.contourf(ids/n, mu2s/mu1, bounds_variational.T)
plt.ylabel(r'$\mu_2 / \mu_1$', fontsize=16)
plt.xlabel(r'$m / n$', fontsize=16)
plt.rc('text', usetex=True)
plt.tick_params(axis='both', which='major', labelsize=14)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)

plt.savefig(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/output/hoeffding/twoblocks_comparison_mu_{mu1}_t_{t}_n_{n}.pdf', dpi=250)#, bbox_inches='tight')
plt.show()