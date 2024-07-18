import numpy as np
import cvxpy as cp

### Univariate setting
def compute_hoeffding_exact_univariate(mu, t, n=1):
    """
    This function returns the exact Hoeffding bound for univariate variable in [0, 1] almost surely.
    :param mu: mean of X
    :param t: deviation parameter
    :param n: number of i.i.d. random variables
    :return:
    """
    assert n == 1, 'This bound does not apply to multivariate random variables.'
    return mu/(mu+t)

###  IID random variables
def compute_hoeffding_exponential(mu, t, n):
    """
    This function returns an upper bound to P(sum(X_i) - n * mu) >= t) in the exponential treatment, given n i.i.d.
    i.i.d. random variables taking their values in [0, 1] and with equal mean mu.
    :param mu: mean of X (float)
    :param t: deviation parameter (float)
    :param n: number of i.i.d. random variables (int)
    :return:
    """
    assert t >= 0 and mu <= 1
    nu = mu + t
    rho_ct = (mu/nu)**(n * nu) * ((1 - mu)/(1 - nu))**(n * (1 - nu))
    return rho_ct

def compute_variational_hoeffding(xs, mu, return_problem=False):
    """
    The function returns the bound in the convex relaxation approach, for n i.i.d. random variables taking their
    values in [0, 1], and with mean mu.
    :param xs: set of extremal points (np.ndarray)
    :param mu: mean of X (float)
    :return:
    """
    m, n = xs.shape # n is the number of variables, and m the number of extremal points
    alpha = cp.Variable(1)
    beta = cp.Variable(1)
    v = cp.Variable(1)
    constraints = [alpha >= 0]
    constraints = constraints + [beta + alpha >= 0]
    for i in range(m):
        constraints = constraints + [sum([-cp.log(alpha + beta * xs[i][j]) for j in range(n)]) <= v]
    objective = cp .Minimize(n * (alpha + beta * mu - 1) + v)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    if return_problem == True:
        return prob, alpha.value, beta.value, v.value, constraints
    else:
        return np.exp(prob.value)

### With different means
def compute_hoeffding_exponential_means(mus, t, n):
    """
    This function returns the (quadratic upper) bound to the exponential treatment for n i.i.d. random variables
    taking their values in [0, 1] a.s., with different means.
    :param mus: mean (np.ndarray)
    :param t: deviation parameter (float)
    :param n: number of variables (int)
    :return:
    """
    sum_mu = np.sum((mus - .5)/np.log(mus/(1 - mus)))
    rho_ct = np.exp(-n**2 * t**2 / 2 / sum_mu)
    return rho_ct

### Divided into two blocks having different means
def compute_variational_hoeffding_two_blocks(xs, mus, idx, return_problem=False):
    """
    This function returns the bound in the variational approach, for n i.i.d. random variables taking their values
    in [0, 1], and such that they are divided into two blocks having different means [1, ..., idx-1] and [idx, ..., n].
    WARNING: before computing this bound, the mean mus must be rearranged with respect to their mean.
    :param xs: extremal points (np.ndarray)
    :param mus: arranged means (np.ndarray)
    :param idx: size of the first subgroup (int)
    :param return_problem: plot alpha, beta, v in addition to the problem.
    :return:
    """
    m, d = xs.shape
    alpha = cp.Variable(2)
    beta = cp.Variable(2)
    v = cp.Variable(1)
    constraints = [alpha >= 0]
    constraints = constraints + [beta + alpha >= 0]
    for i in range(m):
        constraints = constraints + [sum([-cp.log(alpha[0] + beta[0] * xs[i][j]) for j in range(idx)]) +  sum([-cp.log(alpha[1] + beta[1] * xs[i][j]) for j in range(idx, d)])<= v]
    objective = cp .Minimize(idx * (alpha[0] + beta[0] * mus[0] - 1) + (d-idx) * (alpha[1] + beta[1] * mus[idx] - 1) + v)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)

    if return_problem:
        return prob, alpha.value, beta.value, v.value, constraints
    else:
        return np.exp(prob.value)