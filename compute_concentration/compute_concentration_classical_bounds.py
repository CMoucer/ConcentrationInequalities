import numpy as np

### This files provides theoretical guarantees for Chebyshev's, Bennet's, Bernstein's and Hoeffding's inequality.


### HOEFFDING
def compute_hoeffding_bound(t, n, a=0, b=1):
    """
    This function returns the one-sided Hoeffding inequality, for n i.i.d random variables with mean mu and such that
    a <= X<= b almost surely. In other words, it returns an upper bound to P(X - mu >= t).
    :param mu: mean of X (float)
    :param t: deviation parameter (float)
    :param n: number of i.i.d. random variables (int)
    :param a: lower bound to the interval (float)
    :param b: upper bound to the interval (float)
    :return:
    """
    return np.exp(-2 * n * t**2 / (b - a) ** 2)

### CHEBYSHEV
def chebyshev_inequality(sigma, t):
    """
    Return the Chebyshev inequality of a random variable with mean mu and variance sigma^2.
    :param sigma: variance (float)
    :param t: deviation (float)
    :return:
    """
    return min(1, sigma**4 / t**2)

### BENNET INEQUAILTY
def function_h(t):
    return (1 + t) * np.log(1+t) - t

def bennet_inequality(sigma, a, t, n=1):
    """
    This function computes Bennet's inequality, for (X_1, ..., X_n) be random variable with finite the variance such that
    sigma^2 = sum_i (E[X_i - E[X_i]]^2) and taking their values in ]-infty, a].
    :param sigma: sum of the variances (float)
    :param a: upper bound (float)
    :param t: deviation (float)
    :param n: number of variables (int)
    :return:
    """
    value = -sigma**2 / a**2 * function_h(a * t * n /sigma**2)
    return np.exp(value)

### BERNSTEIN INEQUALITY
def bernstein_inequality(mu_1, mu_2, t, n, c=1):
    """
    This function computes Bernstein's inequality for i.i.d. random varibales taking their values in [-c, c],
    with mean mu_1 and with a finite second-order moment mu_2.
    :param mu_1: first-order moment, mean (float)
    :param mu_2: second-order moment (float)
    :param t: deviation parameter (float)
    :param n: number of variables (int)
    :param c: bounds for random variables (float)
    :return:
    """
    var = mu_2 - mu_1**2
    return np.exp( - n * t**2 /2 / (var + c * t / 3))
