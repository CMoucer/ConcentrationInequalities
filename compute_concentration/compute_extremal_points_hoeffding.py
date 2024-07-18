import numpy as np
import itertools



def compute_extremal_points(mu, t, n):
    """
    This function computes extremal points to {(x_1, ..., x_n), x_i in [0, 1], x_1 + ... + x_n >= n(mu + t)}, that
    is in the case of n i.i.d. random variables.
    :param mu: mean (float)
    :param t: deviation parameter (float)
    :param n: number of variables (n)
    :return:
    """
    assert (mu + t) <= 1, 'The solution is equal to 0'
    k = n + 1 - int(n * (mu + t)) # compute the number of elements in the set of
    points = np.ones((k, n))
    for i in range(k-1):
        points[1+i:, i] = 0
    points[k-1][k-2] = n * (mu + t) - int(n * (mu + t))
    return points


def compute_extremal_points_means(mus, t, n):
    """
    This function computes extremal points of{(x_1, ..., x_n), x_i in [0, 1], x_1 + ... + x_n >= mu_1 + ... + m_n + nt)},
    that is in the case of n i.i.d. random variables.
    :param mus: means (np.ndarray)
    :param t: deviation parameter (float)
    :param n: number of variables (n)
    :return:
    """
    comp = np.sum(mus) + n * t
    k = n + 1 - int(comp) # compute the number of elements in the set of
    points = np.ones((k, n))
    for i in range(k-1):
        points[1+i:, i] = 0
    points[k-1][k-2] = comp - int(comp)
    permuted_points = np.unique(np.array(list(itertools.permutations(points[0]))), axis=0)
    for i in range(1, k):
        perm = np.unique(np.array(list(itertools.permutations(points[i]))), axis=0)
        permuted_points = np.concatenate((permuted_points, perm), axis=0)
    return permuted_points

## FOR TWO BLOCKS SIMPLIFIED VERSION

def compute_mus_two_blocks(n, idx, mu_1, mu_2):
    """
    Given n i.i.d. random variables, this functions computes two blocks: a first one of size idx, and corresponding to
    random variables having mean mu_1, and a second-one corresponding to random variables having a mean mu_2.
    :param n: number of random variables (np.ndarray)
    :param idx: size of the group having a mean equal to mu_1 (int)
    :param mu_1: mean of the first subgroup (float)
    :param mu_2: mean of the second subgroup (float)
    :return:
    """
    mus = np.ones(n) * mu_2
    mus[:idx] = mu_1
    return mus

def preprocessing_mu(mus):
    """
    This function preprocess the vector of the means, having a size n, such that the smallest subgroup is the first.
    :param mus: means
    :return: the rearranged means, the size of the smallest subgroup
    """
    mus = np.sort(mus)
    idx = 1
    if mus[0] == mus[1]:
        while mus[idx-1] == mus[idx]:
            idx = idx+1
    if idx >= len(mus) // 2 + 1:
        mus = mus[::-1]
        idx = len(mus) - idx
    return mus, idx

def compute_extremal_points_two_blocks(mus, t, n, idx):
    """
    This function computes extremal points given a set of n i.i.d. random variables divided into two subgroups,
    the first one beeing of minimal size idx.
    :param mus: means (np.ndarray)
    :param t: deviation parameter (float)
    :param n: number of variables (int)
    :param idx: size of the smallest and first subgroup (int)
    :return:
    """
    mean = np.mean(mus)
    assert (mean + t) <= 1, 'The solution is equal to 0'
    k = n + 1 - int( n * (mean+ t)) # compute the number of elements in the set of
    points = np.ones((k, n))
    for i in range(k-1):
        points[1+i:, i] = 0
    points[k-1][k-2] = n * (mean + t) - int(n * (mean + t))
    # Add their permutations
    permutations = []
    # Add extremal points for two values
    for i in range(k-1):
        for j in range(min(idx, i+1)):
            point = points[1+i].copy()
            point[:j+1], point[-(j+1):] = points[1 + i][- (j+1):].copy(), points[1 + i][:j+1].copy()
            permutations.append(point)
            if i == k-2:
                pt = point.copy()
                pt[j], pt[k-2] = point[k-2].copy(), point[j].copy()
                permutations.append(pt)
    permutations = np.array(permutations)
    full_points = np.concatenate((points, permutations), axis=0)
    return full_points