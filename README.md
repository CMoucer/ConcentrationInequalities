# Concentration inequalities

This code allows to compute some concentration inequalities using convex optimization. We propose to compute 
upper bounds in three approaches: a separable and variational approaches based on a family of product-functions,
as well as a polynomial approach, relying on a sums-of-square hierarchy. These techniques are developed in
details in the paper: 

**Constructive approaches to
concentration inequalities with independent
random variables** (to come) - C. Moucer, A. Taylor, F. Bach. 

Given this framework, we compute refined bounds to:
- Hoeffding's inequality,
- Bennett's inequality,
- Bernstein's inequality.

This code allows in particular to draw figures from the paper above.


## Getting Started


### Prerequisites
This codes relies on the following python modules, with versions provided in requirements.txt:

- numpy
```
pip install numpy
```
- cvxpy
```
pip install cvxpy
```
- matplotlib
```
pip install matplotlib
```

We recommend to use the following solver:
- solver MOSEK 
```
pip install mosek
```
- solver SDPA
```
pip install sdpa-python
```

## Compute concentration inequalities
Our code is divided into three main folders:
- **compute_concentration**: containing function returning classical Hoeffding's, Bennett's and Bernstein's inequalities, 
as well as the separable approach applied to the moment generating function, and the variational approach.
- **compute_concentration_sos**: containing functions returning the polynomial approach for Hoeffding's, Bennett's and 
Bernstein's inequalities.
- **figures**: a folder containing the files reproducing figures from the paper.

Our code allows to compute in particular concentration inequalities in the separable and exponential approach given 
finite first-order moments, and in the polynomial approach using semidefinite programming when the first and 
second-order moments are finite.
## Authors
* **Céline MOUCER** 
* **Adrien TAYLOR**
* **Francis BACH** 

## References