# Concentration inequalities

This repository contains the code accompanying our paper on concentration inequalities, a fundamental tool in probability 
theory that quantifies the deviation of a random variable from a certain quantity. Our work introduces a systematic 
convex optimization approach to studying and generating concentration inequalities for independent random variables.
We apply this framework to three standard concentration inequalities: Hoeffding's, Bennett's and Bernstein's inequalities.

This code allows in particular to draw figures from the paper:

**Constructive approaches to
concentration inequalities with independent
random variables** (to come) - C. Moucer, A. Taylor, F. Bach. 

## Key contributions

1. **Separable and variational approaches**: Extends classical moment-generating functions with a focus on first-order moment conditions.
2. **Polynomial approaches**: Develops a hierarchy of sum-of-square approximations to extend techniques to higher-moment conditions

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

### Contents
Our code is divided into three main folders:
- **'compute_concentration/'**: code for classical inequalities (Hoeffding, Bennett, Bernstein, Chebyshev) and code for
the separable and variational approaches in the context of Hoeffding's inequalities
- **'compute_concentration_sos/'**: code for the polynomial approach using sum-of-square approximations, in the context of
Hoeffding's, Bennett's and Bernstein's inequalities.
- **'figures/'**: each file reproduces a figure from the paper.

Our code allows to compute in particular concentration inequalities in the separable and exponential approach given 
finite first-order moments, and in the polynomial approach using semidefinite programming when the first and 
second-order moments are finite.

## Authors
* **Céline MOUCER** 
* **Adrien TAYLOR**
* **Francis BACH** 

## References

* [Concentration inequalities: A Nonasymptotic Theory of Independence](https://academic.oup.com/book/26549?login=true) - S. Boucheron, G. Lugosi, Gabor, P. Massart.
* Constructive approaches to concentration inequalities with independent random variables, C.Moucer, A. Taylor, F. Bach.
