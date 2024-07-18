# Concentration inequalities

This code allows to compute some concentration inequalities using convex optimization. We propose to compute 
upper bounds in three approaches: the exponential, the variational and the polynomial approach, that are developed in
details in the paper: **Constructive approaches to
concentration inequalities with independent
random variables** (to come) - C. Moucer, A. Taylor, F. Bach. 

Given this framework,
we compute refined bounds to:
- Hoeffding's inequality,
- Bennet's inequality,
- Bernstein's inequality.

This code allows in particular to graw figures from the paper above.

### Hoeffding's inequality

### Bennet's inequality

### Bernstein's inequality


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
Our code is divided into two parts: 

Our code allows to compute : 
- concentration inequalities in the exponential approach, given the first-order moment.
- concentration inequalities in the variational approach, given the first-order moment.
- concentration inequalities in the polynomial approach using sums of squares, given the first and second-order moments.


## Authors
* **Céline MOUCER** 
* **Adrien TAYLOR**
* **Francis BACH** 

## References