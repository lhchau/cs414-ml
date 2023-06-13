# Week2 - Exercise - Multiple variable linear regression

## Exercise 1

> Derive a solution for general case of multiple variable linear regression in vectorized form

_Problem:_

$$
\text{Given } X \in \R^n, y \in \R
$$

$$
\text{Define model } \hat{y} = \theta^TX, \text{ where } X = [1, X] \in \R^{n+1}
$$

$$
\text{Define loss } L = \frac{1}{2n}(\theta^TX - y)^2
$$

_Approach:_

$$
\text{Jacobian } J = \nabla_{\theta}L = \frac{1}{n}X \times (\theta^TX - y)^T
$$

$$
\text{Gradient descent: } \theta^{new} = \theta^{old} - \eta \nabla_{\theta}L
$$

## Exercise 2

> Implement Gradient descent for general case of multiple variable linear regression in vectorized form

- Check [main_vector.ipynb](./main_vector.ipynb) for full code
