# Week1 - Exercise

## Exercise 1

> Construct Gradient Descent algorithm to find the optimal value of the simple convex function

_Approach:_

- Build function recognition for 1-degree and 2-degree polynomial
  (Input: $5x^2 + 6x + 10$ => Output: $x^2: 5$, $x: 6$, $const: 10$)
- Using chain rule to calculate $\frac{df}{dx}$
- Build Gradient Descent: $x^{new} = x_{old} - \eta \frac{df}{dx}|_{x=x_{old}}$

## Exercise 2

> Construct Gradient Descent algorithm to find the optimal value of the simple concave function

_Approach:_

- Build function recognition for 1-degree and 2-degree polynomial
  (Input: $-5x^2 + 6x + 10$ => Output: $x^2: -5$, $x: 6$, $const: 10$)
- Using chain rule to calculate $\frac{df}{dx}$
- Build Gradient Descent: $x^{new} = x_{old} + \eta \frac{df}{dx}|_{x=x_{old}}$

## Exercise 3

> Why dont find directly optimal value through solve derivate function equal 0, but approximate optimal value by Gradient Descent algorithm?

