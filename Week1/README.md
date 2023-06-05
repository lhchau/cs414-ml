# Week1 - Exercise

## Exercise 1

> Construct Gradient Descent algorithm to find the optimal value of the simple convex function

_Approach:_

- Build function recognition for 1-degree and 2-degree polynomial
  (Input: $5x^2 + 6x + 10$ => Output: $x^2: 5$, $x: 6$, $const: 10$)
- Using chain rule to calculate $\frac{df}{dx}$
- Build Gradient Descent: $x^{new} = x_{old} - \eta \frac{df}{dx}|_{x=x_{old}}$
- Assign inverted Hessian to _learning_rate_

## Exercise 2

> Construct Gradient Descent algorithm to find the optimal value of the simple concave function

_Approach:_

- Build function recognition for 1-degree and 2-degree polynomial
  (Input: $-5x^2 + 6x + 10$ => Output: $x^2: -5$, $x: 6$, $const: 10$)
- Using chain rule to calculate $\frac{df}{dx}$
- Build Gradient Descent: $x^{new} = x_{old} + \eta \frac{df}{dx}|_{x=x_{old}}$
- Assign inverted Hessian to _learning_rate_

## Exercise 3

> Why dont find directly optimal value through solve derivative function equal 0, but approximate optimal value by Gradient Descent algorithm?

_Answer:_

- Gradient Descent algorithm is used to find the _optimal value_ in _huge solution space_
- If simple function or architecture (rare): _feasible_ to write down the derivative function (know all information about solution space), so we can directly solve derivative function equal 0 to find an optimal value.
- But if complex function or architecture (often): _infeasible_ to write down the derivative function, we can only compute derivative value on some points (vague information about solution space), so we need Gradient Descent to search for an optimal value.
