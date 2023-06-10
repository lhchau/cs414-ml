# Week1 - Exercise - Gradient Descent

## Exercise 1

> Construct Gradient Descent algorithm to find the optimal value of the simple convex function

_Approach:_

- Build function recognition for 1-degree and 2-degree polynomial
  (Input: $5x^2 + 6x + 10$ => Output: $x^2: 5$, $x: 6$, $const: 10$)
- Using chain rule to calculate $\frac{df}{dx}$
- Build Gradient Descent

$$x^{new} = x^{old} - \eta \frac{df}{dx} |_{x = x^{old}}$$

- Assign inverted Hessian to _learning_rate_

_Use normal learning rate_

```
python3 Exercise1/main.py
```

_Use inverted Hessian as learning rate_

```
python3 Exercise1/main.py --useHessian True
```

## Exercise 2

> Construct Gradient Ascent algorithm to find the optimal value of the simple concave function

_Approach:_

- Build function recognition for 1-degree and 2-degree polynomial
  (Input: $-5x^2 + 6x + 10$ => Output: $x^2: -5$, $x: 6$, $const: 10$)
- Using chain rule to calculate $\frac{df}{dx}$
- Build Gradient Ascent

$$x^{new} = x^{old} + \eta \frac{df}{dx}|_{x=x^{old}}$$

- Assign inverted Hessian to _learning_rate_

_Use normal learning rate_

```
python3 Exercise1/main.py
```

_Use inverted Hessian as learning rate_

```
python3 Exercise1/main.py --useHessian True
```

## Exercise 3

> Why dont find directly optimal value through solve derivative function equal 0, but approximate optimal value by Gradient Descent algorithm?

_Answer:_

- Gradient Descent algorithm is used to find the _optimal value_ in _huge solution space_
- If simple function or architecture (rare): _feasible_ to write down the derivative function (know all information about solution space), so we can directly solve derivative function equal 0 to find an optimal value.
- But if complex function or architecture (often): _infeasible_ to write down the derivative function, we can only compute derivative value on some points (vague information about solution space), so we need Gradient Descent to search for an optimal value.

## Extended question

> Which is the best value of learning rate for 2-degree polynomial functions ?

- Consider 2-degree polynomial function, initial value of x denoted as $x_0$
- Use _Taylor Series expansion_ to approximate $f(x)$ at 2-degree around $x_0$

$$f(x) =  f(x_0) + \frac{df}{dx}(x-x_0) + \frac{1}{2} \frac{d^2f}{dxdx}(x-x_0)^2$$

$$\frac{df(x)}{dx} = \frac{df}{dx} + \frac{d^2f}{dxdx}(x-x_0) = 0$$

$$x = x_0 - (\frac{d^2f}{dxdx})^{-1}\frac{df}{dx} |_{x = x_0} \ (1)$$

- We can see that $(1)$ is similar to _Gradient Descent_ and only need to use one GD-update, initial state $x_0$ can reach an optimal point
