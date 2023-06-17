## Logisic Regression

**Problem**

$$
\text{2 classes } C1, \ C2. \text{ Feature vector } X
$$

$$
p(C_1|X) = y(a) = \sigma(w^TX), \ a = w^TX \newline
\text{with } p(C_2|X) = 1 - p(C_1|X)
$$

**Derivative**

$$
\frac{d\sigma}{da} = \sigma(1-\sigma)
$$

**Loss function** (_Cross entropy loss_)

$$
E(w) = -\sum^N_{n=1}(t_n \text{ln}y_n + (1-t_n)\text{ln}(1-y_n)), \newline
\text{ where } y_n = \sigma(a_n), \text{ and } a_n = w^TX_n
$$

_Gradient of Loss function with respect to vector $w$_

$$
\nabla E(w) = \sum^N_{n=1}(y_n - t_n)X_n
$$

## Softmax Regression

**Problem**

$$
\text{N classes } C1, \ C2, \ ... C_n. \text{ Feature vector } X
$$

$$
p(C_k|X) = \frac{e^{a_k}}{\sum^C_{k=1}e^{a_k}}, \ a_k = w_k^TX
$$

**Derivative**

$$
\frac{dy_k}{da_j} = y_k(I_{kj}-y_j)
$$

**Loss function** (_Cross entropy loss_)

$$
E(w) = -\sum^N_{n=1}\sum^K_{k=1}t_{nk} \text{ln}y_{nk}
$$

_Gradient of Loss function with respect to vector $w$_

$$
\nabla E(w) = \sum^N_{n=1}(y_n - t_n)X_n
$$
