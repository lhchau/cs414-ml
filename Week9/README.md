# Self-Attention mechanism

- Inspired by [Stanford note](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)

## Notation

$$
w_t \sim \text{softmax}(f(w_{1:t-1})) \newline
w \in R^{||V||}, E \in R^{n \times ||V||} \newline
x_i = Ew_i \newline
x = Ew \newline
\text{query: } q_i = Qx_i \newline
\text{key: } k_j = Kx_j \newline
\text{value: } v_j = Vx_j \newline
h_i = \sum_j^n \alpha_{ij} v_j \newline
\alpha_{ij} = \frac{\text{exp}(q_i^T k_j)}{\sum_{j'}^n\text{exp}(q_i^T k_{j'})} \newline
x \text{ after adding positional embedding}: \hat{x}_i = P_i + x_i
$$

## Definition

- $x$ is a non-contextual represenation
- $h$ is a contextual representation
- self-attention: taking a query, and softly looking up information in a key-value store by picking the value of the key **most like** the query
- In the self-attention operation, there’s no built-in notion of order => Positional embedding
- 2 approaches of positional embedding:
    - through learned embeddings (popular), adding pos_embed into $x$
    - through changing $\alpha$ directly, adding pos_embed into $\alpha$ 
- Summary of a minimal self-attention architecture
    - the self-attention operation
    - position representations
    - elementwise nonlinearities
    - future masking