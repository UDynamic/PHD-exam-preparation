
## Fisher's Objective Function

**Front:** What is the standard matrix form of Fisher's Linear Discriminant objective function?
**Back:**
The objective is to maximize the ratio of between-class scatter to within-class scatter in the projected space:

* *make the projections on w as **far** as possible and as **compact** as possible*

$$
J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
$$

where $\mathbf{S}_B$ is the between-class scatter matrix and $\mathbf{S}_W$ is the within-class scatter matrix.

## Optimizing the Ratio

**Front:** To find the optimal projection vector $\mathbf{w}$, what mathematical problem arises from maximizing $J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$?
**Back:**
Taking the derivative and setting it to zero leads to the generalized eigenvalue problem:

$$
\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w} \\
(\mathbf{S}_W^{-1} \mathbf{S}_B) \mathbf{w} = \lambda \mathbf{w}
$$

* This means the optimal $\mathbf{w}$ is an eigenvector of $\mathbf{S}_W^{-1}\mathbf{S}_B$.

## Two-Class Special Case

**Front:** In the two-class case, what simpler form does the between-class scatter matrix $\mathbf{S}_B$ take?
**Back:**
For two classes, $\mathbf{S}_B$ simplifies to an outer product:

$$
\mathbf{S}_B = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T
$$

where $\mathbf{\mu}_1$ and $\mathbf{\mu}_2$ are the class mean vectors.

## Key LDA Solution

**Front:** What is the closed-form solution for the optimal LDA projection direction $\mathbf{w}$ in the two-class case?
**Back:**
For two classes, the solution to $\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$ gives:

$$
\mathbf{w} \propto \mathbf{S}_W^{-1} (\mathbf{\mu}_1 - \mathbf{\mu}_2)
$$

The optimal direction is proportional to $\mathbf{S}_W^{-1}$ times the difference between class means.

## Geometric Interpretation of Solution

**Front:** Why isn't the optimal LDA direction simply the line connecting the class means $(\mathbf{\mu}_1 - \mathbf{\mu}_2)$?
**Back:**
The multiplication by $\mathbf{S}_W^{-1}$ accounts for the shape and correlation of the data. If features are correlated or have different variances, $\mathbf{S}_W^{-1}$ "whitens" the space, stretching directions of low variance and compressing directions of high variance to create spherical classes before finding the best separation.
