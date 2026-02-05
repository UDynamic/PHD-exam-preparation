
## Algorithm Input Dimensions

**Front:** For LDA with $d$ features and two classes, what are the dimensions of the key quantities?
**Back:**

- Data point $\mathbf{x}_n$: $(d \times 1)$ vector
- Class mean $\mathbf{\mu}_1, \mathbf{\mu}_2$: $(d \times 1)$ vectors
- Within-class scatter $\mathbf{S}_1, \mathbf{S}_2$: $(d \times d)$ matrices
- Total within-class scatter $\mathbf{S}_W$: $(d \times d)$ matrix
- Between-class scatter $\mathbf{S}_B$: $(d \times d)$ matrix (rank 1 for 2 classes)
- Projection vector $\mathbf{w}$: $(d \times 1)$ vector

## Step 1: Compute Class Means

**Front:** How do you compute the class means for LDA, and what is a common mistake?
**Back:**
For each class $C_k$ with $N_k$ samples:

$$
\mathbf{\mu}_k = \frac{1}{N_k} \sum_{n \in C_k} \mathbf{x}_n
$$

**Dimensions:** $\mathbf{\mu}_k$ is $(d \times 1)$, same as input features.
**Common Mistake:** Forgetting that this is a vector average, not scalar. Each feature dimension is averaged separately.
**Special Consideration:** If features have very different scales, consider standardization before LDA.

## Step 2: Within-Class Scatter Matrices

**Front:** How do you compute the within-class scatter matrix $\mathbf{S}_k$ for class $k$, and what does it represent?
**Back:**
For each class $k$:

$$
\mathbf{S}_k = \sum_{n \in C_k} (\mathbf{x}_n - \mathbf{\mu}_k)(\mathbf{x}_n - \mathbf{\mu}_k)^T
$$

**Dimensions:** $\mathbf{S}_k$ is $(d \times d)$, a covariance-like matrix (but not normalized by $N_k$).
**What it represents:** The spread (covariance structure) of class $k$ around its mean. The $(i,j)$ entry measures how features $i$ and $j$ co-vary within class $k$.
**Important:** This is an **outer product** sum, not inner product. $(\mathbf{x}_n - \mathbf{\mu}_k)$ is $(d \times 1)$, so the product gives $(d \times d)$.

## Step 3: Total Within-Class Scatter

**Front:** How do you compute $\mathbf{S}_W$, and what assumption does its use imply?
**Back:**

$$
\mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2
$$

**Dimensions:** $(d \times d)$ matrix.
**Assumption:** Using a single $\mathbf{S}_W$ assumes both classes share roughly the **same covariance structure**. This is LDA's key assumption. If classes have very different shapes/spreads, LDA may perform poorly.
**Special Case:** If $N < d$ (more features than samples), $\mathbf{S}_W$ will be singular (non-invertible). Regularization may be needed.

## Step 4: Between-Class Scatter Matrix

**Front:** For two classes, what is $\mathbf{S}_B$, and why is it rank 1?
**Back:**

$$
\mathbf{S}_B = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T
$$

**Dimensions:** $(d \times d)$ matrix.
**Rank 1 Reason:** It's an outer product of a single $(d \times 1)$ vector with itself. All columns are multiples of $(\mathbf{\mu}_1 - \mathbf{\mu}_2)$.
**Interpretation:** Captures the separation between class means. Direction $(\mathbf{\mu}_1 - \mathbf{\mu}_2)$ has maximum between-class variance; perpendicular directions have zero.

## Step 5: Solving for the Projection Vector

**Front:** What is the final step to compute $\mathbf{w}$, and what must you check before doing so?
**Back:**

1. **Check invertibility:** Ensure $\mathbf{S}_W$ is invertible. If $N < d$ or features are linearly dependent, it won't be.
2. **Compute:** $\mathbf{w} = \mathbf{S}_W^{-1} (\mathbf{\mu}_1 - \mathbf{\mu}_2)$
3. **Normalize:** Typically scale $\mathbf{w}$ to unit length: $\mathbf{w} \leftarrow \frac{\mathbf{w}}{\|\mathbf{w}\|}$
   **Dimensions:** $\mathbf{S}_W^{-1}$ is $(d \times d)$, $(\mathbf{\mu}_1 - \mathbf{\mu}_2)$ is $(d \times 1)$, so $\mathbf{w}$ is $(d \times 1)$.
   **Key Insight:** The magnitude of $\mathbf{w}$ doesn't matter for projection direction. Only the direction matters for maximizing class separation.

## Algorithm Common Pitfalls

**Front:** What are three common pitfalls when implementing the LDA algorithm?
**Back:**

1. **Singular $\mathbf{S}_W$:** Occurs when $N < d$ or features are perfectly correlated. **Fix:** Add regularization: $\mathbf{S}_W + \lambda\mathbf{I}$, or reduce dimensionality first with PCA.
2. **Mis-scaled features:** If features have different units/scales, those with larger variance dominate. **Fix:** Standardize features to zero mean, unit variance first.
3. **Forgetting LDA's assumptions:** LDA assumes classes have similar covariance (elliptical shapes with same orientation). If classes have very different spreads, Quadratic Discriminant Analysis (QDA) might be better.




---




## LDA Algorithm Dimensions for Binary Classification

**Front:** For binary LDA with $n$ total samples and $m$ features, what are the dimensions of the key quantities?
**Back:**

- Data matrix $\mathbf{X}$ (entire dataset): $(n \times m)$
- Single data point $\mathbf{x}_i$: $(1 \times m)^T=(m \times 1)$ vector
- Class means $\mathbf{\mu}_1, \mathbf{\mu}_2$: $(1 \times m)^T = (m \times 1)$ vectors
- Within-class scatter matrices $\mathbf{S}_1, \mathbf{S}_2$: $(m \times m)$ matrices
- Total within-class scatter $\mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2$: $(m \times m)$ matrix
- Between-class scatter $\mathbf{S}_B$: $(m \times m)$ matrix (rank 1)
- Projection vector $\mathbf{w}$: $(m \times 1)$ vector
- Projected value $\mathbf{w}^T\mathbf{x}$: scalar $(1 \times 1)$

## Step 1: Computing Class Means

**Front:** How do you compute class means for binary LDA, and what's a critical size consideration?
**Back:**
For classes $C_1$ and $C_2$ with $n_1$ and $n_2$ samples ($n_1 + n_2 = n$):

$$
\mathbf{\mu}_1 = \frac{1}{n_1}\sum_{i \in C_1} \mathbf{x}_i, \quad \mathbf{\mu}_2 = \frac{1}{n_2}\sum_{i \in C_2} \mathbf{x}_i
$$

**Dimensions:** Both are $(m \times 1)$.
**Critical Check:** Ensure $n_1 > 0$ and $n_2 > 0$ (both classes must have samples).

* average each of the $m$ features separately within each class.

## Step 2: Within-Class Scatter Matrices

**Front:** How do you compute $\mathbf{S}_1$ and $\mathbf{S}_2$, and what's their geometric meaning?
**Back:**
For each class $k \in \{1, 2\}$:

$$
\mathbf{S}_k = \sum_{i \in C_k} (\mathbf{x}_i - \mathbf{\mu}_k)(\mathbf{x}_i - \mathbf{\mu}_k)^T
$$

**Dimensions:** $(m \times m)$ covariance matrices (unnormalized).
**Geometric Meaning:** $\mathbf{S}_k$ captures the spread and correlation structure of class $k$ in the original $m$-dimensional space.
**Key Point:** Each term $(\mathbf{x}_i - \mathbf{\mu}_k)(\mathbf{x}_i - \mathbf{\mu}_k)^T$ is an $(m \times m)$ outer product. These are summed, not averaged.
**Size Requirement:** For $\mathbf{S}_k$ to be full rank, we need $n_k \geq m$.

* if we subtract the class means $\mu_k$ from the data set $X$ the matrix would be the zero average matrix:

  $$
  X-\mu = \bar{X} \\ 
  \text{then the scatter matrix would be} \quad \bar{X} \times \bar{X}^T \\
  \text{then by muliplying the scatter matrix by } \frac{1}{n-1} \quad \text{we get } \mathbf{S} \text{for each class} \\
  \mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2 \\
  W \propto \mathbf{S}_W^{-1} \times (\mu_1 - \mu_2)
  $$

## Step 3: Total Within-Class Scatter

**Front:** How is $\mathbf{S}_W$ computed, and what singularity issue arises when $n < m$?
**Back:**

$$
\mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2
$$

**Dimensions:** $(m \times m)$ symmetric positive semi-definite matrix.
**Singularity Issue:** If $n < m$ (more features than samples) or if $n_k < m$ for either class, $\mathbf{S}_W$ will be rank-deficient and non-invertible.
**Implication:** Cannot compute $\mathbf{S}_W^{-1}$ directly. Must use regularization or dimensionality reduction first.
**Assumption:** Using a single $\mathbf{S}_W$ assumes equal covariance for both classes (homoscedasticity).

## Step 4: Between-Class Scatter Matrix

**Front:** For binary classification, how is $\mathbf{S}_B$ defined and why is its rank always 1?
**Back:**

$$
\mathbf{S}_B = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T
$$

**Dimensions:** $(m \times m)$ matrix.
**Rank 1 Explanation:** It's an outer product of the single $(m \times 1)$ vector $(\mathbf{\mu}_1 - \mathbf{\mu}_2)$ with itself. All $m$ rows/columns are multiples of this mean difference vector.
**Interpretation:** Captures only the separation direction between classes. Perpendicular directions contain zero between-class variance.
**Note:** For binary classification, we only need 1 projection direction, which aligns with $\mathbf{S}_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$.

## Step 5: Computing the Projection Vector

**Front:** What is the final formula for $\mathbf{w}$, and what practical issues must be addressed?
**Back:**

$$
\mathbf{w} = \mathbf{S}_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)
$$

**Dimensions:** $\mathbf{S}_W^{-1}$ is $(m \times m)$, $(\mathbf{\mu}_1 - \mathbf{\mu}_2)$ is $(m \times 1)$, so $\mathbf{w}$ is $(m \times 1)$.
**Practical Issues:**

1. **Invertibility:** If $\mathbf{S}_W$ is singular (when $n < m$), use pseudoinverse or regularization: $\mathbf{w} = (\mathbf{S}_W + \lambda\mathbf{I})^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$.
2. **Normalization:** Scale doesn't matter for projection direction. Typically normalize to unit length: $\mathbf{w} \leftarrow \frac{\mathbf{w}}{\|\mathbf{w}\|}$.
3. **Numerical Stability:** For large $m$, matrix inversion can be unstable. Use Cholesky decomposition or other stable methods.

## Implementation Checks and Pitfalls

**Front:** What are 4 critical checks before implementing binary LDA?
**Back:**

1. **Sample Size Check:** Verify $n_1 \geq m$ and $n_2 \geq m$ for reliable covariance estimation. If not, use regularization.
2. **Invertibility Check:** Ensure $\mathbf{S}_W$ is invertible. Compute its rank or condition number.
3. **Feature Scaling:** If features have different units/scales, standardize to mean=0, variance=1 first.
4. **Assumption Check:** LDA assumes equal covariance matrices. Check if $\mathbf{S}_1$ and $\mathbf{S}_2$ are roughly proportional. If vastly different, consider QDA.
5. **Class Separation:** If $(\mathbf{\mu}_1 - \mathbf{\mu}_2) \approx \mathbf{0}$, LDA will fail regardless.

## Projection and Classification

**Front:** Once you have $\mathbf{w}$, how do you project data and make classifications?
**Back:**
**Projection:** For any data point $\mathbf{x}$ (size $m \times 1$), compute:

$$
y = \mathbf{w}^T\mathbf{x}
$$

This scalar $y$ is the 1D coordinate on the LDA projection line.

**Classification Threshold:** Compute projected class means:

$$
\tilde{\mu}_1 = \mathbf{w}^T\mathbf{\mu}_1, \quad \tilde{\mu}_2 = \mathbf{w}^T\mathbf{\mu}_2
$$

Use the midpoint (or weighted midpoint) as threshold:

$$
y_{\text{threshold}} = \frac{\tilde{\mu}_1 + \tilde{\mu}_2}{2}
$$

If $y > y_{\text{threshold}}$, classify as class 1; otherwise class 2.

**Note:** The projection reduces dimensionality from $m$ to 1 while preserving maximal class separability.





---








## Algorithm Step 0: Input Data Format

**Front:** What is the input data format for binary LDA with $n$ samples and $m$ features?
**Back:**
We have two classes ($C_1$ and $C_2$) with:

- $\mathbf{X}_1$: $(n_1 \times m)$ matrix for class 1 ($n_1$ samples)
- $\mathbf{X}_2$: $(n_2 \times m)$ matrix for class 2 ($n_2$ samples)
- Total samples: $n = n_1 + n_2$
  Each row is a sample, each column is a feature.

## Algorithm Step 1: Center Each Class

**Front:** How do you center the data for each class?
**Back:**
For each class $k \in \{1, 2\}$:

1. Compute class mean: $\mathbf{\mu}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} \mathbf{x}_i^{(k)}$ (size: $m \times 1$)
2. Center the data: $\mathbf{\bar{X}}_k = \mathbf{X}_k - \mathbf{1}_{n_k} \mathbf{\mu}_k^T$
   where $\mathbf{1}_{n_k}$ is a column vector of ones (size: $n_k \times 1$)
   **Result:** $\mathbf{\bar{X}}_k$ is $(n_k \times m)$ with zero mean for class $k$.

## Algorithm Step 2: Compute Scatter Matrices

**Front:** How do you compute the scatter matrix for each class?
**Back:**
For each centered class matrix $\mathbf{\bar{X}}_k$:

$$
\mathbf{S}_k = \mathbf{\bar{X}}_k^T \mathbf{\bar{X}}_k
$$

**Dimensions:** $\mathbf{\bar{X}}_k^T$ is $(m \times n_k)$, $\mathbf{\bar{X}}_k$ is $(n_k \times m)$, so $\mathbf{S}_k$ is $(m \times m)$.
**Note:** This is the **un-normalized** scatter (sum of outer products). The covariance matrix would be $\frac{1}{n_k-1}\mathbf{S}_k$.

## Algorithm Step 3: Compute Within-Class Scatter

**Front:** How do you combine class scatter matrices into the total within-class scatter?
**Back:**
Sum the scatter matrices from both classes:

$$
\mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2
$$

**Dimensions:** $(m \times m)$ matrix.
**Critical Check:** $\mathbf{S}_W$ must be invertible. Requires $n_1 + n_2 - 2 \geq m$ (more samples than features).
**Note:** $\mathbf{S}_W$ is proportional to the pooled covariance: $\frac{1}{n-2}\mathbf{S}_W$ gives the pooled sample covariance.

## Algorithm Step 4: Compute Mean Difference

**Front:** How do you compute the key direction for class separation?
**Back:**
Using the class means from Step 1:

$$
\mathbf{d} = \mathbf{\mu}_1 - \mathbf{\mu}_2
$$

**Dimensions:** $(m \times 1)$ vector.
**Geometric Meaning:** This is the direction connecting the two class centroids in the original $m$-dimensional space.
**Warning:** Using just $\mathbf{d}$ as the projection direction ignores class covariances (that's what $\mathbf{S}_W^{-1}$ corrects).

## Algorithm Step 5: Compute Optimal Projection

**Front:** What is the final step to compute the LDA projection vector $\mathbf{w}$?
**Back:**
Solve:

$$
\mathbf{w} \propto \mathbf{S}_W^{-1} \mathbf{d}
$$

where $\mathbf{d} = \mathbf{\mu}_1 - \mathbf{\mu}_2$.
**Dimensions:** $\mathbf{S}_W^{-1}$ is $(m \times m)$, $\mathbf{d}$ is $(m \times 1)$, so $\mathbf{w}$ is $(m \times 1)$.
**Practical Computation:**

1. Solve $\mathbf{S}_W \mathbf{w} = \mathbf{d}$ for $\mathbf{w}$ (more stable than explicit inverse)
2. Normalize: $\mathbf{w} \leftarrow \frac{\mathbf{w}}{\|\mathbf{w}\|}$

## Algorithm Complete Formula Chain

**Front:** What is the complete mathematical chain for binary LDA?
**Back:**

1. $\mathbf{\mu}_k = \frac{1}{n_k} \mathbf{X}_k^T \mathbf{1}_{n_k}$
2. $\mathbf{\bar{X}}_k = \mathbf{X}_k - \mathbf{1}_{n_k} \mathbf{\mu}_k^T$
3. $\mathbf{S}_k = \mathbf{\bar{X}}_k^T \mathbf{\bar{X}}_k$
4. $\mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2$
5. $\mathbf{d} = \mathbf{\mu}_1 - \mathbf{\mu}_2$
6. $\mathbf{w} \propto \mathbf{S}_W^{-1} \mathbf{d}$

**Alternative notation:** $\mathbf{w} \propto (\mathbf{S}_1 + \mathbf{S}_2)^{-1} (\mathbf{\mu}_1 - \mathbf{\mu}_2)$

## Dimensionality Verification

**Front:** Verify all dimensions in the LDA computation chain.
**Back:**

- $\mathbf{X}_k$: $(n_k \times m)$
- $\mathbf{\mu}_k$: $(m \times 1)$
- $\mathbf{\bar{X}}_k$: $(n_k \times m)$
- $\mathbf{S}_k$: $(m \times m)$
- $\mathbf{S}_W$: $(m \times m)$
- $\mathbf{d}$: $(m \times 1)$
- $\mathbf{w}$: $(m \times 1)$

Every matrix multiplication is dimensionally consistent.

## Common Implementation Error

**Front:** What's wrong with this implementation: $\mathbf{S}_k = \mathbf{\bar{X}}_k \mathbf{\bar{X}}_k^T$?
**Back:**
This gives $(n_k \times n_k)$ matrix instead of $(m \times m)$!
**Correct:** $\mathbf{S}_k = \mathbf{\bar{X}}_k^T \mathbf{\bar{X}}_k$
**Why:** We want feature-by-feature covariance (features in columns), not sample-by-sample similarity.
**Memory Issue:** For $n_k \gg m$, $\mathbf{\bar{X}}_k \mathbf{\bar{X}}_k^T$ is huge ($n_k \times n_k$) while $\mathbf{\bar{X}}_k^T \mathbf{\bar{X}}_k$ is small ($m \times m$).

## Regularization for Small Samples

**Front:** What to do when $n < m$ (more features than samples)?
**Back:**
$\mathbf{S}_W$ will be singular (non-invertible). Solutions:

1. **Regularized LDA:** $\mathbf{S}_W^{(\text{reg})} = \mathbf{S}_W + \lambda \mathbf{I}_m$
2. **PCA + LDA:** First reduce to $d < n$ dimensions with PCA, then apply LDA
3. **Pseudoinverse:** Use $\mathbf{S}_W^+$ (Moore-Penrose inverse)

**Rule of thumb:** Need $n_1 + n_2 > m + 2$ for stable $\mathbf{S}_W^{-1}$ computation.





---





## Variance Formula (Vector Form)

**Front:** What is the formula for sample variance when data is in vector form?
**Back:**
Given $n$ samples of a single feature in vector $\mathbf{x} = [x_1, x_2, ..., x_n]^T$ with mean $\mu = \frac{1}{n}\sum_{i=1}^n x_i$:

$$
\text{Var}(\mathbf{x}) = \frac{1}{n-1} (\mathbf{x} - \mu\mathbf{1})^T (\mathbf{x} - \mu\mathbf{1}) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \mu)^2
$$

**Dimensions:** $\mathbf{x}$ is $(n \times 1)$, $\mathbf{1}$ is $(n \times 1)$, $(\mathbf{x} - \mu\mathbf{1})$ is $(n \times 1)$, so result is scalar $(1 \times 1)$.

## Covariance Formula (Two Variables)

**Front:** What is the formula for sample covariance between two variables?
**Back:**
Given two vectors $\mathbf{x} = [x_1, ..., x_n]^T$ and $\mathbf{y} = [y_1, ..., y_n]^T$ with means $\mu_x$ and $\mu_y$:

$$
\text{Cov}(\mathbf{x}, \mathbf{y}) = \frac{1}{n-1} (\mathbf{x} - \mu_x\mathbf{1})^T (\mathbf{y} - \mu_y\mathbf{1}) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \mu_x)(y_i - \mu_y)
$$

**Dimensions:** Both centered vectors are $(n \times 1)$, their dot product gives scalar $(1 \times 1)$.

## Covariance Matrix (Multiple Features)

**Front:** How do you compute the sample covariance matrix for $m$ features?
**Back:**
Given data matrix $\mathbf{X}$ of size $(n \times m)$ where columns are features, with column means $\mathbf{\mu} = [\mu_1, \mu_2, ..., \mu_m]$:

1. Center the data: $\mathbf{\bar{X}} = \mathbf{X} - \mathbf{1}_n \mathbf{\mu}^T$
2. Compute covariance: $\mathbf{S} = \frac{1}{n-1} \mathbf{\bar{X}}^T \mathbf{\bar{X}}$

**Dimensions Critical:**

- $\mathbf{X}$: $(n \times m)$ (n samples, m features)
- $\mathbf{1}_n$: $(n \times 1)$ column vector of ones
- $\mathbf{\mu}$: $(1 \times m)$ row vector of means
- $\mathbf{\bar{X}}$: $(n \times m)$ centered data
- $\mathbf{\bar{X}}^T$: $(m \times n)$ transpose
- $\mathbf{S} = \mathbf{\bar{X}}^T \mathbf{\bar{X}}$: $(m \times m)$ covariance matrix

**Key Insight:** $\mathbf{\bar{X}}^T \mathbf{\bar{X}}$ gives feature-by-feature relationships, not sample-by-sample.

## Dimensionality Example: 5×3 Data

**Front:** For a 5×3 data matrix (5 samples, 3 features), what are the dimensions at each step?
**Back:**

1. $\mathbf{X}$: $(5 \times 3)$
2. $\mathbf{\mu}$ (row means): $(1 \times 3)$ or as column vector: $(3 \times 1)$
3. $\mathbf{1}_5$: $(5 \times 1)$ column of ones
4. $\mathbf{1}_5 \mathbf{\mu}^T$: $(5 \times 1) \times (1 \times 3) = (5 \times 3)$
5. $\mathbf{\bar{X}} = \mathbf{X} - \mathbf{1}_5 \mathbf{\mu}^T$: $(5 \times 3) - (5 \times 3) = (5 \times 3)$
6. $\mathbf{\bar{X}}^T$: $(3 \times 5)$
7. $\mathbf{\bar{X}}^T \mathbf{\bar{X}}$: $(3 \times 5) \times (5 \times 3) = (3 \times 3)$ covariance matrix
8. $\frac{1}{n-1}\mathbf{\bar{X}}^T \mathbf{\bar{X}}$: $(3 \times 3)$ sample covariance matrix

**Wrong Order Warning:** $\mathbf{\bar{X}} \mathbf{\bar{X}}^T$ gives $(5 \times 5)$ sample similarity matrix, NOT covariance!

## Algorithm Step 0: Input Data Format

**Front:** What is the input data format for binary LDA with $n$ samples and $m$ features?
**Back:**
We have two classes ($C_1$ and $C_2$) with:

- $\mathbf{X}_1$: $(n_1 \times m)$ matrix for class 1
- $\mathbf{X}_2$: $(n_2 \times m)$ matrix for class 2
- Each row is a sample, each column is a feature
- Column means are computed separately for each class

## Algorithm Step 1: Center Each Class

**Front:** How do you center the data for each class, preserving dimensions?
**Back:**
For each class $k \in \{1, 2\}$:

1. Compute row vector of means (means of features for all the datas):
   $\mathbf{\mu}_k = \frac{1}{n_k} \mathbf{1}_{n_k}^T \mathbf{X}_k$ (size: $(1 \times m)$)
2. Center the data(each data's feature values subtracted by relative mean for that feature):
   $\mathbf{\bar{X}}_k = \mathbf{X}_k - \mathbf{1}_{n_k} \mathbf{\mu}_k$
   where $\mathbf{1}_{n_k}$ is $(n_k \times 1)$ column vector
   **Dimensions Preserved:** $\mathbf{\bar{X}}_k$ remains $(n_k \times m)$

## Algorithm Step 2: Compute Scatter Matrices

**Front:** How do you compute scatter matrices with correct dimensionality?
**Back:**
For each centered class matrix $\mathbf{\bar{X}}_k$ (n data * m features):

$$
\mathbf{S}_k = \mathbf{\bar{X}}_k^T \mathbf{\bar{X}}_k
$$

**Dimensions:** $\mathbf{\bar{X}}_k^T$ is $(m \times n_k)$, $\mathbf{\bar{X}}_k$ is $(n_k \times m)$, so $\mathbf{S}_k$ is $(m \times m)$.
**Note:** This is the **scatter matrix** (unnormalized sum of squares). 

* Covariance would be $\frac{1}{n_k-1}\mathbf{S}_k$.

**Critical:** The order $\mathbf{\bar{X}}_k^T \mathbf{\bar{X}}_k$ yields feature covariance. Reverse order gives sample similarity.

## Algorithm Step 3: Total Within-Class Scatter

**Front:** How do you compute $\mathbf{S}_W$ and what's the dimensional requirement?
**Back:**

$$
\mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2
$$

**Dimensions:** $(m \times m)$ matrix.
**Invertibility Requirement:** For $\mathbf{S}_W^{-1}$ to exist, need $n_1 + n_2 - 2 \geq m$ (more samples than features minus 2).
**If $m > n_1 + n_2 - 2$:** $\mathbf{S}_W$ is singular. Use regularization: $\mathbf{S}_W + \lambda\mathbf{I}_m$.

## Algorithm Step 4: Mean Difference Vector

**Front:** How do you compute the mean difference, watching dimensions?
**Back:**
From Step 1 means $\mathbf{\mu}_1$ and $\mathbf{\mu}_2$ (each $(1 \times m)$ row vectors):

$$
\mathbf{d} = (\mathbf{\mu}_1 - \mathbf{\mu}_2)^T
$$

**Dimensions:** Transpose gives $(m \times 1)$ column vector.
**Alternative:** If means computed as column vectors $(m \times 1)$, then $\mathbf{d} = \mathbf{\mu}_1 - \mathbf{\mu}_2$ directly.
**Consistency is key:** Ensure means and $\mathbf{d}$ are column vectors for $\mathbf{S}_W^{-1}\mathbf{d}$ multiplication.

## Algorithm Step 5: Compute Projection Vector

**Front:** What's the final computation with dimension verification?
**Back:**
Solve:

$$
\mathbf{w} = \mathbf{S}_W^{-1} \mathbf{d}
$$

**Dimension Check:**

- $\mathbf{S}_W^{-1}$: $(m \times m)$
- $\mathbf{d}$: $(m \times 1)$
- $\mathbf{w}$: $(m \times 1)$ ✓
  **Practical:** Solve $\mathbf{S}_W \mathbf{w} = \mathbf{d}$ (avoid explicit inverse).
  **Normalize:** $\mathbf{w} \leftarrow \frac{\mathbf{w}}{\|\mathbf{w}\|}$ (direction matters, magnitude doesn't).

## Complete Dimension Chain

**Front:** Show the complete dimension chain for 100 samples, 10 features, two classes.
**Back:**
Assume $n_1=60$, $n_2=40$, $m=10$:

1. $\mathbf{X}_1$: $(60 \times 10)$, $\mathbf{X}_2$: $(40 \times 10)$
2. $\mathbf{\mu}_1$: $(1 \times 10)$, $\mathbf{\mu}_2$: $(1 \times 10)$
3. $\mathbf{\bar{X}}_1$: $(60 \times 10)$, $\mathbf{\bar{X}}_2$: $(40 \times 10)$
4. $\mathbf{S}_1 = \mathbf{\bar{X}}_1^T \mathbf{\bar{X}}_1$: $(10 \times 10)$
   $\mathbf{S}_2 = \mathbf{\bar{X}}_2^T \mathbf{\bar{X}}_2$: $(10 \times 10)$
5. $\mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2$: $(10 \times 10)$
6. $\mathbf{d} = (\mathbf{\mu}_1 - \mathbf{\mu}_2)^T$: $(10 \times 1)$
7. $\mathbf{w} = \mathbf{S}_W^{-1} \mathbf{d}$: $(10 \times 1)$

## Common Dimension Error

**Front:** What dimension errors occur with $\mathbf{X} - \mathbf{\mu}$ (without broadcasting)?
**Back:**
**Error:** $\mathbf{X}$ is $(n \times m)$, $\mathbf{\mu}$ is $(1 \times m)$ or $(m \times 1)$.
Subtraction fails: $(n \times m) - (1 \times m)$ is not defined in standard matrix algebra.
**Correct:** Use broadcasting: $\mathbf{X} - \mathbf{1}_n \mathbf{\mu}$

- $\mathbf{1}_n$: $(n \times 1)$ column vector
- $\mathbf{\mu}$: $(1 \times m)$ row vector
- $\mathbf{1}_n \mathbf{\mu}$: $(n \times m)$ replicated means
  **Programming:** In NumPy, $\mathbf{X} - \mathbf{\mu}$ works with broadcasting if $\mathbf{\mu}$ is $(1 \times m)$.
