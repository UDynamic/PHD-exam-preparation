<img src="./assets/regularization.png" alt="Regularization Path"width="600"height="150">

## The Bias-Variance Tradeoff

**Front:** What fundamental ML problem do we address with regularization? `<br/>`

**Back:** We address the bias-variance tradeoff. Models with high variance (complex models) overfit to training data, while models with high bias (simple models) underfit. Regularization helps find the sweet spot between these extremes.

* Regularization's main purpose is to reduce variance by decreasing complexity

## Regularization Intuition

**Front:** In simple terms, what does regularization do to a machine learning model? `<br/>`

**Back:** It "simplifies" the model by penalizing complexity, making it less likely to overfit to noise in the training data and more likely to generalize well to new, unseen data.

## Mathematical Framework

**Front:** What is the general mathematical form of a regularized objective function? `<br/>`

**Back:**

$$
J(\theta) = L(\theta) + \lambda R(\theta)
$$

Where $L(\theta)$ is the original loss function, $R(\theta)$ is the regularization term, and $\lambda$ is the regularization parameter controlling the penalty strength.

## The Regularization Parameter λ

**Front:** What happens when λ is set to 0, very small, very large, or extremely large? `<br/>`

**Back:**

- λ = 0: No regularization, pure loss minimization
- Small λ: Light regularization, slight complexity penalty (complex model that might overfit)
- Large λ: Strong regularization, high complexity penalty (simple model that might underfit)
- λ → ∞: Complete regularization, model collapses to simplest form (often all zero weights)

## L1 Regularization (Lasso)

**Front:** What is the mathematical form and primary characteristic of L1 regularization? `<br/>`

**Back:**

Mathematical form: 

$$
R(\theta) = \sum_{i=1}^{n} |\theta_i|
$$

Primary characteristic: **Sparsity** - tends to drive some parameters exactly to zero, performing automatic feature selection.

## L2 Regularization (Ridge)

**Front:** What is the mathematical form and primary characteristic of L2 regularization? `<br/>`

**Back:**

Mathematical form: 

$$
R(\theta) = \sum_{i=1}^{n} \theta_i^2
$$

Primary characteristic: **Weight shrinkage** - shrinks all parameters toward zero but rarely makes them exactly zero, producing more stable models.

* L2 norm put's sum under the $\sqrt{}$ but not in regularization

## Geometric Interpretation

**Front:** How can we visualize the difference between L1 and L2 constraints? `<br/>`

**Back:** L1 regularization creates a diamond-shaped feasible region ($\sum|\theta_i| \leq t$), where solutions often land at corners (creating sparsity). L2 creates a circular/spherical region ($\sum\theta_i^2\leq t$), where solutions land smoothly on the boundary (creating shrinkage).

* $R(\theta) = \sum_{i=1}^{n} |\theta_i| < t \rightarrow\text{forms a diamond} $
* $R(\theta) = \sum_{i=1}^{n} \theta_i^2 < t \rightarrow\text{forms a circle} $
* in geometric form we want to choose best combination, so we choose the point of contact on both regularization and cost function diagrams.

## L0 Norm Definition

**Front:** What is the mathematical definition of the L0 "norm"? `<br/>`

**Back:** 

$$
\|\theta\|_0 = \sum_{j=1}^{p} \mathbb{1}(\theta_j\neq0)
$$

It counts the number of non-zero parameters in the model, where $\mathbb{1}$ is the indicator function (1 if true, 0 if false).

## L0 Regularization Objective

**Front:** What does the L0-regularized loss function look like? `<br/>`

**Back:** 

$$
J(\theta) = L(\theta) + \lambda \|\theta\|_0
$$

Where $L(\theta)$ is the original loss (e.g., MSE), $\lambda$ is the regularization strength, and $\|\theta\|_0$ counts non-zero parameters. Each additional non-zero parameter costs $\lambda$ in the objective.

## Computational Complexity

**Front:** Why is exact L0 regularization computationally challenging? `<br/>`

**Back:** L0 regularization requires solving a combinatorial optimization problem. For p features, there are $2^p$ possible subsets to evaluate. This exponential growth makes exact solutions NP-hard for large p, requiring heuristic approximations instead.

## Geometric Interpretation

**Front:** What geometric shape represents the L0 constraint region? `<br/>`

**Back:** The L0 constraint $\|\theta\|_0\leq k$ forms a union of coordinate subspaces. In 2D: the axes themselves. In 3D: the coordinate planes and axes. It's not a convex set like L1's diamond or L2's circle, making optimization much harder.

## L0 vs L1 Comparison

**Front:** How does L0 differ fundamentally from L1 regularization? `<br/>`

**Back:** L0 directly counts non-zero parameters (true sparsity), while L1 sums absolute values (approximate sparsity). L1 is convex and computationally efficient; L0 is non-convex and NP-hard. L1 often approximates L0 solutions but doesn't guarantee exact zeros.

## Gradient Descent with Regularization

**Front:** How does L2 regularization modify the gradient descent update rule? `<br/>`

**Back:** For parameter θ with learning rate α:`<br/>`

Original: 

$$
\theta := \theta - \alpha\frac{\partial L}{\partial\theta}
$$


With L2: 

$$
\theta := \theta - \alpha\left(\frac{\partial L}{\partial\theta} + \lambda\theta\right) = \theta(1 - \alpha\lambda) - \alpha\frac{\partial L}{\partial\theta}
$$


The additional $-\alpha\lambda\theta$ term causes "weight decay" at each update.

## Elastic Net Regularization

**Front:** What problem does Elastic Net solve, and what is its mathematical form? `<br/>`

**Back:** Elastic Net combines L1 and L2 to overcome limitations of each: L1 can be unstable with correlated features, L2 doesn't perform feature selection.

Mathematical form:

$$
J(\theta) = L(\theta) + \lambda_1 R_1(\theta) + \lambda_2 R_2(\theta)
$$

single regularization parameter: 

$$
R(\theta) = \alpha\sum|\theta_i| + (1-\alpha)\sum\theta_i^2
$$


Where α controls the mix between L1 and L2.

## Geometric Interpretation of Regularization

**Front:** In the constraint view of regularization, how does λ relate to the diamond/circle boundary? `<br/>`

**Back:** λ doesn't multiply the diamond/circle coordinates. Instead, each λ value corresponds to a different **size** of the constraint region. Small λ ↔ large diamond/circle (weak constraint), large λ ↔ small diamond/circle (strong constraint). The optimal point is where a loss contour touches the constraint boundary.

## Finding the Optimal Contact Point

**Front:** How do you find the optimal model parameters in the geometric constraint view? `<br/>`

**Back:** For a given constraint size t (diamond/circle of fixed size), find where a contour of the loss function L(θ) just touches the constraint boundary. This contact point represents the optimal θ that minimizes loss while satisfying R(θ) ≤ t. Different t values give different solutions along the regularization path.

## Early Stopping

**Front:** How does early stopping work as an implicit regularization technique? `<br/>`

**Back:** Training is stopped when validation error starts increasing while training error continues decreasing. This prevents the model from over-optimizing to training noise, effectively limiting model complexity by restricting training time.

## Dropout Regularization

**Front:** How does dropout work in neural networks, and why is it effective? `<br/>`

**Back:** During training, dropout randomly "drops" (sets to zero) a percentage of neurons in each layer for each training example. This prevents co-adaptation of neurons, forcing the network to learn robust, redundant features, acting like training an ensemble of many thinned networks.

## Weight Decay

**Front:** What is weight decay and how does it relate to L2 regularization? `<br/>`

**Back:** Weight decay is another name for L2 regularization. In the weight update equation, it appears as:

$$
\theta := \theta - \alpha(\frac{\partial L}{\partial\theta} + \lambda\theta) = \theta(1 - \alpha\lambda) - \alpha\frac{\partial L}{\partial\theta}
$$

The $(1 - \alpha\lambda)$ factor causes exponential decay of weights toward zero at each update.

## Practical Regularization Strategies

**Front:** What are three practical considerations when applying regularization? `<br/>`

**Back:**

1.**Cross-validation**: Use CV to find optimal λ value

2.**Feature scaling**: Regularization assumes features are on similar scales (standardize features first)

3.**Don't regularize bias**: Typically exclude the bias/intercept term from regularization penalties

## Regularization in Linear Regression

**Front:** update rule for L2 in linear regression with SSE cost function? `<br/>`

**Back:**

$\Delta\theta_0= -\alpha\sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})$

$\Delta\theta_j= -\alpha [\sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} + \lambda\theta_j]$ `<br/>`

or `<br/>`

$\theta_j = \theta_j(1 - \alpha\lambda) - \alpha(\sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)})$

## Feature Selection Comparison

**Front:** How do L1 and L2 differ in their handling of correlated features? `<br/>`

**Back:** L1 tends to select one feature from a correlated group and zero out the others. L2 tends to distribute weight among correlated features, keeping all but shrinking their coefficients. L1 performs feature selection; L2 performs feature "grouping."

## Regularization Strength Path

**Front:** What does a regularization path plot show us? `<br/>`

**Back:** It shows how model coefficients change as λ increases from 0 to large values. For L1, we see coefficients gradually shrinking to zero (creating sparsity). For L2, we see all coefficients smoothly shrinking toward zero but rarely reaching exactly zero.

## Bayesian Interpretation

**Front:** What is the Bayesian interpretation of regularization? `<br/>`

**Back:** Regularization corresponds to placing a prior distribution on the parameters:

- L1 regularization ↔ Laplace (double exponential) prior
- L2 regularization ↔ Gaussian prior

The MAP (maximum a posteriori) estimate with these priors equals the regularized MLE estimate.

## Double Descent Phenomenon

**Front:** What is the double descent phenomenon and how does it relate to regularization? `<br/>`

**Back:** In modern overparameterized models, test error can decrease even as models become more complex (beyond the interpolation point). Explicit regularization or early stopping helps navigate this curve, avoiding the peak at the interpolation threshold where models fit training data perfectly but generalize poorly.
