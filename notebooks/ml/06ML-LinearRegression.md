## Linear Regression Hypothesis

**Front:** What is the hypothesis function for simple linear regression? `<br/>`
**Back:** $h_{\theta}(x) = \theta_0 + \theta_1x$, where $\theta_0$ is the intercept and $\theta_1$ is the slope. This gives the prediction $\hat{y}$.

## Cost Function Goal

**Front:** What is the goal when training a linear regression model? `<br/>`
**Back:** To find parameters $\theta_0, \theta_1$ that minimize the cost function $J(\theta_0, \theta_1)$.

## SSE Cost Function**Front:** What is the Sum of Squared Errors (SSE) cost function? `<br/>`

**Back:** $J(\theta) = \frac{1}{2} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$. The $\frac{1}{2}$ simplifies derivatives.

## MSE Cost Function

**Front:** What is Mean Squared Error (MSE) and how does it differ from SSE? `<br/>`
**Back:** $J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$. It averages error by dividing by $m$, making it comparable across datasets.

## RMSE Cost Function

**Front:** What is Root Mean Squared Error (RMSE)? `<br/>`
**Back:** $J(\theta) = \sqrt{\frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2}$. It's in the same units as $y$ for easier interpretation.

## MAE Cost Function

**Front:** What is Mean Absolute Error (MAE) and its key difference from MSE? `<br/>`
**Back:** $J(\theta) = \frac{1}{m} \sum_{i=1}^m |h_\theta(x^{(i)}) - y^{(i)}|$. Unlike MSE, it's less sensitive to outliers but uses sign() in gradient.

## Gradient Descent Algorithm

**Front:** What are the 3 main steps of Gradient Descent? `<br/>`
**Back:** 1. Initialize $\theta$ randomly. 2. Update: $\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$. 3. Repeat until convergence.

## Learning Rate $\alpha$

**Front:** What are the effects of too small or too large learning rate $\alpha$? `<br/>`
**Back:** Too small: very slow convergence. Too large: may overshoot minimum, causing divergence or oscillation.

## Convergence Condition

**Front:** How do we know when Gradient Descent has converged? `<br/>`
**Back:** When parameter changes are small: $|\Delta\theta_j| \leq \epsilon$ for all $j$, where $\epsilon$ is a small tolerance.

## Simultaneous Parameter Update

**Front:** Why must Gradient Descent update all parameters simultaneously? `<br/>`
**Back:** To ensure each gradient $\frac{\partial J}{\partial\theta_j}$ is computed using the same parameter values before any updates.

## MSE Gradient for $\theta_0$

**Front:** What is $\frac{\partial J}{\partial\theta_0}$ for MSE cost? `<br/>`
**Back:** $\frac{\partial J}{\partial\theta_0} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})$.

## MSE Gradient for $\theta_j$

**Front:** What is $\frac{\partial J}{\partial\theta_j}$ for MSE cost? `<br/>`
**Back:** $\frac{\partial J}{\partial\theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$.

## Convexity in Linear Regression

**Front:** Why is MSE with linear regression guaranteed to find global minimum? `<br/>`
**Back:** The cost function is convex (bowl-shaped), so Gradient Descent converges to global optimum, not local minima.

## Epoch vs Iteration

**Front:** What's the difference between an epoch and an iteration? `<br/>`
**Back:** **Iteration:** One parameter update. **Epoch:** One full pass through entire training dataset (contains multiple iterations in mini-batch GD).

## Batch Gradient Descent (BGD)

**Front:** How does Batch Gradient Descent compute gradients? `<br/>`
**Back:** Uses entire training set: $\theta_j := \theta_j - \alpha\cdot\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$.

## BGD Pros and Cons

**Front:** What are advantages and disadvantages of BGD? `<br/>`
**Back:** **Pros:** Accurate, guaranteed convergence for convex functions. **Cons:** Slow for large datasets, high memory usage, can get stuck in local minima for non-convex functions.

## Stochastic Gradient Descent (SGD)

**Front:** How does SGD compute gradients differently from BGD? `<br/>`
**Back:** Uses one random example: $\theta_j := \theta_j - \alpha\cdot (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$.

## SGD Pros and Cons

**Front:** What are advantages and disadvantages of SGD? `<br/>`
**Back:** **Pros:** Fast, low memory, can escape local minima. **Cons:** Noisy updates, may not converge exactly, needs data shuffling.

## Mini-Batch Gradient Descent

**Front:** What is the compromise of Mini-Batch GD? `<br/>`
**Back:** Uses $b$ examples per update: $\theta_j := \theta_j - \alpha\cdot\frac{1}{b} \sum_{i=1}^b (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$. Balances speed and stability.

## Normal Equation Method

**Front:** What is the closed-form solution for linear regression? `<br/>`
**Back:** $\theta = (X^T X)^{-1} X^T y$, where $X$ is design matrix with rows as training examples.

## Normal Equation Pros

**Front:** What are advantages of the Normal Equation over Gradient Descent? `<br/>`
**Back:** No iterations needed, no learning rate to tune, guaranteed exact solution (if $X^TX$ invertible).

## Normal Equation Cons

**Front:** What are limitations of the Normal Equation? `<br/>`
**Back:** Computationally expensive for large features ($O(n^3)$ for inversion), fails if $X^TX$ non-invertible (redundant features or $n > m$).

## When to Use Normal Equation vs GD

**Front:** When should you use Normal Equation versus Gradient Descent? `<br/>`
**Back:** **Normal Equation:** Small number of features ($n < 10,000$). **Gradient Descent:** Large $n$, very large $m$, or non-linear models.

## MAE Gradient Computation

**Front:** Why is MAE's gradient different from MSE? `<br/>`
**Back:** MAE uses $\text{sign}(h_\theta(x^{(i)}) - y^{(i)})$ instead of $(h_\theta(x^{(i)}) - y^{(i)})$, making it non-differentiable at zero but robust to outliers.

## Cost Function Comparison

**Front:** When might you choose MAE over MSE? `<br/>`
**Back:** Choose **MAE** when your data has many outliers (robust). Choose **MSE** for normally distributed errors (efficient, convex).

## Batch Size in Mini-Batch GD

**Front:** What are typical mini-batch sizes and their effects? `<br/>`
**Back:** 32, 64, 128. Larger batches: more stable but slower. Smaller batches: faster updates but noisier.

## Non-invertible $X^TX$ Causes

**Front:** Why might $X^TX$ be non-invertible? `<br/>`
**Back:** 1. Redundant/linearly dependent features. 2. More features than examples ($n > m$). 3. Features with zero variance.

## Practical Gradient Descent Tip

**Front:** What's a good practice when implementing Gradient Descent? `<br/>`
**Back:** Plot $J(\theta)$ versus iterations to diagnose convergence issues and choose appropriate $\alpha$.

## Supervised Learning

**Front:** What is the core paradigm of supervised learning? `<br/>`

**Back:** Learning a mapping from input variables $x$ to an output variable $y$, using a labeled dataset of example input-output pairs $\{ (x^{(i)}, y^{(i)}) \}_{i=1}^m$.

## Hypothesis Function (Linear Model)

**Front:** In linear regression, what is the form of the hypothesis function $h_\theta(x)$ for a single feature? `<br/>`

**Back:** $h_\theta(x) = \theta_0 + \theta_1 x$, where $\theta_0$ is the intercept/bias and $\theta_1$ is the weight/slope. It predicts the output $\hat{y}$.

## Cost Function - Intuition

**Front:** What is the purpose of a cost function $J(\theta)$ in machine learning? `<br/>`

**Back:** It quantifies the error between the model's predictions $h_\theta(x)$ and the true targets $y$. The learning goal is to find parameters $\theta$ that minimize $J(\theta)$.

## Mean Squared Error (MSE) Cost

**Front:** What is the formula for the Mean Squared Error (MSE) cost function for linear regression? `<br/>`

**Back:** $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$. The $\frac{1}{2}$ is often included to cancel the factor of 2 from differentiation.

## Gradient Descent - Core Idea

**Front:** Describe the Gradient Descent algorithm in one sentence. `<br/>`

**Back:** An iterative optimization algorithm that adjusts parameters $\theta$ by taking steps proportional to the negative gradient of the cost function $J(\theta)$.

## Gradient Descent Update Rule

**Front:** Write the general parameter update rule for Gradient Descent. `<br/>`

**Back:** $\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j} J(\theta)$, repeated simultaneously for all $j$. $\alpha$ is the learning rate.

## Learning Rate ($\alpha$)

**Front:** What are the consequences of setting the learning rate $\alpha$ too small or too large? `<br/>`

**Back:** Too small: very slow convergence. Too large: can overshoot the minimum, causing divergence or oscillatory, slow convergence.

## Partial Derivative for MSE (Intercept)

**Front:** What is the partial derivative $\frac{\partial J}{\partial\theta_0}$ for the MSE cost with hypothesis $h_\theta(x)=\theta_0+\theta_1x$? `<br/>`

**Back:** $\frac{\partial J}{\partial\theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$.

## Partial Derivative for MSE (Slope)

**Front:** What is the partial derivative $\frac{\partial J}{\partial\theta_1}$ for the MSE cost with hypothesis $h_\theta(x)=\theta_0+\theta_1x$? `<br/>`

**Back:** $\frac{\partial J}{\partial\theta_1} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$.

## Convexity in Linear Regression

**Front:** Why is using (M)SE for linear regression advantageous? `<br/>`

**Back:** The cost function $J(\theta)$ is convex. Gradient Descent on a convex function is guaranteed to converge to the global minimum, not a local one.

## Batch Gradient Descent (BGD)

**Front:** How does Batch Gradient Descent compute the gradient in each iteration? `<br/>`

**Back:** It uses the **entire training set** to compute the gradient. The update for $\theta_j$ sums over all $m$ examples: $\theta_j := \theta_j - \alpha\cdot\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$.

## Stochastic Gradient Descent (SGD)

**Front:** How does Stochastic Gradient Descent compute the gradient for an update? `<br/>`

**Back:** It uses **one randomly chosen training example** $(x^{(i)}, y^{(i)})$ per update: $\theta_j := \theta_j - \alpha\cdot (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$.

## Mini-batch Gradient Descent

**Front:** What is the compromise offered by Mini-batch Gradient Descent? `<br/>`

**Back:** It uses a small random subset (mini-batch) of size $b$ (e.g., 32) to compute the gradient: $\theta_j := \theta_j - \alpha\cdot\frac{1}{b} \sum_{i=1}^{b} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$.

## Epoch vs. Iteration

**Front:** Distinguish between an *epoch* and an *iteration* in Gradient Descent. `<br/>`

**Back:****Iteration:** One update of the model parameters. **Epoch:** One full pass through the entire training dataset (may contain many iterations, e.g., in mini-batch GD).

## Normal Equation Method

**Front:** What is the closed-form solution for linear regression parameters $\theta$? `<br/>`

**Back:** $\theta = (X^T X)^{-1} X^T y$, where $X$ is the design matrix (with a column of 1s for the intercept) and $y$ is the target vector.

## Cost Functions: SSE vs. MSE

**Front:** What is the practical difference between Sum of Squared Errors (SSE) and Mean Squared Error (MSE)? `<br/>`

**Back:** SSE: $J = \frac{1}{2}\sum (h-y)^2$. MSE: $J = \frac{1}{2m}\sum (h-y)^2$. MSE normalizes by dataset size $m$, making cost comparable across different sized datasets. Their gradients differ by a factor of $1/m$.

## Cost Functions: MAE

**Front:** What is the Mean Absolute Error (MAE) cost function and its key property? `<br/>`

**Back:** $J(\theta) = \frac{1}{m} \sum_{i=1}^m |h_\theta(x^{(i)}) - y^{(i)}|$. It is less sensitive to outliers than MSE, but its gradient involves the sign function, which is not differentiable at zero.

## Gradient Descent Pitfall - Simultaneous Update

**Front:** What is a critical implementation rule for the vanilla Gradient Descent update step? `<br/>`

**Back:** All parameters $\theta_j$ must be updated **simultaneously** using the *old* values of $\theta$. Do not use a newly updated $\theta_0$ to compute the update for $\theta_1$.

## BGD vs. SGD - Trade-offs

**Front:** Compare the convergence behavior of BGD and SGD. `<br/>`

**Back:****BGD:** Converges smoothly to the minimum. **SGD:** Converges with noise; the path is erratic. It can escape shallow local minima but may oscillate near the global minimum.

## Normal Equation vs. GD - When to Use

**Front:** When might you prefer Gradient Descent over the Normal Equation? `<br/>`

**Back:** When $n$ (number of features) is very large (>10,000), as inverting $X^TX$ ($O(n^3)$) is computationally expensive. GD scales better.

## Normal Equation Limitation

**Front:** What are two reasons the Normal Equation $\theta = (X^TX)^{-1}X^Ty$ might fail? `<br/>`

**Back:** 1. **Non-invertibility:** $X^TX$ is singular if features are linearly dependent (redundant). 2. **Computational inefficiency:** If $n$ is very large, the inversion is slow.
