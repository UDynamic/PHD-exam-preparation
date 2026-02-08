## Bayes' Theorem Formula

**Front:** Write Bayes' theorem in probability notation. `<br/>`

**Back:** 

$$
p(\theta | x) = \frac{p(x | \theta) \cdot p(\theta)}{p(x)}
$$

* Posterior = Likelihood * prior / evidence
* Ps(Posterior & Prior) like $\theta$, likelihood dislikes $\theta$ : $\rightarrow\begin{cases} \text{prior} & p(\theta) \\ \text{posterior} & p(\theta|x) \\ \text{likelihood} & p(x|\theta)\end{cases} $

## Prior Probability

**Front:** What is $p(\theta)$ called in Bayes' theorem? `<br/>`

**Back:** The **prior probability** - our initial belief about $\theta$ before seeing any data.

* Ps(Posterior & Prior) like $\theta$, likelihood dislikes $\theta$ : $\rightarrow\begin{cases} \text{prior} & p(\theta) \\ \text{posterior} & p(\theta|x) \\ \text{likelihood} & p(x|\theta)\end{cases} $

## Probability Basics

**Front:** What is $p(\theta | x)$ called in Bayes' theorem? `<br/>`

**Back:** The **posterior probability** - our updated belief about parameters $\theta$ after seeing data $x$.

* Ps(Posterior & Prior) like $\theta$, likelihood dislikes $\theta$ : $\rightarrow\begin{cases} \text{prior} & p(\theta) \\ \text{posterior} & p(\theta|x) \\ \text{likelihood} & p(x|\theta)\end{cases} $

## Likelihood

**Front:** What is $p(x | \theta)$ called in Bayes' theorem? `<br/>`

**Back:** The **likelihood** - how probable the observed data $x$ is given parameters $\theta$.

* Ps(Posterior & Prior) like $\theta$, likelihood dislikes $\theta$ : $\rightarrow\begin{cases} \text{prior} & p(\theta) \\ \text{posterior} & p(\theta|x) \\ \text{likelihood} & p(x|\theta)\end{cases} $

## MLE (Maximum Likelihood Estimation)

**Front:** What does MLE find? Write its formula. `<br/>`

**Back:** The $\theta$ that maximizes the likelihood: 

$$
\hat{\theta}_{MLE} = \arg\max_{\theta} p(x | \theta)
$$

It assumes $\theta$ is fixed but unknown.

## MAP (Maximum A Posteriori)

**Front:** What does MAP find? Write its formula. `<br/>`

**Back:** The $\theta$ that maximizes the posterior: 

$$
\hat{\theta}_{MAP} = \arg\max_{\theta} p(\theta | x) = \arg\max_{\theta} p(x | \theta)p(\theta)
$$

It assumes $\theta$ is random with a prior distribution.

## MLE vs MAP Core Difference

**Front:** What's the fundamental difference between MLE and MAP? `<br/>`

**Back:** MLE uses only likelihood: $\arg\max p(x|\theta)$. MAP adds prior: $\arg\max p(x|\theta)p(\theta)$. MAP is "MLE with regularization" from prior belief.

* MAP multiplies $p(\theta)$ (and p(x) is for normalization btw)so: $\begin{cases} p(x|\theta)p(\theta) & Posterior\\ p(\theta|x) & prior\end{cases} $

## Generative Models

**Front:** What question do generative models answer? `<br/>`

**Back:** "What does each class look like?" They model $p(x|y)$ - the probability of features given the class.

* Generative models learn the data($x$), Discriminants focus on classification($y$) : $\rightarrow\begin{cases} \text{Generative} & \text{learns} \quad p(\text{data|class})=p(x|y)=p(x|\theta) \\ \text{Discriminative} & \text{learns} \quad p(\text{class|data})=p(y|x)=p(\theta|x)\end{cases} $

## Discriminative Models

**Front:** What question do discriminative models answer? `<br/>`

**Back:** "How do we tell classes apart?" They model $p(y|x)$ - the probability of class given the features.

* Generative models learn the data($x$), Discriminants focus on classification($y$) : $\rightarrow\begin{cases} \text{Generative} & \text{learns} \quad p(\text{data|class})=p(x|y)=p(x|\theta) \\ \text{Discriminative} & \text{learns} \quad p(\text{class|data})=p(y|x)=p(\theta|x)\end{cases} $

## Generative Model Formula

**Front:** What does a generative model compute for classification? `<br/>`

**Back:**  by modeling $p(x|y)$ and $p(y)$, then using Bayes' theorem.

$p(y|x) = \frac{p(x|y) \cdot p(y)}{p(x)}$

* Generative models learn the data($x$), Discriminants focus on classification($y$) : $\rightarrow\begin{cases} \text{Generative} & \text{learns} \quad p(\text{data|class})=p(x|y)=p(x|\theta) \\ \text{Discriminative} & \text{learns} \quad p(\text{class|data})=p(y|x)=p(\theta|x)\end{cases} $

## Discriminative Model Formula

**Front:** What does a discriminative model directly model? `<br/>`

**Back:** It directly models $p(y|x)$, ignoring $p(x|y)$ and $p(x)$.

* Generative models learn the data($x$), Discriminants focus on classification($y$) : $\rightarrow\begin{cases} \text{Generative} & \text{learns} \quad p(\text{data|class})=p(x|y)=p(x|\theta) \\ \text{Discriminative} & \text{learns} \quad p(\text{class|data})=p(y|x)=p(\theta|x)\end{cases} $

## Generative Model Example: Spam Detection

**Front:** How would a generative model detect spam? `<br/>`

**Back:** 1. Learn $p(\text{words}|\text{spam})$ and $p(\text{words}|\text{ham})$. 2. Learn $p(\text{spam})$ and $p(\text{ham})$. 3. Classify using Bayes: $p(\text{spam}|\text{words}) \propto p(\text{words}|\text{spam})p(\text{spam})$.

## Discriminative Model Example: Spam Detection

**Front:** How would a discriminative model detect spam? `<br/>`

**Back:** Directly learn $p(\text{spam}|\text{words}) = \sigma(\theta^T \cdot\text{features}(\text{words}))$ where $\sigma$ is sigmoid. Ignore what spam/ham individually look like.

## Generative Models Can Generate Data

**Front:** What can generative models do that discriminative ones cannot? `<br/>`

**Back:** Generate/sample new data points for each class because they learn $p(x|y)$.

## Discriminative Models Focus on Boundaries

**Front:** What do discriminative models focus on? `<br/>`

**Back:** Only the decision boundary between classes, not the full data distribution.

---



## Intro to Classification Approaches

**Front:** What are the two fundamental approaches for building a classifier? `<br/>`

**Back:** 1. **Discriminative:** Find a decision boundary that directly separates the classes. 2. **Generative:** Model the class-conditional distribution $p(x|C_k)$ for each class, then assign a new point to the class with the highest posterior probability $p(C_k|x)$ using Bayes' theorem.

## Generative vs. Discriminative Methods

**Front:** How do generative and discriminative models differ conceptually? Provide one example of each. `<br/>`

**Back:****Generative** models learn the joint distribution $p(x, C_k)$ of inputs and labels (e.g., Naive Bayes, Linear Discriminant Analysis). **Discriminative** models learn the conditional distribution $p(C_k|x)$ or directly map inputs to decision boundaries (e.g., Logistic Regression/Linear Classifier, Perceptron, SVM).

## Linear Classifier Main Condition

**Front:** What is the primary condition required for a *linear* classifier to work perfectly? `<br/>`

**Back:** The data must be **linearly separable/classifiable**. This means a hyperplane exists that can perfectly separate the data points of different classes in the feature space.

## Linear Binary Classification

**Front:** In a 2D feature space, what is the general form of the linear function used for binary classification? `<br/>`

**Back:** $f(x) = w_0 + w_1x_1 + w_2x_2$, where $x=[x_1, x_2]^T$. The decision boundary is the line defined by $f(x)=0$. Points are classified based on the sign of $f(x)$.

## Classification Score as Distance

**Front:** For a data point $x$, what does the quantity $W^Tx$ (or $w^Tx + w_0$) represent geometrically? `<br/>`

**Back:** It is proportional to the signed **distance** from the point $x$ to the decision boundary hyperplane, scaled by the norm of $w$. The sign indicates which side of the boundary the point lies on.

## SSE Cost Function for Classification

**Front:** Why is directly applying the Sum of Squared Errors (SSE) cost function from regression, $J(W) = \sum_i (y^{(i)} - W^Tx^{(i)})^2$, problematic for classification? `<br/>`

**Back:** It assumes target $y^{(i)}$ is a continuous value. In binary classification, targets (e.g., +1/-1) are discrete. Minimizing SSE does not guarantee the sign of $W^Tx^{(i)}$ matches $y^{(i)}$, which is the primary goal.

## Sign Function Inside SSE

**Front:** How can we modify the SSE cost to make it more suitable for classification? What function is introduced? `<br/>`

**Back:** Replace the continuous prediction $W^Tx^{(i)}$ with the discrete class prediction: $\text{sign}(W^Tx^{(i)})$. The cost becomes $J(W) = \sum_i (y^{(i)} - \text{sign}(W^Tx^{(i)}))^2$. However, this is non-differentiable and difficult to optimize directly.

## Perceptron Classifier - Core Idea

**Front:** What is the Perceptron algorithm's fundamental rule for classification? `<br/>`

**Back:** For a data point $(x^{(i)}, y^{(i)})$ where $y^{(i)} \in \{-1, +1\}$, the prediction is $\hat{y} = \text{sign}(W^Tx^{(i)})$. The model is correct if $W^Tx^{(i)}y^{(i)} > 0$.

## Perceptron Cost Function

**Front:** What is the Perceptron cost function? Describe it in words and mathematically. `<br/>`

**Back:** It sums a measure of "error" over all **misclassified** points only. The cost is $J(W) = -\sum_{i \in\mathcal{M}} W^Tx^{(i)}y^{(i)}$, where $\mathcal{M}$ is the set of misclassified points for which $W^Tx^{(i)}y^{(i)} \le0$.

## Perceptron and Distance

**Front:** How does the Perceptron cost function incorporate the distance of a misclassified point to the boundary? `<br/>`

**Back:** For a misclassified point, $W^Tx^{(i)}y^{(i)}$ is negative. Its magnitude $|W^Tx^{(i)}|$ is proportional to the point's distance to the boundary. Therefore, the cost $J(W)$ sums the *signed negative distances*, penalizing misclassified points more severely the farther they are from the boundary.

## Perceptron Learning Rule via Gradient Descent

**Front:** Derive the parameter update rule for the Perceptron using (Stochastic) Gradient Descent on its cost function. `<br/>`

**Back:**`<br/>`

For a single misclassified point $(x^{(i)}, y^{(i)})$, $J^{(i)}(W) = -W^Tx^{(i)}y^{(i)}$. `<br/>`

The gradient is $\nabla_W J^{(i)} = -x^{(i)}y^{(i)}$.`<br/>`

The GD update is: $W := W + \eta x^{(i)}y^{(i)}$, where $\eta$ is the learning rate.

* Correctly classified points do not trigger an update.

## Perceptron Convergence & Optimality

**Front:** What are the conditions for the Perceptron algorithm to converge to a solution with zero training error? `<br/>`

**Back:** Two main conditions: 1. The training data must be **linearly separable**. 2. The learning rate $\eta$ must be sufficiently small. Under these conditions, the Perceptron Convergence Theorem guarantees the algorithm will find a separating hyperplane in a finite number of steps.

## Generalization & Pitfalls of the Perceptron

**Front:** What are key generalization considerations and pitfalls of the Perceptron? `<br/>`

**Back:****Pitfalls:** 1. If data is not linearly separable, the algorithm will never converge (oscillates). 2. It finds *a* separating plane, not necessarily the *best* one (max-margin). 3. The zero-one loss (cost) is non-differentiable. 4. The solution depends on the initialization and order of data presentation (SGD). **Consideration:** Simplicity makes it prone to high variance if data is noisy but separable.

---

## MLE vs MAP vs Generative vs Discriminative: Core Confusion

**Front:** Are MLE and MAP the same as generative and discriminative models? `<br/>`

**Back:****No!** MLE/MAP are **parameter estimation methods**. Generative/discriminative are **model types**. These are independent choices.

## Parameter Estimation Methods

**Front:** What do MLE and MAP determine about a model? `<br/>`

**Back:** How the model's **parameters are learned** from data. They don't define whether a model is generative or discriminative.

## Model Types

**Front:** What do generative and discriminative determine about a model? `<br/>`

**Back:** What **probability distribution** the model learns: $p(x|y)$ (generative) or $p(y|x)$ (discriminative).

## The Independence Clarification

**Front:** Can a generative model use MLE? Can a discriminative model use MAP? `<br/>`

**Back:****Yes to both!** Generative models can use MLE or MAP. Discriminative models can use MLE or MAP. These are separate choices.

## Example 1: Naive Bayes

**Front:** What estimation methods can Naive Bayes (generative) use? `<br/>`

**Back:****MLE**: $\arg\max p(\text{features}|\text{class})$ `<br/>`

**or MAP**: $\arg\max p(\text{features}|\text{class})p(\text{class})$

## Example 2: Logistic Regression

**Front:** What estimation methods can logistic regression (discriminative) use? `<br/>`

**Back:****MLE**: $\arg\max p(\text{class}|\text{features},\theta)$ `<br/>`

**or MAP**: $\arg\max p(\text{class}|\text{features},\theta)p(\theta)$ (equivalent to adding regularization)

## Quick Diagnostic Question

**Front:** How to check if something is MLE/MAP vs generative/discriminative? `<br/>`

**Back:** Ask: 1. "Does it model $p(x|y)$ or $p(y|x)$?" → generative/discriminative. 2. "Does it include a prior on parameters?" → MAP (yes) vs MLE (no).

## Common Pitfall: Assuming MAP = Generative

**Front:** What's wrong with saying "MAP is generative"? `<br/>`

**Back:** MAP is just parameter estimation with a prior. A **discriminative** model (like logistic regression) can use MAP estimation when regularized.

## Special Case: Regularization Connection

**Front:** How does regularization relate to these concepts? `<br/>`

**Back:** Adding L2 regularization to discriminative models = MAP estimation with Gaussian prior. So MAP isn't exclusive to generative models.

---

## Naive Bayes Classification

**Front:** Is Naive Bayes generative or discriminative? Why? `<br/>`

**Back:****Generative**. It models $p(x|y)$ with the "naive" assumption that features are independent given the class: $p(x|y) = \prod_i p(x_i|y)$.

## Logistic Regression Classification

**Front:** Is logistic regression generative or discriminative? Why? `<br/>`

**Back:****Discriminative**. It directly models $p(y|x) = \sigma(\theta^Tx)$ where $\sigma$ is the sigmoid function.

## SVM Classification

**Front:** Is SVM generative or discriminative? Why? `<br/>`

**Back:****Discriminative**. It finds the optimal separating hyperplane without modeling class distributions.

## LDA (Linear Discriminant Analysis)

**Front:** Is LDA generative or discriminative? Why? `<br/>`

**Back:****Generative**. It models each class as a Gaussian distribution with different means but same covariance matrix.

## QDA (Quadratic Discriminant Analysis)

**Front:** How is QDA different from LDA? `<br/>`

**Back:** QDA is also **generative** but allows each class to have its own covariance matrix, giving quadratic decision boundaries.

## Decision Trees Classification

**Front:** Are decision trees generative or discriminative? `<br/>`

**Back:****Discriminative**. They build rules to separate classes without modeling $p(x|y)$.

## Neural Networks Classification

**Front:** Are standard neural networks for classification generative or discriminative? `<br/>`

**Back:****Discriminative**. They learn complex $p(y|x)$ through multiple layers of transformations.

## k-Nearest Neighbors

**Front:** Is k-NN generative or discriminative? `<br/>`

**Back:** Neither strictly, but **discriminative in spirit**. It's non-parametric and classifies based on nearby examples without modeling distributions.

## Bayesian Networks

**Front:** Are Bayesian networks generative or discriminative? `<br/>`

**Back:****Generative**. They model the joint probability distribution $p(x,y)$ using a directed graph structure.

## Hidden Markov Models

**Front:** Are HMMs generative or discriminative? `<br/>`

**Back:****Generative**. They model sequences of observations as being generated by hidden states.

## Linear Regression

**Front:** Is linear regression generative or discriminative? `<br/>`

**Back:****Neither** in traditional sense. It's regression, not classification. But it models $p(y|x)$ with continuous $y$, making it discriminative-like.

## PCA (Principal Component Analysis)

**Front:** Is PCA generative or discriminative? `<br/>`

**Back:****Neither**. It's unsupervised dimensionality reduction with no concept of labels or classification boundaries.

## Clustering Methods

**Front:** Are clustering methods (k-means, hierarchical) generative or discriminative? `<br/>`

**Back:****Neither**. They're unsupervised. If clusters are treated as "classes," the approach is generative-like as they model $p(x|\text{cluster})$.

## Parzen Window / Kernel Density Estimation

**Front:** Can Parzen window methods be generative or discriminative? `<br/>`

**Back:****Generative** when used for density estimation. The Parzen window classifier estimates $p(x|y)$ for each class.

## Regularization Connection to MAP

**Front:** How is L2 regularization in logistic regression related to MAP? `<br/>`

**Back:** Adding L2 regularization $\lambda\|\theta\|^2$ is equivalent to MAP estimation with a Gaussian prior on $\theta$.

## When to Use Generative Models

**Front:** When should you prefer generative models? `<br/>`

**Back:** 1. When you need to generate data. 2. With missing data. 3. Small datasets (priors help). 4. When $p(x|y)$ is naturally simple.

## When to Use Discriminative Models

**Front:** When should you prefer discriminative models? `<br/>`

**Back:** 1. Pure classification task. 2. Large datasets. 3. When decision boundary is simpler than full distributions.

## Common Pitfall: MAP vs MLE Assumptions

**Front:** What's a common mistake when choosing between MAP and MLE? `<br/>`

**Back:** Using MAP without justifying the prior, or using MLE when you actually have useful prior knowledge about parameters.

## Common Pitfall: Naive Bayes Independence

**Front:** What's the main limitation of Naive Bayes? `<br/>`

**Back:** The "naive" assumption that features are independent given the class is often false in real data, though it still works surprisingly well.

## Special Consideration: Neural Networks Can Be Both

**Front:** Can neural networks be both generative and discriminative? `<br/>`

**Back:** Yes! Standard NN classifiers are discriminative. But GANs and VAEs are generative networks that learn data distributions.

## Special Consideration: Bayesian Methods Framework

**Front:** How do Bayesian methods relate to generative/discriminative categories? `<br/>`

**Back:** Bayesian methods provide a framework that's naturally generative, as they model full distributions with priors. But Bayesian logistic regression is discriminative with Bayesian inference on parameters.

## Practical Tip: Testing Model Type

**Front:** Quick test: is a model generative or discriminative? `<br/>`

**Back:** Ask: "Can it generate/sample new data points for each class?" Yes → Generative. No → Discriminative.
