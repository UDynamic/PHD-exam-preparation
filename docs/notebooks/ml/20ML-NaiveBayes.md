
## 01. Bayes Theorem Foundation

**Front:** What is Bayes' Theorem and how is it formulated mathematically?
**Back:**
Bayes' Theorem describes the probability of an event based on prior knowledge of conditions related to the event. For class $y$ and features $x$:

$$
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
$$

Where:

- $P(y|x)$ = Posterior probability
- $P(x|y)$ = Likelihood
- $P(y)$ = Prior probability
- $P(x)$ = Evidence/Marginal likelihood

## 02. Naive Bayes Core Assumption

**Front:** What is the fundamental assumption of Naive Bayes classifiers?
**Back:**
The "naive" conditional independence assumption: features are independent given the class label. This means:

$$
P(x_1, x_2, ..., x_n|y) = \prod_{i=1}^{n} P(x_i|y)
$$

This simplifies computation but rarely holds in real data.

## 03. Naive Bayes Decision Rule

**Front:** How does Naive Bayes make predictions?
**Back:**
Naive Bayes selects the class with maximum posterior probability:

$$
\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i|y)
$$

The denominator $P(x)$ is constant for all classes and can be omitted.

## 04. Prior Probability Calculation

**Front:** How is the prior probability $P(y)$ calculated in Naive Bayes?
**Back:**
Maximum likelihood estimate:

$$
P(y) = \frac{N_y}{N}
$$

Where:

- $N_y$ = Number of training examples with class $y$
- $N$ = Total number of training examples

## 05. Likelihood for Categorical Features

**Front:** How is likelihood $P(x_i|y)$ calculated for categorical features?
**Back:**
Maximum likelihood estimate:

$$
P(x_i = v|y) = \frac{N_{y,x_i=v}}{N_y}
$$

Where:

- $N_{y,x_i=v}$ = Count of examples with class $y$ and feature $i$ taking value $v$
- $N_y$ = Count of examples with class $y$

## 06. Likelihood for Gaussian Naive Bayes

**Front:** How is likelihood modeled for continuous features in Gaussian Naive Bayes?
**Back:**
Features are assumed to follow a normal distribution:

$$
P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right)
$$

Where $\mu_{y,i}$ and $\sigma_{y,i}^2$ are the mean and variance of feature $i$ for class $y$.

## 07. Zero Frequency Problem

**Front:** What is the zero frequency problem in Naive Bayes?
**Back:**
If a feature-category combination never appears in training, $P(x_i|y) = 0$, causing the entire product to become zero regardless of other evidence.

For example: In text classification, if word "miracle" never appeared in "Spam" training emails but appears in test, $P(\text{"miracle"}|\text{Spam}) = 0$ kills all other evidence.

## 08. Laplace (Additive) Smoothing

**Front:** What is Laplace smoothing and how does it solve the zero frequency problem?
**Back:**
Laplace smoothing adds a small constant $\alpha$ to all counts:

$$
P(x_i = v|y) = \frac{N_{y,x_i=v} + \alpha}{N_y + \alpha \cdot k}
$$

Where:

- $\alpha$ = Smoothing parameter (typically 1 for Laplace)
- $k$ = Number of possible values for feature $i$

With $\alpha=1$, this is called "add-one smoothing".

## 09. M-Estimate Smoothing

**Front:** What is M-estimate smoothing in Naive Bayes? [! reference my notes on M-estimate]
**Back:**
M-estimate provides more flexible smoothing than Laplace by incorporating a prior belief $p$:

$$
P(x_i = v|y) = \frac{N_{y,x_i=v} + m \cdot p}{N_y + m}
$$

Where:

- $m$ = Equivalent sample size (weight of prior)
- $p$ = Prior probability (often $1/k$ for uniform prior)

When $m=0$: Maximum likelihood
When $m=k$ and $p=1/k$: Equivalent to Laplace with $\alpha=1$

## 10. Normalization of Posterior Probabilities

**Front:** How do you calculate and normalize posterior probabilities for binary classification? [! reference my notes on p(y) and p(y') normalization]
**Back:**
For two classes $y$ and $y'$, compute unnormalized posteriors:

$$
\text{score}(y|x) = P(y)\prod_i P(x_i|y)
$$

$$
\text{score}(y'|x) = P(y')\prod_i P(x_i|y')
$$

Normalize to get proper probabilities:

$$
P(y|x) = \frac{\text{score}(y|x)}{\text{score}(y|x) + \text{score}(y'|x)}
$$

$$
P(y'|x) = \frac{\text{score}(y'|x)}{\text{score}(y|x) + \text{score}(y'|x)}
$$

Since $P(x)$ is canceled, this ensures $\sum P(y|x) = 1$.

## 11. Log-Space Computation

**Front:** Why are Naive Bayes computations performed in log-space?
**Back:**
Product of many small probabilities causes numerical underflow:

$$
\prod_i P(x_i|y) \approx 0
$$

Solution: Use logarithms to convert products to sums:

$$
\log(P(y|x)) \propto \log(P(y)) + \sum_i \log(P(x_i|y))
$$

This preserves numerical stability and monotonicity of decision.

## 12. Multinomial Naive Bayes

**Front:** What is Multinomial Naive Bayes and when is it used?
**Back:**
Designed for discrete count features (e.g., word counts in documents):

$$
P(x|y) = \frac{(\sum_i x_i)!}{\prod_i x_i!} \prod_i P(w_i|y)^{x_i}
$$

For classification, the multinomial coefficient cancels out:

$$
\hat{y} = \arg\max_y P(y) \prod_i P(w_i|y)^{x_i}
$$

Common in text classification with term frequencies.

## 13. Bernoulli Naive Bayes

**Front:** What is Bernoulli Naive Bayes and how does it differ from Multinomial?
**Back:**
Bernoulli Naive Bayes models binary features (word presence/absence, 0/1):

$$
P(x|y) = \prod_{i=1}^n [P(w_i|y)]^{x_i} [1 - P(w_i|y)]^{1-x_i}
$$

Key difference from Multinomial: Bernoulli explicitly models absence (0 values), while Multinomial only models observed counts and ignores non-occurrence.

## 14. Complement Naive Bayes

**Front:** What is Complement Naive Bayes and why is it useful?
**Back:**
Complement Naive Bayes estimates parameters using data from all classes except the target class:

$$
P(w_i|y) = \frac{N_{y', w_i} + \alpha}{N_{y'} + \alpha \cdot |V|}
$$

Where $y'$ = all classes except $y$. Useful for imbalanced datasets as it prevents estimates from being biased toward majority class.

## 15. Naive Bayes vs Logistic Regression

**Front:** What is the relationship between Naive Bayes and Logistic Regression?
**Back:**
Naive Bayes is a generative model with conditional independence assumption.
Logistic Regression is a discriminative model that directly models $P(y|x)$.

When features are conditionally independent, Naive Bayes converges to Logistic Regression asymptotically. However, Naive Bayes reaches its optimum faster (converges in $O(\log n)$), while Logistic Regression has lower asymptotic error ($O(1/\sqrt{n})$).

## 16. Advantages of Naive Bayes

**Front:** What are the key advantages of Naive Bayes classifiers?
**Back:**

1. Simple and fast to train (one pass over data)
2. Works well with high-dimensional data
3. Handles both continuous and discrete data
4. Robust to irrelevant features
5. Performs well with small datasets
6. Naturally handles missing values (omit during inference)
7. Well-calibrated probabilities with proper smoothing

## 17. Disadvantages and Pitfalls

**Front:** What are the main disadvantages and pitfalls of Naive Bayes? [! include from my notes corrections]
**Back:**

1. **Conditional independence assumption** - rarely true in practice
2. **Zero frequency problem** - requires smoothing (M-estimate, Laplace)
3. **Feature correlation** - correlated features overweight evidence
4. **Poor probability estimates** - often extreme (near 0 or 1)
5. **Sensitive to data representation** - Bernoulli vs Multinomial choice matters
6. **Dominant prior** - strong prior can overwhelm evidence with small samples
7. **Normalization required** - raw scores aren't probabilities; must compute $\frac{P(y)}{P(y)+P(y')}$ for binary cases

## 18. Pitfall: Correlated Features Overweighting

**Front:** Why do correlated features harm Naive Bayes performance?
**Back:**
Naive Bayes multiplies probabilities of independent features. When features are correlated, the model double-counts the same evidence:

If $x_1$ and $x_2$ are highly correlated, $P(x_1,x_2|y) \approx P(x_1|y)$ but Naive Bayes computes $P(x_1|y)P(x_2|y)$, effectively squaring the contribution.

This pushes posterior probabilities toward extremes (0 or 1) and degrades ranking performance.

## 19. Pitfall: M-Estimate Parameter Selection

**Front:** What considerations should be made when selecting m and p in M-estimate?
**Back:**

- $m=0$: No smoothing (vulnerable to zero counts)
- $m$ too large: Prior dominates, model biased
- $m$ too small: Insufficient smoothing
- $p$ should reflect prior knowledge; default $p=1/k$ (uniform)
- Rule of thumb: $m$ between 1 and $|V|$ (vocabulary size)

M-estimate with $p=1/2$ and $m=2$ equals Laplace for binary features.

## 20. Special Consideration: Numeric Stability with Small Probabilities

**Front:** How should you handle extremely small probabilities in Naive Bayes?
**Back:**
For very small $P(x_i|y)$, even log-space computation can be problematic:

1. Use double-precision floating point
2. Apply smoothing even when zero counts aren't present
3. Consider feature selection to remove rare features
4. Add a floor value (e.g., $10^{-10}$) to all probabilities
5. Use M-estimate with appropriate $m$ to prevent probabilities from becoming infinitesimal
