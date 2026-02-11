subjects: {bayes classifier and risk matrix (definition, formulation and everything)}
more specifically add flashcards on these notes too [ Don't miss on anything] :

1. fundamental formulas of bayes theorem for C_k classes of data
   1. posterior is likelihood times prior on evidence
2. minimum posterir is error
3. if don't have prior p(c_k) then compare lilkelihoods p(x|c_k). [! why]
4. if don'thave the prior the evidence p(x) is not concidered in the posterior formula for p(c_k|x) = p(x|c_k). [! why and how]
5. what is risk matrix and it's use case.
6. for a binary classification : p(x|c_1) = 1 - p(x|c_2) [! generalize on this]
7. for decision boundary : p(c_1|x) = p(c_2|x) \rightarrow p(x|c_1)p(c_1) = p(x|c_2)p(c_2)
8. for binary classification: if p is the probability of x belonging to c_1 [! clarify on which one of the posterior or likelihood is meant here] then 1-p is the probability of x belonging to c_2 and for p>1-p according to bayes x belongs to c_1 and the error is 1-p
   1. for more than 2 classes x belongs to c_1 if p_{c_1} > \sum p_{c_n} for all other classes and the error will be that exact sum over ps
9. total error for the bayes classifier for binary class is min{p(c_1), p(c_2)} \times (1 - min{p(c_1), p(c_2)}) [! what does it mean and clarify on it]

---

## 01. Bayes' Theorem for Classification

**Front:** What is the fundamental formula of Bayes' theorem for a class $C_k$ given data point $\mathbf{x}$?
**Back:**
The posterior probability $P(C_k | \mathbf{x})$ is proportional to the likelihood times the prior.

$$
P(C_k | \mathbf{x}) = \frac{p(\mathbf{x} | C_k) P(C_k)}{p(\mathbf{x})}
$$

where:
$p(\mathbf{x} | C_k)$ is the likelihood.
$P(C_k)$ is the prior.
$p(\mathbf{x})$ is the evidence (or marginal likelihood).

## 01. Marginal Likelihood (Evidence)

**Front:** What is $P(\mathbf{x})$, the evidence, in terms of the class priors and likelihoods?
**Back:**
The marginal likelihood $P(\mathbf{x})$ is obtained by summing (or integrating) the joint distribution over all possible classes. It is the sum of the numerators of Bayes' theorem for each class, not the sum of posteriors.

$$
P(\mathbf{x}) = \sum_{k=1}^{K} P(\mathbf{x} \cap C_k) = \sum_{k=1}^{K} p(\mathbf{x} | C_k) P(C_k)
$$

This ensures the posterior probabilities $P(C_k | \mathbf{x})$ sum to 1.

## 02. The Evidence Term

**Front:** Why is the evidence $p(\mathbf{x})$ often omitted when comparing posteriors for different classes?
**Back:**
Because $p(\mathbf{x}) = \sum_k p(\mathbf{x} | C_k) P(C_k)$ is constant for all classes given $\mathbf{x}$. For comparison/decision making, we only need the numerator: $p(\mathbf{x} | C_k) P(C_k)$. The evidence acts as a normalization constant ensuring posteriors sum to 1.

## 02. Pitfall: Evidence vs. Sum of Posteriors

**Front:** Is $P(\mathbf{x})$ the sum of the posteriors $P(C_k | \mathbf{x})$ for all classes?
**Back:**
No, this is incorrect and a common pitfall. The sum of posteriors is always 1 for any $\mathbf{x}$:

$$
\sum_{k=1}^{K} P(C_k | \mathbf{x}) = 1
$$

The evidence $P(\mathbf{x})$ is a probability (density) for $\mathbf{x}$ itself. It is a normalization constant that *makes* the posteriors sum to 1, not the result of that sum.

## 03. Pitfall: Reversing the Dependency

**Front:** What is the fundamental logical error in stating "$P(\mathbf{x}) = \sum_k P(C_k | \mathbf{x})$"?
**Back:**
It reverses the dependency. In probability theory, $P(\mathbf{x})$ is calculated **first** from the generative model (priors and likelihoods). Then, posteriors $P(C_k | \mathbf{x})$ are derived **from** $P(\mathbf{x})$ via Bayes' theorem. The sum of posteriors is always 1 by definition, but this is a **consequence** of normalization using $P(\mathbf{x})$, not a way to compute $P(\mathbf{x})$.

## 03. Bayes Classifier Decision Rule

**Front:** What is the optimal (Bayes) classification rule to minimize probability of error?
**Back:**
Assign $\mathbf{x}$ to the class $C_k$ with the highest posterior probability.

$$
\text{Assign to } C_k \text{ if } P(C_k | \mathbf{x}) > P(C_j | \mathbf{x}) \text{ for all } j \neq k
$$

This is equivalent to maximizing the numerator $p(\mathbf{x} | C_k) P(C_k)$.

## 35. Likelihood vs. Posterior: What Does $p(\mathbf{x}|C_k)$ Represent?

**Front:** Does the function $p(\mathbf{x}|C_k)$ give the probability of $\mathbf{x}$ belonging to class $C_k$?
**Back:**
**No—critical distinction.** $p(\mathbf{x}|C_k)$ is the **likelihood**, not the posterior probability of class membership.

- **Likelihood $p(\mathbf{x}|C_k)$:** "Given that the class is $C_k$, what is the probability (density) of observing this specific $\mathbf{x}$?"
- **Posterior $P(C_k|\mathbf{x})$:** "Given that we observed this specific $\mathbf{x}$, what is the probability it belongs to class $C_k$?"

**Likelihood is not a probability over classes.** It does not sum to 1 over $k$; it integrates to 1 over $\mathbf{x}$ for each fixed $k$.

## 37. Pitfall: The "Probability of $\mathbf{x}$ Being in $C_k$" Confusion

**Front:** Why is it incorrect to say "$p(\mathbf{x}|C_k)$ gives the probability that $\mathbf{x}$ belongs to class $C_k$"?
**Back:**
This statement **reverses the conditioning**. The correct statement is:

**Correct:** $p(\mathbf{x}|C_k)$ is the probability of observing $\mathbf{x}$ **given that** the class is $C_k$.

**Incorrect:** $p(\mathbf{x}|C_k)$ is the probability that the class is $C_k$ **given that** we observed $\mathbf{x}$.

The latter is $P(C_k|\mathbf{x})$, which requires **Bayes' theorem** to compute from the likelihood, prior, and evidence:

$$
P(C_k|\mathbf{x}) = \frac{p(\mathbf{x}|C_k)P(C_k)}{p(\mathbf{x})}
$$

**Analogy:** If $C_k$ = "rain" and $\mathbf{x}$ = "wet grass", then:

- $p(\text{wet grass}|\text{rain})$ is high (likelihood)
- $P(\text{rain}|\text{wet grass})$ is also high but not the same quantity—it depends on the prior probability of rain and other causes of wet grass

## 38. Connection to Decision Boundaries

**Front:** In binary classification, if we say "$p$ is the probability of $\mathbf{x}$ belonging to $C_1$", which quantity are we actually referring to?
**Back:**
**We are (usually) referring to the posterior $P(C_1|\mathbf{x})$, not the likelihood.**

This is evident from common statements:

- "If $p > 0.5$, classify as $C_1$" → This compares posterior probability to 0.5 threshold
- "$1-p$ is the probability of belonging to $C_2$" → This uses the fact that posteriors sum to 1: $P(C_2|\mathbf{x}) = 1 - P(C_1|\mathbf{x})$

**Important:** This only works for **posteriors**. Likelihoods do **not** satisfy $p(\mathbf{x}|C_2) = 1 - p(\mathbf{x}|C_1)$. Likelihoods are independent probability densities, each integrating to 1 over $\mathbf{x}$.

**Rule of thumb:** If you're making a decision about class membership, you're working with **posteriors** (or their proportional equivalents $p(\mathbf{x}|C_k)P(C_k)$). If you're generating data from a known class, you're working with **likelihoods**.

## 04. Bayes Error for Binary Classification

**Front:** In binary classification, what is the error probability if $p = P(C_1 | \mathbf{x})$?
**Back:**
If we predict the class with the higher posterior, the probability of error for that specific $\mathbf{x}$ is the posterior of the *other* class.

$$
P(\text{error} | \mathbf{x}) = \min(p, 1-p)
$$

If $p > 1-p$, we predict $C_1$ and the error is $1-p$.

## 05. Bayes Error for Multiclass Classification

**Front:** For $K>2$ classes, what is the error probability at a point $\mathbf{x}$ for the Bayes classifier?
**Back:**
The error is one minus the posterior of the winning class.

$$
P(\text{error} | \mathbf{x}) = 1 - \max_k P(C_k | \mathbf{x}) = \sum_{j \neq k^*} P(C_j | \mathbf{x})
$$

where $k^*$ is the class with the highest posterior.

## 06. Decision Boundary for Two Classes

**Front:** How is the optimal decision boundary defined for two classes using Bayes?
**Back:**
The boundary is the set of points $\mathbf{x}$ where the posteriors are equal.

$$
P(C_1 | \mathbf{x}) = P(C_2 | \mathbf{x})
$$

Using Bayes theorem, this simplifies to:

$$
p(\mathbf{x} | C_1) P(C_1) = p(\mathbf{x} | C_2) P(C_2)
$$

## 07. Likelihood Ratio and Priors

**Front:** For two classes, how do the likelihood ratio and prior ratio relate at the decision boundary?
**Back:**
At the Bayes decision boundary:

$$
\frac{p(\mathbf{x} | C_1)}{p(\mathbf{x} | C_2)} = \frac{P(C_2)}{P(C_1)}
$$

* The class boundary shifts if priors are unequal; higher prior pulls the boundary towards the other class.
* Stronger prior belief in $C_1$ means you need less evidence (a lower likelihood ratio) from the data $\mathbf{x}$ to classify as $C_1$

## 01. Decision Boundary with Unequal Priors

**Front:** How do unequal class priors $P(C_1)$ and $P(C_2)$ affect the optimal Bayes decision boundary?
**Back:**
The boundary shifts. A higher prior for a class expands its decision region. The boundary moves *away* from the more probable class and *towards* the less probable class.
**Intuition:** Stronger prior belief in $C_1$ means you need less evidence (a lower likelihood ratio) from the data $\mathbf{x}$ to classify as $C_1$. The threshold on the likelihood ratio is lowered, favoring $C_1$ over more of the input space.

## 04. Pitfall: Direction of the Shift

**Front:** Clarify: "Higher prior pulls the boundary towards the other class." Which way does it move?
**Back:**
This phrasing can be ambiguous. Precise statement: **A higher prior for class $C_k$ causes the decision boundary to shift *away from* the densest regions of $p(\mathbf{x}|C_k)$ and *into* the region more typical of the other class.** For example, if $P(C_1)$ increases, the boundary moves toward the mean of $C_2$, effectively giving $C_1$ a larger share of the feature space.

## 08. Classification without Priors (Likelihood-Only)

**Front:** If class priors $P(C_k)$ are unknown/assumed equal, what simplified rule is used?
**Back:**
Maximize the likelihood. Assign $\mathbf{x}$ to class $C_k$ with the highest $p(\mathbf{x} | C_k)$.

$$
\text{Assign to } C_k \text{ if } p(\mathbf{x} | C_k) > p(\mathbf{x} | C_j) \text{ for all } j \neq k
$$

This is a special case of the Bayes rule with uniform priors.

## 09. Pitfall: Likelihood Sum for Binary Case

**Front:** Is it true that $p(\mathbf{x} | C_1) = 1 - p(\mathbf{x} | C_2)$ for binary classification?
**Back:**
No. This is a common pitfall. Likelihoods $p(\mathbf{x} | C_k)$ are probability *densities* (or masses) for $\mathbf{x}$ under class $C_k$. They are not complementary probabilities. Each is defined by its own class-conditional distribution and they do not sum to 1 over classes.

## 10. Generalizing the Complementary Probability

**Front:** What probabilities *do* sum to 1 for a binary classification problem at a point $\mathbf{x}$?
**Back:**
The posterior probabilities sum to 1.

$$
P(C_1 | \mathbf{x}) + P(C_2 | \mathbf{x}) = 1
$$

If $p$ denotes $P(C_1 | \mathbf{x})$, then $P(C_2 | \mathbf{x}) = 1 - p$. This is true for posteriors, not likelihoods.

## 11. From Error to Risk: The Loss Matrix

**Front:** What is a loss matrix (or risk matrix) $L$ and why is it used?
**Back:**
A matrix $L_{kj}$ specifying the penalty (loss) incurred for assigning a pattern to class $C_j$ when its true class is $C_k$. It generalizes the 0-1 loss (where error counts equally) to applications where some mistakes are more costly than others (e.g., medical diagnosis).

---

## 43. Decision Boundary for Gaussian Binary Classification

**Front:** Derive the Bayes decision boundary for two classes with Gaussian class-conditional densities:

$$
p(\mathbf{x}|C_1) \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)
$$

$$
p(\mathbf{x}|C_2) \sim \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)
$$

using the equality $p(\mathbf{x}|C_1)P(C_1) = p(\mathbf{x}|C_2)P(C_2)$.

**Back:**
**Step 1:** Write the Gaussian PDFs explicitly.

$$
\frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_1|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_1^{-1}(\mathbf{x} - \boldsymbol{\mu}_1)\right) P(C_1)
$$

$$
= \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_2|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}_2^{-1}(\mathbf{x} - \boldsymbol{\mu}_2)\right) P(C_2)
$$

**Step 2:** Take the natural logarithm of both sides.

$$
\ln P(C_1) - \frac{1}{2}\ln|\boldsymbol{\Sigma}_1| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_1^{-1}(\mathbf{x} - \boldsymbol{\mu}_1)
$$

$$
= \ln P(C_2) - \frac{1}{2}\ln|\boldsymbol{\Sigma}_2| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}_2^{-1}(\mathbf{x} - \boldsymbol{\mu}_2)
$$

The $(2\pi)^{d/2}$ terms cancel.

**Step 3:** Rearrange to get the general quadratic discriminant function.

$$
(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_1^{-1}(\mathbf{x} - \boldsymbol{\mu}_1) - (\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}_2^{-1}(\mathbf{x} - \boldsymbol{\mu}_2)
$$

$$
= \ln\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} + 2\ln\frac{P(C_1)}{P(C_2)}
$$

This is a **quadratic decision boundary** in $\mathbf{x}$.

## 44. Special Case: Equal Covariance Matrices

**Front:** Simplify the Gaussian decision boundary when $\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \boldsymbol{\Sigma}$ (equal covariance matrices).
**Back:**
**Step 1:** Starting from the log form:

$$
-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}_1) + \ln P(C_1)
$$

$$
= -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}_2) + \ln P(C_2)
$$

**Step 2:** Cancel $-\frac{1}{2}$ and expand the quadratic forms:

$$
\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - 2\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 - 2\ln P(C_1)
$$

$$
= \mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - 2\boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2 - 2\ln P(C_2)
$$

**Step 3:** Cancel $\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$ and rearrange:

$$
2(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T\boldsymbol{\Sigma}^{-1}\mathbf{x} = \boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 + 2\ln\frac{P(C_1)}{P(C_2)}
$$

**Step 4:** This is a **linear decision boundary**:

$$
(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T\boldsymbol{\Sigma}^{-1}\mathbf{x} = \frac{1}{2}(\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2) + \ln\frac{P(C_2)}{P(C_1)}
$$

## 45. Special Case: Equal Covariance and Equal Priors

**Front:** Further simplify the decision boundary when $\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \boldsymbol{\Sigma}$ and $P(C_1) = P(C_2)$.
**Back:**
With equal priors, $\ln\frac{P(C_2)}{P(C_1)} = \ln(1) = 0$.

The boundary simplifies to:

$$
(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T\boldsymbol{\Sigma}^{-1}\mathbf{x} = \frac{1}{2}(\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2)
$$

This can be rewritten as:

$$
(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T\boldsymbol{\Sigma}^{-1}\left(\mathbf{x} - \frac{\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2}{2}\right) = 0
$$

**Interpretation:** The decision boundary is a **hyperplane** passing through the midpoint $\frac{\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2}{2}$ and orthogonal to $\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$.

In the case of isotropic covariance $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$, this further reduces to:

$$
(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T\mathbf{x} = \frac{1}{2}(||\boldsymbol{\mu}_1||^2 - ||\boldsymbol{\mu}_2||^2)
$$

which is a hyperplane perpendicular to the line joining the means.

## 46. Condition for Minimum Error in Linear Classification

**Front:** Under what conditions is the Bayes optimal decision boundary **linear** for binary classification?
**Back:**
The Bayes decision boundary is linear if and only if the class-conditional densities belong to **any exponential family with equal dispersion parameters**. The most common cases:

**1. Gaussian with equal covariance matrices:**

$$
p(\mathbf{x}|C_1) \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}), \quad p(\mathbf{x}|C_2) \sim \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma})
$$

The log ratio $\ln\frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)}$ is linear in $\mathbf{x}$.

**2. Other exponential family distributions with equal scale:**

- Bernoulli (Naive Bayes) with equal variance
- Poisson with equal rate parameters
- Multinomial with equal dispersion

**Key insight:** The boundary is linear when the **only difference between classes is the mean parameter**, not the covariance/scale/dispersion.

## 47. Deriving the Linear Condition from Log-Likelihood Ratio

**Front:** Show mathematically why equal covariance matrices lead to a linear decision boundary.
**Back:**
**Step 1:** The Bayes decision rule compares $p(\mathbf{x}|C_1)P(C_1)$ and $p(\mathbf{x}|C_2)P(C_2)$. Taking logs:

$$
\ln p(\mathbf{x}|C_1) - \ln p(\mathbf{x}|C_2) > \ln\frac{P(C_2)}{P(C_1)}
$$

**Step 2:** For Gaussians with $\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \boldsymbol{\Sigma}$:

$$
-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_1)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}_1) + \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_2)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}_2) > \ln\frac{P(C_2)}{P(C_1)}
$$

**Step 3:** Expand and cancel $\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$:

$$
(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - \frac{1}{2}(\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2) > \ln\frac{P(C_2)}{P(C_1)}
$$

**Step 4:** Rearrange to form:

$$
(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T\boldsymbol{\Sigma}^{-1}\mathbf{x} > \frac{1}{2}(\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2) + \ln\frac{P(C_2)}{P(C_1)}
$$

**Result:** The left side is **linear in x**, right side is constant → **linear decision boundary**.

---

## 51. Worked Example: Binary Gaussian Classification with Equal Covariance

**Front:** Given binary classes with equal covariance matrix, class means $\boldsymbol{\mu}_1 = [0,0]^T$, $\boldsymbol{\mu}_2 = [3,3]^T$, and precision matrix $\boldsymbol{\Sigma}^{-1} = \begin{bmatrix} 0.95 & -0.15 \\ -0.15 & 0.55 \end{bmatrix}$, classify the point $\mathbf{x} = [1, 2.2]^T$ assuming equal priors $P(C_1) = P(C_2) = 0.5$.

**Back:**
**Step 1: Recall the decision rule for equal covariance and equal priors.**

With equal priors and equal covariance, the Bayes optimal decision is to assign $\mathbf{x}$ to the class with the **smaller Mahalanobis distance**:

$$
\text{Assign to } C_1 \text{ if } D_M(\mathbf{x}, \boldsymbol{\mu}_1) < D_M(\mathbf{x}, \boldsymbol{\mu}_2)
$$

where $D_M(\mathbf{x}, \boldsymbol{\mu}_k) = \sqrt{(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)}$.

**Step 2: Compute Mahalanobis distance to $C_1$.**

$\mathbf{x} - \boldsymbol{\mu}_1 = \begin{bmatrix} 1 - 0 \\ 2.2 - 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 2.2 \end{bmatrix}$

First compute $(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1}$:

$$
\begin{bmatrix} 1 & 2.2 \end{bmatrix} \begin{bmatrix} 0.95 & -0.15 \\ -0.15 & 0.55 \end{bmatrix} = \begin{bmatrix} (1)(0.95) + (2.2)(-0.15) & (1)(-0.15) + (2.2)(0.55) \end{bmatrix}
$$

$$
= \begin{bmatrix} 0.95 - 0.33 & -0.15 + 1.21 \end{bmatrix} = \begin{bmatrix} 0.62 & 1.06 \end{bmatrix}
$$

Now compute $(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_1)$:

$$
\begin{bmatrix} 0.62 & 1.06 \end{bmatrix} \begin{bmatrix} 1 \\ 2.2 \end{bmatrix} = (0.62)(1) + (1.06)(2.2) = 0.62 + 2.332 = 2.952
$$

Thus, $D_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = 2.952$, so $D_M(\mathbf{x}, \boldsymbol{\mu}_1) = \sqrt{2.952} \approx 1.718$.

**Step 3: Compute Mahalanobis distance to $C_2$.**

$\mathbf{x} - \boldsymbol{\mu}_2 = \begin{bmatrix} 1 - 3 \\ 2.2 - 3 \end{bmatrix} = \begin{bmatrix} -2 \\ -0.8 \end{bmatrix}$

First compute $(\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1}$:

$$
\begin{bmatrix} -2 & -0.8 \end{bmatrix} \begin{bmatrix} 0.95 & -0.15 \\ -0.15 & 0.55 \end{bmatrix} = \begin{bmatrix} (-2)(0.95) + (-0.8)(-0.15) & (-2)(-0.15) + (-0.8)(0.55) \end{bmatrix}
$$

$$
= \begin{bmatrix} -1.9 + 0.12 & 0.3 - 0.44 \end{bmatrix} = \begin{bmatrix} -1.78 & -0.14 \end{bmatrix}
$$

Now compute $(\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_2)$:

$$
\begin{bmatrix} -1.78 & -0.14 \end{bmatrix} \begin{bmatrix} -2 \\ -0.8 \end{bmatrix} = (-1.78)(-2) + (-0.14)(-0.8) = 3.56 + 0.112 = 3.672
$$

Thus, $D_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = 3.672$, so $D_M(\mathbf{x}, \boldsymbol{\mu}_2) = \sqrt{3.672} \approx 1.916$.

**Step 4: Compare distances and assign label.**

$$
D_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = 2.95
$$

$$
D_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = 3.67
$$

Since $2.95 < 3.67$, the Mahalanobis distance to $C_1$ is **smaller** than to $C_2$.

**Therefore, assign $\mathbf{x} = [1, 2.2]^T$ to class $C_1$.**

**Step 5: Verify with discriminant function (alternative method).**

The linear discriminant value $(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - \frac{1}{2}(\boldsymbol{\mu}_1^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_2) = -5.04 - (-5.4) = 0.36 > 0$, confirming $C_1$ decision.

---

## 48. When Is Linear Bayes Optimal Also Minimum Error?

**Front:** Does a linear decision boundary guarantee that we are achieving the **minimum possible error** (Bayes error)?
**Back:**
**Yes — if the true class-conditional distributions actually satisfy the linearity condition.**

If $p(\mathbf{x}|C_1)$ and $p(\mathbf{x}|C_2)$ are both Gaussian with **equal covariance matrices**, then:

1. The Bayes optimal decision boundary is **linear**
2. Any classifier that implements this linear boundary (with the correct parameters) achieves the **Bayes error rate**
3. No other classifier, linear or nonlinear, can achieve lower error

**Important nuance:** If you **assume** equal covariance but the true distributions have unequal covariances, then:

- The optimal boundary is **quadratic**
- Forcing a linear boundary increases error above Bayes error
- Linear discriminant analysis (LDA) is suboptimal in this case

**Bottom line:** Linear boundary + correct parameters + true equal covariance = minimum achievable error.

## 49. Pitfall: Assuming Linearity Implies Optimality

**Front:** Is a linear classifier always Bayes optimal if the decision boundary looks linear on training data?
**Back:**
**No — this is a dangerous pitfall.**

**Reasons:**

1. **Misspecification:** The true distributions may have unequal covariances, but sampling variability makes the empirical boundary appear linear
2. **Suboptimal threshold:** Even with correct covariances, using incorrect priors or loss ratios shifts the boundary away from the optimal location
3. **Non-Gaussian data:** Linear boundary can be optimal for some non-Gaussian distributions, but this must be verified — never assume

**Visual pitfall:** A scatter plot showing separable data with a linear gap does **not** prove the Bayes boundary is linear. The optimal boundary might curve between overlapping regions, but the overlap might not be visible in a finite sample.

**Always check:** Test whether allowing quadratic or nonlinear features significantly improves held-out accuracy. If yes, the equal covariance assumption likely fails.

## 50. General Sufficient Condition for Linear Bayes Decision Boundary

**Front:** What is the general sufficient condition for the Bayes decision boundary to be linear?
**Back:**
**Condition:** The log-likelihood ratio $\ln\frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)}$ must be a **linear function** of $\mathbf{x}$.

This occurs when both class-conditional densities belong to the **same exponential family** with identical dispersion/nuisance parameters:

$$
p(\mathbf{x}|C_k) = h(\mathbf{x})g(\theta_k)\exp\left(\eta(\theta_k)^T \mathbf{x}\right)
$$

where only the natural parameter $\eta(\theta_k)$ differs between classes, while $h(\mathbf{x})$ and the base measure are identical.

**Examples:**

- Gaussian with equal covariance ✓
- Bernoulli with equal variance ✓
- Poisson with equal exposure ✓
- Multinomial with equal total counts ✓

**Counter-examples (nonlinear Bayes boundaries):**

- Gaussian with unequal covariance matrices (quadratic)
- Mixture distributions
- Uniform distributions with different supports
- Most non-parametric densities

---

## 11. Loss Matrix: Definition and Structure

**Front:** What is a Loss (or Risk) Matrix L, and how is it indexed for a K-class problem?
**Back:**
A K × K matrix L where element L_{kj} quantifies the penalty for predicting class C_j when the true class is C_k.
**Indices:** The first index k is the **true class**, the second index j is the **predicted class/action**.

$$
\mathbf{L} = \begin{pmatrix}
L_{11} & L_{12} & \cdots & L_{1K} \\
L_{21} & L_{22} & \cdots & L_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
L_{K1} & L_{K2} & \cdots & L_{KK}
\end{pmatrix}
$$

It generalizes the 0-1 loss to handle asymmetric costs (e.g., false negatives cost more than false positives in disease screening).

## 12. Interpreting Matrix Entries and Common Forms

**Front:** How do you interpret the diagonal and off-diagonal entries of L? What is the 0-1 loss matrix?
**Back:**

* **Diagonal (L_{kk}):** Loss for a correct prediction. Often 0 (no penalty).
* **Off-diagonal (L_{kj}, k ≠ j):** Loss for a specific error (predicting j when truth is k).
* **0-1 Loss Matrix:** L_{kj} = 0 if k = j, and 1 if k ≠ j. All errors are equally costly.

$$
\mathbf{L}_{0-1} = \begin{pmatrix}
0 & 1 & \cdots & 1 \\
1 & 0 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 0
\end{pmatrix}
$$

## 13. Expected Risk as a Matrix-Vector Product

**Front:** Given posterior vector p(x) and loss matrix L, how is the conditional risk vector R(x) computed?
**Back:**
The conditional risk R(C_j | x) for taking action C_j is the expected loss over the true class posteriors. For all actions, it's a matrix-vector product:

$$
\mathbf{R}(\mathbf{x}) = \mathbf{L}^T \mathbf{p}(\mathbf{x})
$$

where:

$$
\mathbf{p}(\mathbf{x}) = \begin{bmatrix} P(C_1|\mathbf{x}) \\ P(C_2|\mathbf{x}) \\ \vdots \\ P(C_K|\mathbf{x}) \end{bmatrix},
\quad
\mathbf{R}(\mathbf{x}) = \begin{bmatrix} R(C_1|\mathbf{x}) \\ R(C_2|\mathbf{x}) \\ \vdots \\ R(C_K|\mathbf{x}) \end{bmatrix}
$$

Thus, $ R(C_j|x) = ∑_{k=1}^{K} L_{kj} P(C_k|x) $ is the dot product of the j-th column of L with p(x).

## 13. Expected Risk as a Matrix-Vector Product

**Front:** Given the posterior vector p(x) and loss matrix L, how is the conditional risk for action C_j computed using vector notation?
**Back:**
The conditional risk $ R(C_j | \mathbf{x}) $ is the dot product between the **j-th column of the loss matrix L** and the posterior probability vector p(x).

$$
R(C_j | \mathbf{x}) = \sum_{k=1}^{K} L_{kj} P(C_k | \mathbf{x})
$$

In vector form, if we let $ \mathbf{l}_j $ represent the j-th column of L, then:

$$
R(C_j | \mathbf{x}) = \mathbf{l}_j^T \mathbf{p}(\mathbf{x})
$$

where $ \mathbf{l}_j = [L_{1j}, L_{2j}, ..., L_{Kj}]^T $ and $ \mathbf{p}(\mathbf{x}) = [P(C_1|\mathbf{x}), P(C_2|\mathbf{x}), ..., P(C_K|\mathbf{x})]^T $.

## 14. Risk Column Times Posterior

**Front:** What is the physical interpretation of the operation $ \mathbf{l}_j^T \mathbf{p}(\mathbf{x}) $ for computing conditional risk?
**Back:**
It is the **expected value of the loss** incurred when taking action j, where the expectation is taken over the uncertainty in the true class. The vector $ \mathbf{l}_j $ contains all possible losses for action j (one for each possible true class), and $ \mathbf{p}(\mathbf{x}) $ contains the probabilities of those true classes given the observation x. Their dot product gives the weighted average loss.

## 14. Optimal Decision Rule with Loss Matrix

**Front:** What is the optimal Bayes decision rule using the conditional risk vector R(x)?
**Back:**
Choose the action (class) corresponding to the **minimum element** in the conditional risk vector:

$$
\text{Predict } C_{j^*} \text{ where } j^* = \argmin_{j \in \{1,\dots,K\}} R(C_j | \mathbf{x})
$$

In vector terms: j^* = argmin R(x). This minimizes the total expected loss (Bayes Risk).

---

## 29. Binary Decision Rule with General Loss Matrix

**Front:** For a binary classification problem with classes $C_1$ and $C_2$, what are the conditional risks $R(C_1|\mathbf{x})$ and $R(C_2|\mathbf{x})$ under an arbitrary loss matrix $\mathbf{L}$?
**Back:**
The loss matrix for binary classification is:

$$
\mathbf{L} = \begin{pmatrix}
L_{11} & L_{12} \\
L_{21} & L_{22}
\end{pmatrix}
$$

where $L_{kj}$ is the loss for predicting class $C_j$ when true class is $C_k$.

The conditional risks are:

$$
R(C_1|\mathbf{x}) = L_{11}P(C_1|\mathbf{x}) + L_{21}P(C_2|\mathbf{x})
$$

$$
R(C_2|\mathbf{x}) = L_{12}P(C_1|\mathbf{x}) + L_{22}P(C_2|\mathbf{x})
$$

## 30. Deriving the Likelihood Ratio Threshold

**Front:** Derive the decision rule for binary classification under a general loss matrix in terms of the likelihood ratio $\frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)}$.
**Back:**
**Step 1:** Choose $C_1$ if $R(C_1|\mathbf{x}) < R(C_2|\mathbf{x})$:

$$
L_{11}P(C_1|\mathbf{x}) + L_{21}P(C_2|\mathbf{x}) < L_{12}P(C_1|\mathbf{x}) + L_{22}P(C_2|\mathbf{x})
$$

**Step 2:** Rearrange terms:

$$
(L_{11} - L_{12})P(C_1|\mathbf{x}) < (L_{22} - L_{21})P(C_2|\mathbf{x})
$$

**Step 3:** Substitute Bayes' theorem $P(C_k|\mathbf{x}) = \frac{p(\mathbf{x}|C_k)P(C_k)}{p(\mathbf{x})}$:

$$
(L_{11} - L_{12})\frac{p(\mathbf{x}|C_1)P(C_1)}{p(\mathbf{x})} < (L_{22} - L_{21})\frac{p(\mathbf{x}|C_2)P(C_2)}{p(\mathbf{x})}
$$

**Step 4:** Cancel $p(\mathbf{x}) > 0$ and rearrange to obtain the likelihood ratio threshold:

$$
\frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)} > \frac{(L_{22} - L_{21})P(C_2)}{(L_{11} - L_{12})P(C_1)}
$$

## 42. ⚠️ Pitfall: Sign Error in Loss Matrix Derivation

**Front:** What is the most common algebraic mistake when deriving the binary Bayes decision rule with a loss matrix?
**Back:**
**Forgetting to flip the inequality when dividing by negative numbers.**

$$
(L_{11} - L_{12}) < 0 \quad \text{and} \quad (L_{22} - L_{21}) < 0
$$

**Wrong:** "Divide both sides by $(L_{11} - L_{12})$ and $(L_{22} - L_{21})$"

This preserves the original inequality direction and yields an incorrect threshold.

**Correct:** Multiply by -1 first to make coefficients positive, *then* divide.

**Quick sanity check:**

- If your derived threshold has $(L_{22} - L_{21})$ in numerator and $(L_{11} - L_{12})$ in denominator, the inequality **must** be `>`
- If you see `<`, your sign is flipped

## 31. The General Binary Bayes Decision Rule

**Front:** What is the complete Bayes optimal decision rule for binary classification under an arbitrary loss matrix?
**Back:**
**Predict class $C_1$ if:**

$$
\frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)} > \frac{(L_{22} - L_{21})P(C_2)}{(L_{11} - L_{12})P(C_1)}
$$

**Predict class $C_2$ if:**

$$
\frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)} < \frac{(L_{22} - L_{21})P(C_2)}{(L_{11} - L_{12})P(C_1)}
$$

**If equality holds,** either decision yields the same expected loss.

**Assumptions:** This derivation assumes $L_{11} < L_{12}$ and $L_{22} < L_{21}$ (correct decisions have lower loss than errors), so $(L_{11} - L_{12}) < 0$ and $(L_{22} - L_{21}) < 0$, making both sides positive.

## 32. Special Case: 0-1 Loss Recovers Likelihood Ratio Test

**Front:** Show that the general binary decision rule reduces to the standard Bayes classifier under 0-1 loss.
**Back:**
For 0-1 loss matrix:

$$
\mathbf{L}_{0-1} = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$

where $L_{11} = L_{22} = 0$ and $L_{12} = L_{21} = 1$.

Substitute into the threshold:

$$
\frac{(L_{22} - L_{21})P(C_2)}{(L_{11} - L_{12})P(C_1)} = \frac{(0 - 1)P(C_2)}{(0 - 1)P(C_1)} = \frac{(-1)P(C_2)}{(-1)P(C_1)} = \frac{P(C_2)}{P(C_1)}
$$

Thus the decision rule becomes:

$$
\text{Predict } C_1 \text{ if } \frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)} > \frac{P(C_2)}{P(C_1)}
$$

Multiplying both sides by $P(C_1)p(\mathbf{x}|C_2)$ yields $p(\mathbf{x}|C_1)P(C_1) > p(\mathbf{x}|C_2)P(C_2)$, which is equivalent to $P(C_1|\mathbf{x}) > P(C_2|\mathbf{x})$ — the standard Bayes classifier for minimum error.

## 33. Pitfall: Sign and Inequality Direction

**Front:** What critical assumptions about loss values ensure the threshold $\frac{(L_{22} - L_{21})P(C_2)}{(L_{11} - L_{12})P(C_1)}$ is positive and the inequality direction is correct?
**Back:**
**Assumption 1:** Correct decisions cost less than errors.

$$
L_{11} < L_{12} \quad \text{and} \quad L_{22} < L_{21}
$$

This ensures $(L_{11} - L_{12}) < 0$ and $(L_{22} - L_{21}) < 0$, so their ratio is **positive**.

**Assumption 2:** Loss differences are nonzero.
If $L_{11} = L_{12}$ or $L_{22} = L_{21}$, there is no penalty for misclassifying that class, and the decision rule degenerates (always predict one class).

**Pitfall:** If you accidentally reverse the indices when defining $L_{kj}$, the inequality direction flips. Always verify: $L_{kj}$ = loss when **truth = k, prediction = j**.

## 34. Interpretation: Loss Ratios as Cost-Weighted Priors

**Front:** How can we interpret the threshold $\frac{(L_{22} - L_{21})P(C_2)}{(L_{11} - L_{12})P(C_1)}$ intuitively?
**Back:**
The threshold acts as **cost-weighted effective priors**:

$$
\text{Threshold} = \frac{P(C_2)}{P(C_1)} \times \frac{(L_{22} - L_{21})}{(L_{11} - L_{12})}
$$

- $\frac{P(C_2)}{P(C_1)}$ is the prior odds for $C_2$ vs. $C_1$
- $\frac{(L_{22} - L_{21})}{(L_{11} - L_{12})}$ is the **loss ratio** — the relative cost of misclassifying $C_2$ vs. misclassifying $C_1$

**Intuition:**

- If misclassifying $C_2$ is expensive ($L_{22} - L_{21}$ is large negative magnitude), the threshold **decreases**, making it easier to classify as $C_2$ (conservative toward the expensive-to-miss class)
- If misclassifying $C_1$ is expensive ($L_{11} - L_{12}$ is large negative magnitude), the threshold **increases**, making it easier to classify as $C_1$

This generalizes the prior odds ratio by incorporating asymmetric misclassification costs.

---

## 14. Special Case: 0-1 Loss Matrix

**Front:** What does the 0-1 loss matrix look like, and what does the resulting rule become?
**Back:**
$L_{kj} = 0$ if $k = j$, and $1$ if $k \neq j$.

$$
\mathbf{L}_{0-1} = \begin{pmatrix}
0 & 1 & \cdots & 1 \\
1 & 0 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 0
\end{pmatrix}
$$

The conditional risk becomes:

$$
R(C_j | \mathbf{x}) = \sum_{k \neq j} P(C_k | \mathbf{x}) = 1 - P(C_j | \mathbf{x})
$$

Minimizing this is equivalent to maximizing the posterior $P(C_j | \mathbf{x})$, recovering the standard Bayes classifier for minimum error.

---

## 15. Pitfall: Misinterpreting Total Bayes Error

**Front:** Clarify the statement: "Total error for the Bayes classifier for binary class is $\min\{P(C_1), P(C_2)\} \times (1 - \min\{P(C_1), P(C_2)\})$".
**Back:**
This is incorrect and a major pitfall. This expression actually calculates something else entirely - it's the variance of a Bernoulli random variable with probability $\min\{P(C_1), P(C_2)\}$.

The correct Bayes error is:

1. **Pointwise**: $P(\text{error} | \mathbf{x}) = \min[P(C_1|\mathbf{x}), P(C_2|\mathbf{x})]$
2. **Overall**: $P_{\text{error}} = \mathbb{E}_{\mathbf{x}}[\min[P(C_1|\mathbf{x}), P(C_2|\mathbf{x})]]$

The error depends on the overlap of distributions $p(\mathbf{x}|C_k)$, not just priors.

## 01. Bayes Error Rate: Definition

**Front:** What is the Bayes Error Rate (or Bayes Risk for 0-1 loss)?
**Back:**
The minimum possible error rate achievable by any classifier for a given classification problem. It is the expected error of the optimal (Bayes) classifier, which makes decisions by selecting the class with maximum posterior probability $P(C_k | \mathbf{x})$.

$$
P_{\text{error}} = \mathbb{E}_{\mathbf{x}}[\min\{P(\text{error} | \mathbf{x})\}]
$$

where $P(\text{error} | \mathbf{x}) = 1 - \max_k P(C_k | \mathbf{x})$ is the minimum conditional error at point $\mathbf{x}$.

## 10. Mathematical Expectation: Weighted Average, Not Multiplication

**Front:** Does mathematical expectation $\mathbb{E}_{\mathbf{x}}[f(\mathbf{x})]$ mean multiplying the function $f(\mathbf{x})$ by something to get the Bayes error?
**Back:**
No. Expectation is a **weighted average** over all possible values of $\mathbf{x}$, not a simple multiplication.

**Incorrect interpretation:**

$$
P_{\text{error}} \neq \min[P(C_1|\mathbf{x}), P(C_2|\mathbf{x})] \times \text{(some constant)}
$$

**Correct interpretation:**

$$
P_{\text{error}} = \mathbb{E}_{\mathbf{x}}[\min[P(C_1|\mathbf{x}), P(C_2|\mathbf{x})]] = \int \min[P(C_1|\mathbf{x}), P(C_2|\mathbf{x})] \, p(\mathbf{x}) \, d\mathbf{x}
$$

The expectation:

- **Integrates** (continuous) or **sums** (discrete) over the entire feature space
- **Weights** each point's conditional error by the probability density $p(\mathbf{x})$ of observing that $\mathbf{x}$
- Produces a **single scalar** summarizing the average error across the whole input distribution

You cannot obtain the total Bayes error by simply multiplying a pointwise error—you must integrate over all $\mathbf{x}$.

## 02. Integral Formulation of Bayes Error

**Front:** What is the integral formula for the Bayes error rate?
**Back:**
The Bayes error is the expectation of the conditional error over the entire feature space:

$$
P_{\text{error}} = \int_{\mathcal{X}} \left[1 - \max_k P(C_k | \mathbf{x})\right] p(\mathbf{x}) d\mathbf{x}
$$

Alternatively, it can be expressed as:

$$
P_{\text{error}} = 1 - \int_{\mathcal{X}} \max_k [p(\mathbf{x}|C_k)P(C_k)] d\mathbf{x}
$$

since $p(\mathbf{x}) = \sum_j p(\mathbf{x}|C_j)P(C_j)$.

## 03. Bayes Error for Binary Classification (Explicit Form)

**Front:** For a two-class problem, what is the explicit formula for Bayes error?
**Back:**
For classes $C_1$ and $C_2$:

$$
P_{\text{error}} = \int_{\mathcal{X}} \min\left[P(C_1|\mathbf{x}), P(C_2|\mathbf{x})\right] p(\mathbf{x}) d\mathbf{x}
$$

Equivalently, using the decision regions $\mathcal{R}_1$ and $\mathcal{R}_2$:

$$
P_{\text{error}} = P(C_1) \int_{\mathcal{R}_2} p(\mathbf{x}|C_1) d\mathbf{x} + P(C_2) \int_{\mathcal{R}_1} p(\mathbf{x}|C_2) d\mathbf{x}
$$

where $\mathcal{R}_1$ is where $P(C_1|\mathbf{x}) > P(C_2|\mathbf{x})$, and $\mathcal{R}_2$ is the complement.

## 17. Example: Bayes Error for Univariate Gaussians with Equal Priors

**Front:** Compute the Bayes error for a binary classification problem with:

- Class $C_1$: $p(x|C_1) = \mathcal{N}(-1, 1)$
- Class $C_2$: $p(x|C_2) = \mathcal{N}(1, 1)$
- Equal priors: $P(C_1) = P(C_2) = 0.5$
  **Back:**
  **Step 1: Find decision boundary.** With equal priors, the boundary is where $p(x|C_1) = p(x|C_2)$.

$$
\frac{1}{\sqrt{2\pi}} e^{-\frac{(x+1)^2}{2}} = \frac{1}{\sqrt{2\pi}} e^{-\frac{(x-1)^2}{2}}
$$

Simplifying: $(x+1)^2 = (x-1)^2 \Rightarrow x = 0$

**Step 2: Apply Bayes error integral formula.**

$$
P_{\text{error}} = \int_{-\infty}^{\infty} \min[P(C_1|x), P(C_2|x)] \, p(x) \, dx
$$

With equal priors, $P(C_1|x) > P(C_2|x)$ when $x < 0$, so:

- For $x < 0$: error = $P(C_2|x)$
- For $x > 0$: error = $P(C_1|x)$

**Step 3: Compute using decision regions formulation.**

$$
P_{\text{error}} = P(C_1) \int_{0}^{\infty} p(x|C_1) dx + P(C_2) \int_{-\infty}^{0} p(x|C_2) dx
$$

$$
P_{\text{error}} = 0.5 \int_{0}^{\infty} \mathcal{N}(-1,1) dx + 0.5 \int_{-\infty}^{0} \mathcal{N}(1,1) dx
$$

**Step 4: Evaluate.**
$\int_{0}^{\infty} \mathcal{N}(-1,1) dx = P(Z > 1)$ where $Z \sim \mathcal{N}(0,1) = 0.1587$
$\int_{-\infty}^{0} \mathcal{N}(1,1) dx = P(Z < -1) = 0.1587$

$$
P_{\text{error}} = 0.5(0.1587) + 0.5(0.1587) = 0.1587
$$

**Result:** Bayes error = **15.87%**

## 18. Example: Bayes Error for Completely Separable Classes

**Front:** Compute the Bayes error when classes are perfectly separable with no overlap:

- Class $C_1$: $p(x|C_1) = \mathcal{N}(-2, 0.1)$ (very narrow)
- Class $C_2$: $p(x|C_2) = \mathcal{N}(2, 0.1)$ (very narrow)
- Equal priors: $P(C_1) = P(C_2) = 0.5$
  **Back:**
  **Step 1: Decision boundary.** With nearly no overlap, the boundary is approximately at $x = 0$.

**Step 2: Apply Bayes error integral.**

$$
P_{\text{error}} = P(C_1) \int_{\mathcal{R}_2} p(x|C_1) dx + P(C_2) \int_{\mathcal{R}_1} p(x|C_2) dx
$$

**Step 3: Evaluate the integrals.**
$\int_{0}^{\infty} \mathcal{N}(-2,0.1) dx \approx 0$ (all $C_1$ mass is far left of 0)
$\int_{-\infty}^{0} \mathcal{N}(2,0.1) dx \approx 0$ (all $C_2$ mass is far right of 0)

**Step 4: Compute.**

$$
P_{\text{error}} = 0.5(0) + 0.5(0) = 0
$$

**Result:** Bayes error = **0%** (perfect classification possible)

## 19. Example: Bayes Error for Completely Overlapping Classes

**Front:** Compute the Bayes error when classes are identical (complete overlap):

- Class $C_1$: $p(x|C_1) = \mathcal{N}(0, 1)$
- Class $C_2$: $p(x|C_2) = \mathcal{N}(0, 1)$
- Equal priors: $P(C_1) = P(C_2) = 0.5$
  **Back:**
  **Step 1: Posteriors equal priors everywhere.**
  Since $p(x|C_1) = p(x|C_2)$, we have:

$$
P(C_1|x) = P(C_2|x) = 0.5 \quad \forall x
$$

**Step 2: Bayes decision rule.** Any decision yields error 0.5 at every point.

**Step 3: Apply Bayes error integral.**

$$
P_{\text{error}} = \int_{-\infty}^{\infty} \min[0.5, 0.5] \, p(x) \, dx = \int_{-\infty}^{\infty} 0.5 \, p(x) \, dx = 0.5 \times 1 = 0.5
$$

**Result:** Bayes error = **50%** (random guessing performance)

## 20. Example: Bayes Error with Unequal Priors

**Front:** Compute Bayes error for:

- Class $C_1$: $p(x|C_1) = \mathcal{N}(-1, 1)$
- Class $C_2$: $p(x|C_2) = \mathcal{N}(1, 1)$
- Unequal priors: $P(C_1) = 0.9$, $P(C_2) = 0.1$
  **Back:**
  **Step 1: Find decision boundary.** Boundary satisfies $p(x|C_1)P(C_1) = p(x|C_2)P(C_2)$.

$$
\frac{0.9}{\sqrt{2\pi}} e^{-\frac{(x+1)^2}{2}} = \frac{0.1}{\sqrt{2\pi}} e^{-\frac{(x-1)^2}{2}}
$$

Taking logs and simplifying:

$$
-\frac{(x+1)^2}{2} + \ln(0.9) = -\frac{(x-1)^2}{2} + \ln(0.1)
$$

$$
-\frac{(x+1)^2}{2} + \frac{(x-1)^2}{2} = \ln(0.1) - \ln(0.9) = \ln(1/9) \approx -2.197
$$

Solving: $-2x = -2.197 \Rightarrow x \approx 1.0985$

**Step 2: Apply Bayes error integral (decision regions formulation).**

$$
P_{\text{error}} = P(C_1) \int_{1.0985}^{\infty} p(x|C_1) dx + P(C_2) \int_{-\infty}^{1.0985} p(x|C_2) dx
$$

**Step 3: Evaluate.**
$\int_{1.0985}^{\infty} \mathcal{N}(-1,1) dx = P(Z > 2.0985) \approx 0.018$ where $Z \sim \mathcal{N}(0,1)$
$\int_{-\infty}^{1.0985} \mathcal{N}(1,1) dx = P(Z < 0.0985) \approx 0.539$

**Step 4: Compute.**

$$
P_{\text{error}} = 0.9(0.018) + 0.1(0.539) = 0.0162 + 0.0539 = 0.0701
$$

**Result:** Bayes error = **7.01%** (lower than equal priors case due to strong prior)

## 25. Universal Bound: Bayes Error Cannot Exceed 1 - 1/M

**Front:** For an M-class classification problem with equal priors, what is the absolute maximum possible Bayes error rate $P_e$?
**Back:**

$$
0 \leq P_e \leq 1 - \frac{1}{M}
$$

**Upper bound:** $P_e \leq 1 - \frac{1}{M}$

**Lower bound:** $P_e \geq 0$

**Intuition:** Even in the worst case—complete overlap of all class-conditional distributions—the Bayes classifier can always achieve accuracy at least $1/M$ by random guessing. Therefore, the error cannot exceed $1 - 1/M$.

## 04. Factors Determining Bayes Error

**Front:** What factors determine the Bayes error rate of a classification problem?
**Back:**

1. **Class Overlap:** The overlap between class-conditional distributions $p(\mathbf{x}|C_k)$
2. **Class Priors:** $P(C_k)$
3. **Dimensionality & Separability:** How well separated the classes are in feature space

The Bayes error is fundamentally determined by the inherent ambiguity in the data - how much the classes overlap in the feature space.

## 05. Bayes Error vs. Empirical Error

**Front:** What is the difference between Bayes error and the error of a trained classifier?
**Back:**

- **Bayes Error:** Theoretical lower bound on error rate, determined by data distribution. Unachievable if estimated distributions are imperfect.
- **Empirical Error:** Error rate of a specific classifier on test data. Always $\geq$ Bayes error (asymptotically).
- **Excess Risk:** Difference between classifier's error and Bayes error, due to model mismatch, finite data, or suboptimal training.

## 06. Estimating Bayes Error in Practice

**Front:** How can we estimate the Bayes error rate in practical problems?
**Back:**

1. **Monte Carlo Simulation:** Generate data from known distributions and compute error of Bayes classifier
2. **Plug-in Estimator:** Estimate $p(\mathbf{x}|C_k)$ and $P(C_k)$ from data, then compute integral numerically
3. **Lower Bounds:** Use information-theoretic bounds (Fano, Hellinger)
4. **1-NN Error Bound:** For large samples, error of 1-NN classifier $\leq 2 \times \text{Bayes error}$

## 11. 1-Nearest Neighbor Error Bound

**Front:** What is the theoretical bound relating the error rate of the 1-Nearest Neighbor (1-NN) classifier to the Bayes error rate?
**Back:**
For large training samples (as the number of training examples $N \to \infty$), the asymptotic error rate $P_{\text{1-NN}}$ of the 1-NN classifier is bounded by:

$$
P_{\text{Bayes}} \leq P_{\text{1-NN}} \leq 2 P_{\text{Bayes}} (1 - P_{\text{Bayes}}) \leq 2 P_{\text{Bayes}}
$$

More simply: **The 1-NN error rate is at most twice the Bayes error rate.**

$$
P_{\text{1-NN}} \leq 2 \times P_{\text{Bayes}}
$$

This shows that even this simple, non-parametric classifier achieves an error rate within a factor of 2 of the optimal Bayes classifier in the large-sample limit.

## 13. Pitfall: Misinterpreting the Bound

**Front:** Does the bound $P_{\text{1-NN}} \leq 2P_{\text{Bayes}}$ mean that 1-NN is always at most twice as bad as Bayes?
**Back:**
No. This bound is **asymptotic** (requires $N \to \infty$) and holds **in expectation** over random training sets. Common misinterpretations:

1. **Finite sample:** With small training sets, 1-NN error can be much larger than $2 \times$ Bayes error
2. **Deterministic bound:** It's not a guaranteed upper bound for any specific training set—it's an asymptotic expected bound
3. **Curse of dimensionality:** In high dimensions, the "nearest neighbor" may not be close at all, violating the asymptotic assumption

The bound is theoretically important but practically optimistic for high-dimensional problems.

## 08. Pitfall: Simplistic Formula Fallacy

**Front:** Why is $P_{\text{error}} \neq \min\{P(C_1), P(C_2)\}$ in general?
**Back:**
This formula would only be correct if the classes were **completely separated** (no overlap) and we always guessed the minority class incorrectly. In reality, Bayes error depends on distribution overlap. For example, with equal priors $P(C_1)=P(C_2)=0.5$, Bayes error can range from 0% (perfect separation) to 50% (complete overlap), not fixed at 25% as $\min(0.5,0.5)\times(1-\min(0.5,0.5))=0.25$ would suggest.

## 14. Bayes Error vs. Empirical Error: Integral vs. Count

**Front:** For a binary classification problem, what is the difference between computing Bayes error using the integral formula versus simply counting misclassified test points?
**Back:**
This is the distinction between **theoretical Bayes error** and **empirical error rate**.

**Integral Formula (Theoretical Bayes Error):**

$$
P_{\text{Bayes}} = \int \min[P(C_1|\mathbf{x}), P(C_2|\mathbf{x})] \, p(\mathbf{x}) \, d\mathbf{x}
$$

- Requires knowledge of true distributions $p(\mathbf{x}|C_k)$ and $P(C_k)$
- Is a **population quantity** — the true minimum achievable error
- Cannot be computed exactly from finite data

**Counting Misclassifications (Empirical Error):**

$$
\hat{P}_{\text{error}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i \neq \hat{y}_i)
$$

- Computed from finite test set of $N$ examples
- Is an **estimate** of the classifier's error rate
- Depends on the specific classifier, not necessarily Bayes optimal

**Key Difference:** The integral gives the **true Bayes error** (theoretical lower bound). Counting gives an **empirical estimate** of error for a specific classifier on finite data.

## 15. When Does Counting Approximate the Integral?

**Front:** Under what conditions does counting misclassified test points approximate the Bayes error integral?
**Back:**
Counting approximates the Bayes error **only if**:

1. **Bayes optimal classifier**: You are using the true Bayes decision rule (maximum posterior)
2. **Infinite test data**: As $N_{\text{test}} \to \infty$, the empirical error converges to the true error of your classifier by the law of large numbers
3. **Correct model specification**: Your estimated posteriors match the true $P(C_k|\mathbf{x})$

Even then, the empirical error converges to:

$$
\hat{P}_{\text{error}} \xrightarrow[N \to \infty]{} \int \mathbb{I}(\text{Bayes decision at $\mathbf{x}$ is wrong}) \, p(\mathbf{x}) \, d\mathbf{x}
$$

This is **exactly** the Bayes error integral — but only if your decision rule is truly optimal and your test set is infinite.

## 16. Pitfall: Confusing Empirical Error with Bayes Error

**Front:** Why can't we just report test set accuracy and call it the Bayes error?
**Back:**
This is a common and serious pitfall. Reasons:

1. **Classifier suboptimality**: Your classifier may not be Bayes optimal, so its error is an **upper bound**, not the true Bayes error
2. **Finite sample variance**: Test set error is a random variable with variance $\frac{P_{\text{error}}(1-P_{\text{error}})}{N}$
3. **Overfitting**: Training on test data contaminates the estimate
4. **Distribution mismatch**: Test data may not truly represent $p(\mathbf{x})$

**Correct view:** Test error estimates **your classifier's performance**, not the fundamental Bayes error of the problem. The Bayes error is the **floor**; your classifier's error is somewhere **above** it (or equal in ideal cases).
