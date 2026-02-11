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
This is incorrect and a major pitfall. The expression $\min(p, 1-p)$ is the *conditional* error at a point $\mathbf{x}$. The *total* Bayes error (or Bayes error rate) is the expected value of this over all $\mathbf{x}$:

$$
P_{\text{error}} = \int \min[P(C_1 | \mathbf{x}), P(C_2 | \mathbf{x})] p(\mathbf{x}) d\mathbf{x}
$$

It is not simply a function of the priors alone; it depends on the overlap of the class-conditional distributions $p(\mathbf{x} | C_k)$.

## 01. Bayes Error Rate: Definition

**Front:** What is the Bayes Error Rate (or Bayes Risk for 0-1 loss)?
**Back:**
The minimum possible error rate achievable by any classifier for a given classification problem. It is the expected error of the optimal (Bayes) classifier, which makes decisions by selecting the class with maximum posterior probability $P(C_k | \mathbf{x})$.

$$
P_{\text{error}} = \mathbb{E}_{\mathbf{x}}[\min\{P(\text{error} | \mathbf{x})\}]
$$

where $P(\text{error} | \mathbf{x}) = 1 - \max_k P(C_k | \mathbf{x})$ is the minimum conditional error at point $\mathbf{x}$.

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
4. **1-NN Error Bound:** For large samples, error of 1-NN classifier $\leq 2 \times$ Bayes error

## 07. Pitfall: Confusion with Minimum Conditional Error

**Front:** What is a common misconception about the relationship between $P(\text{error}|\mathbf{x})$ and $P_{\text{error}}$?
**Back:**
Confusing the **pointwise conditional error** $P(\text{error}|\mathbf{x}) = \min\{P(C_1|\mathbf{x}), P(C_2|\mathbf{x})\}$ with the **overall Bayes error** $P_{\text{error}}$. The former is a function of $\mathbf{x}$; the latter is its expectation over $p(\mathbf{x})$. They are not equal except in trivial cases.

## 08. Pitfall: Simplistic Formula Fallacy

**Front:** Why is $P_{\text{error}} \neq \min\{P(C_1), P(C_2)\}$ in general?
**Back:**
This formula would only be correct if the classes were **completely separated** (no overlap) and we always guessed the minority class incorrectly. In reality, Bayes error depends on distribution overlap. For example, with equal priors $P(C_1)=P(C_2)=0.5$, Bayes error can range from 0% (perfect separation) to 50% (complete overlap), not fixed at 25% as $\min(0.5,0.5)\times(1-\min(0.5,0.5))=0.25$ would suggest.
