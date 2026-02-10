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

## 02. The Evidence Term

**Front:** Why is the evidence $p(\mathbf{x})$ often omitted when comparing posteriors for different classes?
**Back:**
Because $p(\mathbf{x}) = \sum_k p(\mathbf{x} | C_k) P(C_k)$ is constant for all classes given $\mathbf{x}$. For comparison/decision making, we only need the numerator: $p(\mathbf{x} | C_k) P(C_k)$. The evidence acts as a normalization constant ensuring posteriors sum to 1.

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

The class boundary shifts if priors are unequal; higher prior pulls the boundary towards the other class.

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

## 12. Expected Risk (Conditional Risk)

**Front:** Given a loss matrix $L$, how is the expected risk (conditional risk) $R(C_j | \mathbf{x})$ for taking action $C_j$ calculated?
**Back:**
It's the expected loss for choosing class $C_j$ at point $\mathbf{x}$, averaged over the true classes.

$$
R(C_j | \mathbf{x}) = \sum_{k=1}^{K} L_{kj} P(C_k | \mathbf{x})
$$

The optimal decision minimizes this conditional risk.

## 13. Bayes Risk Minimizer

**Front:** What is the Bayes decision rule under a general loss matrix?
**Back:**
For each $\mathbf{x}$, choose the class $C_j$ that minimizes the conditional risk $R(C_j | \mathbf{x})$.

$$
\text{Assign to } C_j \text{ if } R(C_j | \mathbf{x}) < R(C_m | \mathbf{x}) \text{ for all } m \neq j
$$

This minimizes the total expected loss (Bayes risk).

## 14. Special Case: 0-1 Loss Matrix

**Front:** What does the 0-1 loss matrix look like, and what does the resulting rule become?
**Back:**
$L_{kj} = 0$ if $k = j$, and $1$ if $k \neq j$. The conditional risk becomes:

$$
R(C_j | \mathbf{x}) = \sum_{k \neq j} P(C_k | \mathbf{x}) = 1 - P(C_j | \mathbf{x})
$$

Minimizing this is equivalent to maximizing the posterior $P(C_j | \mathbf{x})$, recovering the standard Bayes classifier for minimum error.

## 15. Pitfall: Misinterpreting Total Bayes Error

**Front:** Clarify the statement: "Total error for the Bayes classifier for binary class is $\min\{P(C_1), P(C_2)\} \times (1 - \min\{P(C_1), P(C_2)\})$".
**Back:**
This is incorrect and a major pitfall. The expression $\min(p, 1-p)$ is the *conditional* error at a point $\mathbf{x}$. The *total* Bayes error (or Bayes error rate) is the expected value of this over all $\mathbf{x}$:

$$
P_{\text{error}} = \int \min[P(C_1 | \mathbf{x}), P(C_2 | \mathbf{x})] p(\mathbf{x}) d\mathbf{x}
$$

It is not simply a function of the priors alone; it depends on the overlap of the class-conditional distributions $p(\mathbf{x} | C_k)$.
