## Parameter Estimation

**Front:** What is the fundamental goal of parameter estimation in statistical modeling?
**Back:**
To infer the unknown parameters $\theta$ of a probability distribution, given observed data $x$. We want to find the most plausible $\theta$ that explains our data.

## Likelihood Function

**Front:** What is the likelihood function $p(x | \theta )$?
**Back:**
It is a function of the parameters $\theta$. It measures the probability (or probability density) of observing the given data $x$ under a specific setting of the parameters. It is *not* a probability distribution over $\theta$.

$$
p(x | \theta )
$$

## Prior Distribution

**Front:** What is the prior distribution $p(\theta)$?
**Back:**
It represents our belief about the parameters $\theta$ *before* seeing any data. It incorporates domain knowledge or assumptions.

$$
p(\theta)
$$

## Bayes' Theorem for Parameters

**Front:** State Bayes' Theorem as it applies to parameter estimation.
**Back:**
It updates our belief about parameters $\theta$ after observing data $x$. The prior $p(\theta)$ is combined with the likelihood $p(x | \theta)$ to form the posterior.

$$
p(\theta | x) = \frac{p(x | \theta ) \cdot p(\theta)}{p(x)}
$$

## Posterior Distribution

**Front:** What is the posterior distribution $p(\theta | x)$?
**Back:**
It represents our updated belief about the parameters $\theta$ *after* observing the data $x$. It is the main output of Bayesian inference, combining prior knowledge and empirical evidence.

$$
p(\theta | x)
$$

## Evidence (Marginal Likelihood)

**Front:** What is the evidence or marginal likelihood $p(x)$?
**Back:**
The total probability of the data under all possible parameter settings, weighted by the prior. It serves as a normalizing constant for the posterior.

$$
p(x) = \int p(x | \theta) p(\theta) d\theta
$$

## Maximum A Posteriori (MAP) Estimation

**Front:** What is the Maximum A Posteriori (MAP) estimate $\hat{\theta}_{MAP}$?
**Back:**
It is the mode (peak) of the posterior distribution. It is the single most probable parameter value given the data and the prior.

$$
\hat{\theta}_{MAP} = \arg \max_{\theta} p(\theta | x)
$$

## MAP as Optimized Likelihood * Prior

**Front:** How is the MAP estimate computed in practice, ignoring constants?
**Back:**
Since the evidence $p(x)$ is independent of $\theta$, we maximize the unnormalized posterior: the likelihood times the prior.

$$
\hat{\theta}_{MAP} = \arg \max_{\theta} p(\theta | x) = \arg \max_{\theta} [ p(x | \theta ) \cdot p(\theta) ]
$$

## Maximum Likelihood Estimation (MLE)

**Front:** What is the Maximum Likelihood Estimate (MLE), and how does it relate to MAP?
**Back:**
MLE is $\hat{\theta}_{MLE} = \arg \max_{\theta} p(x | \theta)$. It is a special case of MAP where the prior $p(\theta)$ is a uniform (flat) distribution. MAP introduces a regularization effect via the prior.

## Practical Computation: Logarithm

**Front:** Why do we typically maximize the log-posterior instead of the posterior directly?
**Back:**
The logarithm is a monotonically increasing function, so the argmax is unchanged. It converts products into sums, which is numerically more stable, especially with many data points.

$$
\hat{\theta}_{MAP} = \arg \max_{\theta} [ \log p(x | \theta) + \log p(\theta) ]
$$

## Pitfall: Misinterpreting the Posterior

**Front:** What is a key pitfall in interpreting a single MAP estimate versus the full posterior?
**Back:**
The MAP estimate is just one point. It discards all other information about the shape of the posterior (e.g., uncertainty, skewness, multimodality). The full posterior distribution provides a complete picture of parameter uncertainty.

## Pitfall: Prior Sensitivity

**Front:** When can the MAP estimate be problematic?
**Back:**
When the prior $p(\theta)$ is chosen poorly (too informative or misspecified) or when the data is scarce, the MAP estimate can be overly influenced by the prior. It's crucial to perform sensitivity analysis.
