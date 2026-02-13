z = \sum c_iW_i

sigmoid activation function is \sigma = 1/1+e^{-z}

out put = 1/1+e^{-z}

E= 1/2 (y- o)^2

for a_k output to the nuran k (o = a_j)):

\partial E/ \partial w_ij = \partial E/ \partial a_j \times \partial a_j / \partial z_j \times \partial z_j / \partial w_ij

for \delta_j as local error signal (sensitivity) : \partial E/ \partial w_ij = \delta_j \times \a_i

for \eta as learning rate : w_ij = w_ij_old - \eta \times \partial E/ \partial w_ij \right arrow \Delta w_ij = - \eta \times \delta_j \times \a_i

calculating \delta_j: (j as current and k as next node)

* for j as the last node: \delta_k = (y_k - a_k) \times f'(z_k)
  * y as true label, a_k as output, f'(z_k) as the derivative of the activation function on the neuran's input (z_k) [! not partial]
* for j as a hidden layer node: \delta_j = (\sum \delta_k \times w_jk) \times f'(z_j)

derivative technique for sigmoid activation function: f = 1/(1 + e^x) \rightarrow f' = f \times (1 - f) ( reminder: f = x \)rightarrow f' = 1

add perceptron algorithm and the cost function of J as:
J_p(w)= - \sum W^T \times X^i  \times Y^i
then add the gradient descent for it.

is sigmoid output zero centered?

mathematical representation and attributes for tanh activation function plus it's derivative

mathematical representation and attributes for relu activation function plus it's derivative
faster converge, dead neurons,

mathematical representation and attributes for Elu activation function plus it's derivative

mathematical representation and attributes for maxout activation function (2 times more parameters) plus it's derivative

mention and analize the gradient vanishing and exploding for all the activations

analysis for layers and parameter counts and overfitting

dropout as a bagging technich for complex NNs with high parameter count

perceptron makes linear classification, so for using it the data must be linearly classifiable

is true to say perceptron for each missclasified data, updates networks weights for once

is it true to say perceptron for training neuron with binary output is used (sign output)

is it true to say perceptron converges to zero error if the data is linearly seperable and the learning rate is good

learning rate should be decreased over the iterations

---

## 01. Neuron Computation Unit

**Front:** What are the three fundamental components of a single artificial neuron's forward computation?
**Back:**

1. Weighted sum: $z = \sum_{i} w_i x_i + b$ (bias term, often absorbed as $w_0$ with $x_0=1$)
2. Activation: $a = f(z)$
3. Output: $a$ (passed to next layer or used as prediction)

$$
z = \sum_i w_i x_i + b, \quad a = f(z)
$$

## 02. Sigmoid Activation Function

**Front:** Write the formula for the sigmoid activation function and its key property regarding output range. Is sigmoid output zero-centered?
**Back:**
The sigmoid function squashes real-valued input to a range (0, 1).

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Zero-centered?** No. Sigmoid outputs are strictly positive (0 to 1). This causes gradient updates to move weights in the same sign direction, leading to inefficient zig-zagging optimization.

Property: $\sigma(z) \in (0, 1)$, monotonic and smooth.

## 03. Derivative of Sigmoid

**Front:** Derive the compact formula for the derivative of the sigmoid function $\sigma(z)$ in terms of $\sigma(z)$ itself.
**Back:**
Let $f = \sigma(z) = \frac{1}{1+e^{-z}}$.
Then $\frac{df}{dz} = f \cdot (1 - f)$.

$$
\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))
$$

This form is computationally efficient as it reuses the forward pass value.

## 04. Tanh Activation Function

**Front:** Write the formula for tanh activation, its output range, zero-centered property, and its derivative.
**Back:**
Tanh is a scaled and shifted sigmoid, output range (-1, 1).

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
$$

**Attributes:** Zero-centered (outputs negative and positive values), mitigates the sign-constraint issue of sigmoid.

**Derivative:**

$$
\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)
$$

## 05. ReLU Activation Function

**Front:** Write the formula for ReLU, its derivative, and its key advantages/disadvantages.
**Back:**
Rectified Linear Unit (ReLU):

$$
\text{ReLU}(z) = \max(0, z)
$$

**Derivative:**

$$
\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
$$

**Attributes:**

- **Faster convergence:** Non-saturating gradient (1) for $z>0$
- **Computationally efficient:** Simple thresholding
- **Dead Neurons:** If $z \leq 0$, gradient = 0; neuron never recovers (dying ReLU problem)

## 06. ELU Activation Function

**Front:** Write the formula for ELU, its derivative, and its key advantages over ReLU.
**Back:**
Exponential Linear Unit (ELU):

$$
\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}
$$

Where $\alpha > 0$ is a hyperparameter (typically $\alpha = 1.0$).

**Derivative:**

$$
\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \text{ELU}(z) + \alpha & \text{if } z \leq 0 \end{cases}
$$

**Attributes:** Smooth negative region, closer to zero-centered, reduces dead neurons, but computationally more expensive than ReLU.

## 07. Maxout Activation Function

**Front:** Describe Maxout activation, its parameter count, and its derivative.
**Back:**
Maxout generalizes ReLU by taking the maximum over $k$ linear functions.

$$
\text{Maxout}(z) = \max(z_1, z_2, ..., z_k)
$$

Where each $z_i = w_i^T x + b_i$.

**Parameter Count:** $k$ times more parameters than a standard neuron (each unit requires $k$ weight vectors and $k$ biases).

**Derivative:**

$$
\frac{\partial}{\partial z_i}\text{Maxout} = \begin{cases} 1 & \text{if } z_i = \max(z_1,...,z_k) \\ 0 & \text{otherwise} \end{cases}
$$

**Attributes:** No saturation, no dead neurons, universal approximator of convex functions, but high parameter cost.

## 08. Loss Function: Squared Error

**Front:** Write the squared error loss function for a single training example.
**Back:**
Measures the squared difference between target $y$ and predicted output $o$.

$$
E = \frac{1}{2}(y - o)^2
$$

The $\frac{1}{2}$ simplifies the derivative, canceling the factor of 2.

## 09. Perceptron Algorithm

**Front:** Describe the perceptron algorithm, its activation function, and its decision boundary.
**Back:**
**Model:** $f(x) = \text{sign}(w^T x + b)$, output ∈ {+1, -1}

**Perceptron Criterion:** Minimize number of misclassifications.

**Update Rule:** For each misclassified example $(x^i, y^i)$ where $y^i \in \{+1, -1\}$:

- If $w^T x^i > 0$ but $y^i = -1$: $w \leftarrow w - \eta x^i$
- If $w^T x^i < 0$ but $y^i = +1$: $w \leftarrow w + \eta x^i$

**Unified update:** $w \leftarrow w + \eta \cdot y^i \cdot x^i$ (only for misclassified points)

**Decision Boundary:** Linear hyperplane $w^T x + b = 0$. Perceptron can only solve **linearly separable** problems.

## 10. Perceptron Cost Function

**Front:** Write the perceptron cost function $J_p(w)$ and its relationship to gradient descent.
**Back:**
The perceptron criterion is defined over misclassified samples $M$:

$$
J_p(w) = -\sum_{i \in M} y^i (w^T x^i)
$$

Where $y^i \in \{-1, +1\}$. For misclassified points, $y^i(w^T x^i) < 0$, making the sum negative; the negative sign makes $J_p(w) > 0$.

**Gradient:**

$$
\nabla J_p(w) = -\sum_{i \in M} y^i x^i
$$

**Gradient Descent Update:**

$$
w \leftarrow w - \eta \nabla J_p(w) = w + \eta \sum_{i \in M} y^i x^i
$$

**Note:** Standard perceptron updates **per misclassified sample** (stochastic), not full batch gradient.

## 11. Perceptron Convergence Properties

**Front:** State three key properties and conditions for perceptron convergence.
**Back:**

1. **Linear Separability:** Perceptron converges to zero training error **if and only if** the data is linearly separable.
2. **Update Frequency:** Perceptron updates weights **once per misclassified example** (not per epoch).
3. **Learning Rate:** With fixed $\eta > 0$, convergence rate is independent of $\eta$ (scales decision boundary). However, for non-separable data, $\eta$ should be **decreased over iterations** to aid convergence.
4. **Output:** Perceptron uses **binary sign output** (±1), not probabilistic outputs.

**Pitfall:** Perceptron does not converge on non-separable data; weights will oscillate.

## 12. Chain Rule for Weight Update

**Front:** Decompose the gradient $\frac{\partial E}{\partial w_{ij}}$ using the chain rule through pre-activation $z_j$ and activation $a_j$.
**Back:**

$$
\frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial a_j} \cdot \frac{\partial a_j}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}
$$

- $\frac{\partial z_j}{\partial w_{ij}} = a_i$ (input from previous neuron)
- $\frac{\partial a_j}{\partial z_j} = f'(z_j)$ (activation derivative)

## 13. Local Error Signal (Sensitivity) δⱼ

**Front:** Define the local error signal $\delta_j$ for neuron $j$.
**Back:**
$\delta_j$ represents the sensitivity of the total error $E$ to the pre-activation input $z_j$ of neuron $j$.

$$
\delta_j \equiv \frac{\partial E}{\partial z_j} = \frac{\partial E}{\partial a_j} \cdot \frac{\partial a_j}{\partial z_j}
$$

This simplifies the weight gradient to a product of $\delta_j$ and the input $a_i$:

$$
\frac{\partial E}{\partial w_{ij}} = \delta_j \cdot a_i
$$

## 14. Gradient Descent Weight Update Rule

**Front:** State the Gradient Descent update rule for a weight $w_{ij}$ using learning rate $\eta$.
**Back:**

$$
w_{ij}^{\text{new}} = w_{ij}^{\text{old}} - \eta \cdot \frac{\partial E}{\partial w_{ij}}
$$

Equivalently, the weight change $\Delta w_{ij}$ is:

$$
\Delta w_{ij} = -\eta \cdot \delta_j \cdot a_i
$$

**Learning Rate Scheduling:** For convergence, $\eta$ is often decreased over iterations (learning rate decay).

## 15. Output Layer δₖ Calculation

**Front:** Derive $\delta_k$ for an output neuron $k$.
**Back:**
For output neuron $k$, $a_k = o_k$. Using squared error loss:

1. $\frac{\partial E}{\partial a_k} = -(y_k - a_k)$ (from $E = \frac{1}{2}(y_k - a_k)^2$)
2. $\frac{\partial a_k}{\partial z_k} = f'(z_k)$

$$
\delta_k = \frac{\partial E}{\partial a_k} \cdot \frac{\partial a_k}{\partial z_k} = -(y_k - a_k) \cdot f'(z_k)
$$

*Note: Some texts omit the negative sign and adjust the update rule; the core concept is the product of error and derivative.*

## 16. Hidden Layer δⱼ Backpropagation

**Front:** Derive $\delta_j$ for a hidden layer neuron $j$ connected to downstream neurons $k$.
**Back:**
Using the chain rule through all outgoing connections:

$$
\delta_j = \frac{\partial E}{\partial z_j} = \sum_k \frac{\partial E}{\partial z_k} \cdot \frac{\partial z_k}{\partial z_j}
$$

Since $\frac{\partial z_k}{\partial z_j} = w_{jk} \cdot f'(z_j)$ and $\delta_k = \frac{\partial E}{\partial z_k}$:

$$
\delta_j = \left( \sum_k \delta_k \cdot w_{jk} \right) \cdot f'(z_j)
$$

## 17. Backpropagation Algorithm Summary

**Front:** List the 4 main steps of the backpropagation algorithm.
**Back:**

1. **Forward Pass:** Compute $z_j$ and $a_j$ for all neurons layer by layer.
2. **Output δ:** Compute $\delta_k = (y_k - a_k) \cdot f'(z_k)$ for all output neurons.
3. **Backward Pass:** Propagate $\delta$ backwards: $\delta_j = (\sum_k \delta_k w_{jk}) \cdot f'(z_j)$ for hidden layers.
4. **Update Weights:** $\Delta w_{ij} = -\eta \cdot \delta_j \cdot a_i$; update all weights.

## 18. Bias Unit Treatment

**Front:** How are bias weights typically handled in the weight update equations?
**Back:**
A bias is treated as a weight $w_{0j}$ connected to a constant input $a_0 = +1$.

- Forward: $z_j = \sum_{i=1}^n w_{ij} a_i + b_j = \sum_{i=0}^n w_{ij} a_i$ (with $a_0=1$, $w_{0j}=b_j$)
- Update: $\Delta w_{0j} = -\eta \cdot \delta_j \cdot 1 = -\eta \cdot \delta_j$

## 19. Vanishing Gradient Problem Analysis

**Front:** Compare vanishing gradient susceptibility across activation functions.
**Back:**

| Activation        | Vanishing Risk      | Mechanism                                    |
| ----------------- | ------------------- | -------------------------------------------- |
| **Sigmoid** | **High**      | Derivative ≤ 0.25; saturates at large\|z\|  |
| **Tanh**    | **High**      | Derivative ≤ 1.0; saturates at large\|z\|   |
| **ReLU**    | **Low** (z>0) | Gradient = 1 for z>0; 0 for z≤0             |
| **ELU**     | **Low** (z>0) | Gradient = 1 for z>0; α for negative region |
| **Maxout**  | **Low**       | Gradient = 1 for max unit; piecewise linear  |

**Exploding Gradient:** Can occur with any activation if weights are large; mitigated by gradient clipping, batch norm, careful initialization.

## 20. Dead Neuron Problem

**Front:** Explain the "dead neuron" (dying ReLU) problem and how ELU/Maxout address it.
**Back:**
**Dead ReLU:** When $z \leq 0$, ReLU output = 0 and gradient = 0. If a neuron consistently receives negative input, it never updates and remains permanently inactive.

**Mitigations:**

- **ELU:** For $z \leq 0$, output is $\alpha(e^z - 1)$ with **non-zero gradient** $\text{ELU}(z) + \alpha$, allowing recovery.
- **Maxout:** Never saturates; always selects a linear unit with gradient 1 for the winner.
- **Leaky ReLU:** Small positive slope (e.g., 0.01) for $z<0$.

## 21. Parameter Count and Overfitting Analysis

**Front:** How does parameter count relate to overfitting in neural networks, and how do activations affect this?
**Back:**
**Parameter Count:**

- Each weight and bias is a trainable parameter.
- Fully connected layer: $(n_{in} + 1) \times n_{out}$ parameters.

**Activation Impact:**

- **Maxout:** $k \times$ parameters of standard layer (significant overfitting risk).
- **ReLU/ELU/Tanh/Sigmoid:** No additional parameters.

**Overfitting:**

- High parameter count → high capacity → memorization risk on small data.
- **Regularization:** Dropout, weight decay, early stopping.
- **Architecture:** Parameter efficiency improves generalization.

**Rule of thumb:** More parameters require more data or stronger regularization.

## 22. Dropout as Bagging

**Front:** Explain Dropout as an approximation to bagging (bootstrap aggregating).
**Back:**
**Dropout:** During training, randomly drop neurons with probability $p$ (or keep with probability $1-p$).

**As Bagging:**

- Each forward/backward pass trains a different **thinned network** (unique subset of weights).
- Weight sharing across models (unlike true bagging with independent models).
- At test time, use **all neurons** with weights scaled by $p$ (weight scaling inference rule).

**Effect:**

- Prevents co-adaptation of neurons.
- Reduces overfitting in high-parameter-count networks.
- Approximates ensemble averaging.

## 23. Pitfall: Vanishing Gradients with Sigmoid/Tanh

**Front:** Explain the vanishing gradient problem in the context of sigmoid activation.
**Back:**
When $|z|$ is large, $\sigma(z) \approx 0$ or $1$, and $\sigma'(z) = \sigma(z)(1-\sigma(z)) \approx 0$.

- **Pitfall:** Deep networks using sigmoid suffer from near-zero gradients in early layers ($\delta_j$ shrinks during backpropagation).
- **Result:** Weights in early layers stop updating; network fails to learn long-range dependencies.
- **Mitigation:** Use ReLU, careful initialization, or batch normalization.

For Tanh: $\tanh'(z) = 1 - \tanh^2(z) \leq 1.0$, also saturates at large |z| causing vanishing gradients.

## 24. Pitfall: Perceptron Non-Convergence

**Front:** What happens when perceptron is trained on non-linearly separable data?
**Back:**
**Pitfall:** Perceptron does **not converge**; weights will oscillate indefinitely as decision boundary flips to correct different misclassified points.

**Misconception:** Decreasing learning rate helps but does not guarantee convergence to zero error.

**Truth:** Perceptron only guarantees convergence (to zero training error) **if data is linearly separable**. For non-separable data, use logistic regression or SVM with soft margin.

## 25. Pitfall: Symmetry Breaking

**Front:** What happens if all weights in a layer are initialized to the same constant value?
**Back:**
**Pitfall:** All neurons receive identical gradients and update identically. They remain symmetric and learn identical features.

**Result:** Layer capacity collapses to single neuron.

**Requirement:** Random initialization (Xavier/Glorot for sigmoid/tanh, He for ReLU) to break symmetry.

## 26. Pitfall: Local Error Signal Sign Convention

**Front:** Common sign convention error in $\delta_k$ derivation.
**Back:**

- **Your note:** $\delta_k = (y_k - a_k) \times f'(z_k)$ (No negative sign)
- **Standard:** $\delta_k = \frac{\partial E}{\partial z_k} = -(y_k - a_k) \cdot f'(z_k)$

**Pitfall:** Inconsistency leads to gradient **ascent** if update rule is $\Delta w = -\eta \delta a$.

**Solution:** Be consistent. If using $\delta_k = -(y_k - a_k)f'(z_k)$, then $\Delta w = -\eta \delta a$ yields gradient descent.

## 27. Pitfall: Activation Function Derivative Confusion

**Front:** Common mistake when applying the derivative chain rule for $\frac{\partial a_j}{\partial z_j}$.
**Back:**

- **Correct:** $a_j = f(z_j)$, so $\frac{\partial a_j}{\partial z_j} = f'(z_j)$.
- **Incorrect:** Mistaking $f'(a_j)$ for $f'(z_j)$. For sigmoid, $f'(z) = f(z)(1-f(z)) = a(1-a)$, which is numerically equal but conceptually distinct.
- **Pitfall:** Miswriting the backpropagation formula for $\delta_j$ as $(\sum \delta_k w_{jk}) \times f'(a_j)$ instead of $f'(z_j)$.

## 28. Pitfall: Batch vs Stochastic Gradient Descent

**Front:** Distinguish between weight update frequency in SGD vs Batch GD in backprop.
**Back:**

- **Stochastic (SGD):** Update weights after every single training example. $\Delta w = -\eta \delta_j a_i$.
- **Batch GD:** Accumulate gradients over entire dataset, then update once.
- **Mini-batch:** Average gradients over a subset.
- **Pitfall:** Using SGD notation but implementing batch averaging without adjusting learning rate or accumulation logic.

## 29. Special Consideration: Cross-Entropy Loss with Sigmoid

**Front:** Why is cross-entropy loss often preferred over squared error for sigmoid output units?
**Back:**
For sigmoid output and squared error, $\delta_k = (y_k - a_k) \cdot a_k(1-a_k)$. When $a_k$ is very wrong (close to 0 when $y_k=1$), the $(1-a_k)$ term saturates the gradient.

For cross-entropy loss $E = -[y \ln a + (1-y) \ln(1-a)]$, the derivative cancels the sigmoid derivative:

$$
\delta_k = a_k - y_k
$$

**Advantage:** Gradient is **proportional to error**, not attenuated by $a_k(1-a_k)$. No vanishing gradient even at saturation.

## 30. Special Consideration: Learning Rate Decay

**Front:** Why should learning rate often be decreased over iterations?
**Back:**
**Rationale:**

- **Early training:** Large $\eta$ for fast progress and escaping poor local minima.
- **Late training:** Small $\eta$ for fine-tuning and convergence stability.

**Methods:**

- Step decay: Reduce by factor every $k$ epochs.
- Exponential decay: $\eta = \eta_0 e^{-kt}$
- 1/t decay: $\eta = \eta_0 / (1 + kt)$

**Perceptron context:** For non-separable data, decreasing $\eta$ can help dampen oscillations, though zero error is not guaranteed.

## 31. Special Consideration: Zero-Centered Activations

**Front:** Why are zero-centered activations (tanh) preferred over non-zero-centered (sigmoid)?
**Back:**
**Sigmoid:** Outputs ∈ (0,1), always positive.

- Gradients for weights connected to a given neuron are **all same sign** (sign of $\delta_j \times$ positive input).
- Causes zig-zagging optimization: weights must decrease/increase together before direction can reverse.

**Tanh:** Outputs ∈ (-1,1), zero-centered.

- Gradients can be positive or negative.
- More efficient, stable optimization.

**Modern solution:** Batch normalization centers activations, reducing dependence on activation function zero-centering.
