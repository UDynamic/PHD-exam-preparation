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

---
