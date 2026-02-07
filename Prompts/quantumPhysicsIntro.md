
## 1. Wave-Particle Duality

**Front:** What is the central, surprising concept that distinguishes classical from quantum objects?
**Back:**
Quantum objects, like electrons or photons, exhibit both wave-like and particle-like properties depending on the experiment. This duality is fundamental and not just a limitation of measurement.

## 2. Quantization

**Front:** What does it mean for a physical property to be "quantized"?
**Back:**
The property can only take on specific, discrete values, not a continuous range. It comes in "packets" or quanta. Energy, angular momentum, and other properties in bound systems are quantized.

## 3. Photon & Planck's Relation

**Front:** What is a photon, and how is its energy related to the light's frequency?
**Back:**
A photon is a quantum (particle) of light. Its energy is directly proportional to the frequency of the radiation, with Planck's constant $h$ as the proportionality factor.

$$
E_{\text{photon}} = h\nu = \frac{hc}{\lambda}
$$

## 4. Planck's Constant

**Front:** What is the significance and value of Planck's constant $h$ (and $\hbar$)?
**Back:**
It is the fundamental constant of quantum mechanics, setting the scale for quantum effects. Its small value (~6.626 × 10⁻³⁴ J·s) explains why quantum behavior is not obvious macroscopically. $\hbar = h/(2\pi)$ is often used.

$$
\hbar \equiv \frac{h}{2\pi} \approx 1.055 \times 10^{-34} \, \text{J} \cdot \text{s}
$$

## 5. de Broglie Wavelength

**Front:** What did Louis de Broglie propose about matter, and what is the formula for the matter wavelength?
**Back:**
He proposed that all matter has an associated wavelength, extending wave-particle duality to electrons and other particles. The wavelength is inversely proportional to momentum.

$$
\lambda = \frac{h}{p}
$$

## 6. Wavefunction ($\psi$)

**Front:** What is the primary role of the wavefunction $\Psi(x, t)$ in quantum mechanics?
**Back:**
The wavefunction is a mathematical description (often complex-valued) that contains all possible information about a quantum system. It is a probability amplitude.

## 7. Born Rule (Probability Interpretation)

**Front:** According to the Born rule, how do you find the probability of locating a particle?
**Back:**
The probability density of finding the particle at position $x$ at time $t$ is given by the squared magnitude of the wavefunction. For a 1D case:

$$
P(x, t) \, dx = |\Psi(x, t)|^2 \, dx = \Psi^*(x, t) \Psi(x, t) \, dx
$$

The total probability over all space must be 1 (normalization).

## 8. Heisenberg Uncertainty Principle (Position-Momentum)

**Front:** What is the Heisenberg Uncertainty Principle for position and momentum?
**Back:**
It states a fundamental limit on the simultaneous knowledge of a particle's position ($x$) and momentum ($p_x$). The product of their uncertainties has a minimum value.

$$
\Delta x \, \Delta p_x \geq \frac{\hbar}{2}
$$

## 9. Schrödinger Equation (Time-Dependent)

**Front:** What is the equation that governs the evolution in time of a quantum system's wavefunction?
**Back:**
The time-dependent Schrödinger equation (TDSE) is the core dynamical law of non-relativistic quantum mechanics.

$$
i\hbar \frac{\partial}{\partial t} \Psi(x,t) = \hat{H} \Psi(x,t) = \left[ -\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} + V(x,t) \right] \Psi(x,t)
$$

## 10. Hamiltonian Operator ($\hat{H}$)

**Front:** What is the Hamiltonian operator $\hat{H}$ in the Schrödinger equation?
**Back:**
It is the operator corresponding to the total energy of the system. It is the sum of the kinetic energy operator ($-\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2}$) and the potential energy operator ($V(x,t)$).

## 11. Stationary States

**Front:** What defines a stationary state, and what is special about its time dependence?
**Back:**
A stationary state is an eigenstate of the Hamiltonian with definite energy $E$. Its wavefunction separates into spatial and temporal parts: $\Psi(x,t) = \psi(x) e^{-iEt/\hbar}$. All probabilities ($|\Psi|^2$) are constant in time.

## 12. Time-Independent Schrödinger Equation (TISE)

**Front:** What equation determines the spatial wavefunction $\psi(x)$ and energy $E$ of a stationary state?
**Back:**
The Time-Independent Schrödinger Equation (TISE), which is an eigenvalue equation for the Hamiltonian.

$$
-\frac{\hbar^2}{2m} \frac{d^2 \psi(x)}{dx^2} + V(x)\psi(x) = E \psi(x)
$$

## 13. Quantum Superposition

**Front:** What is the principle of superposition in quantum mechanics?
**Back:**
If $\psi_1$ and $\psi_2$ are possible states of a system, then any linear combination $\psi = a\psi_1 + b\psi_2$ is also a possible state. This leads to interference effects.

## 14. Quantum Tunneling

**Front:** What is quantum tunneling?
**Back:**
It is a quantum phenomenon where a particle has a non-zero probability of passing through a potential energy barrier even when its total energy is less than the barrier height. This is impossible in classical mechanics.

## 15. Pitfall: The Wavefunction is Not Physical

**Front:** What is a common misconception about the wavefunction $\psi$ itself?
**Back:**
$\psi$ is a *probability amplitude*, not a physical wave like sound or water. It can be complex-valued. Only $|\psi|^2$, the probability density, is directly related to physical measurement outcomes.

## 16. Pitfall: Uncertainty from Measurement Only

**Front:** Is the Heisenberg Uncertainty Principle solely about the disturbance caused by measurement?
**Back:**
No. It is a fundamental limit inherent in the quantum nature of a system, arising from the wave-like description (e.g., a wave packet has inherent spread in position and momentum). It exists even in principle with perfect measurements.

## 17. Consideration: The Measurement Problem

**Front:** What special consideration does the act of measurement introduce in quantum mechanics?
**Back:**
Measurement is not passive. The "measurement problem" notes that upon measurement, the wavefunction appears to "collapse" from a superposition into a single eigenstate corresponding to the measured value. The mechanism for this is a foundational interpretational issue.
