## Constant Summation Formula

**Front:** What is the closed-form expression for summing a constant $c$ from index $n_1$ to $n_2$?
**Back:**
The sum is the constant multiplied by the number of terms.

$$
\sum_{i=n_1}^{n_2} c = (n_2 - n_1 + 1) \cdot c
$$

*Note: The number of terms is $(n_2 - n_1 + 1)$, not $(n_2 - n_1)$.*

## Finite Geometric Series

**Front:** What is the formula for the sum $S_n$ of a finite geometric series $a + ar + ar^2 + ... + ar^{n-1}$ for $r \neq 1$?
**Back:**
The sum is given by the first term times the ratio of one minus the common ratio to the power $n$ to one minus the common ratio.

$$
S_n = a \cdot \frac{1 - r^n}{1 - r}, \quad r \neq 1
$$

## Infinite Geometric Series

**Front:** Under what condition does an infinite geometric series converge, and what is its sum?
**Back:**
It converges if the absolute value of the common ratio is less than 1. The sum is the first term divided by one minus the common ratio.

$$
\text{If } |r| < 1, \quad S = a + ar + ar^2 + ... = \frac{a}{1 - r}
$$

## Limit Definition of Theta Notation (Constant)

**Front:** What does $\lim_{n \to \infty} \frac{f(n)}{g(n)} = c$ (where $0 < c < \infty$) imply about the asymptotic relationship between $f(n)$ and $g(n)$?
**Back:**
It implies that $f(n)$ grows at the same rate as $g(n)$ asymptotically, denoted by Theta notation.

$$
\lim_{n \to \infty} \frac{2n+1}{n+1} = 2 \quad \Rightarrow \quad 2n+1 = \Theta(n)
$$

## Absorption Rule in Asymptotics

**Front:** What is the "absorption rule" when summing terms with different asymptotic growth rates?
**Back:**
In a sum of functions, the term with the fastest growth rate asymptotically dominates and "absorbs" the others. The overall growth is Theta of the fastest growing term.

$$
2^n + n^5 - \log(n) = \Theta(2^n)
$$

## Logarithm Base in Asymptotic Analysis

**Front:** Why does the base of a logarithm not matter in Big-O, Theta, and Omega notations?
**Back:**
Logarithms of different bases are constant multiples of each other. Asymptotic notations ignore constant factors.

$$
\log_a(n) = \frac{\log_b(n)}{\log_b(a)} = \Theta(\log_b(n))
$$

## Asymptotic Growth of Arithmetic Series Sum

**Front:** What is the asymptotic growth (Theta) of the sum of the first $n$ positive integers: $1 + 2 + 3 + ... + n$?
**Back:**
The sum is $\frac{n(n+1)}{2}$, which is a quadratic polynomial. Therefore, its growth is Theta of $n^2$.

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2} = \Theta(n^2)
$$

## Asymptotic Growth of Partial Arithmetic Series

**Front:** What is the asymptotic growth of the sum from $n/2$ to $n$, i.e., $\sum_{i=n/2}^{n} i$?
**Back:**
This sum also has quadratic growth. The number of terms is ~$n/2$ and the average term is ~$3n/4$, leading to a Theta of $n^2$.

$$
\sum_{i=n/2}^{n} i = \Theta(n^2)
$$

## Splitting Sums with Asymptotic Analysis

**Front:** How do you find the asymptotic growth of $T(n) = \sum_{i=1}^{n} (i + \frac{1}{i})$?
**Back:**
Split the sum, analyze the growth of each part, and apply the absorption rule.

$$
\sum_{i=1}^{n} i = \Theta(n^2), \quad \sum_{i=1}^{n} \frac{1}{i} = \Theta(\log n)
$$

$$
\Theta(n^2) + \Theta(\log n) = \Theta(n^2)
$$

## Pitfall: Off-by-One in Constant Summation

**Front:** What is a common error when applying the constant summation formula $\sum_{i=n_1}^{n_2} c$?
**Back:**
Forgetting to add 1 to the difference in limits to get the correct number of terms. The correct formula is $(n_2 - n_1 + 1) \cdot c$, not $(n_2 - n_1) \cdot c$.

## Special Consideration: Infinite vs. Finite Geometric Series

**Front:** When analyzing algorithms, when would you use the infinite geometric series sum formula versus the finite one?
**Back:**
Use the *finite* formula when the number of iterations/terms $n$ is known and explicit (e.g., a loop running $n$ times). Use the *infinite* formula primarily in convergence proofs or when analyzing theoretical infinite processes, which are less common in basic algorithm analysis.

## Pitfall: Misapplying the Absorption Rule

**Front:** When summing functions, can you always drop all but the fastest-growing term to find the Theta bound?
**Back:**
Yes, for *sums*, the absorption rule holds: $\Theta(f(n)) + \Theta(g(n)) = \Theta(\max(f(n), g(n)))$. However, be careful with products (e.g., in nested loops) where terms multiply, requiring different rules.
