## Recurrence Relations: Linear Shrink (n-c)

**Front:** What is the general form and solution approach for a recurrence relation where the argument shrinks by a constant 'c'? Example: $T(n) = A \cdot T(n-c) + f(n)$.
**Back:**
The recursion tree has a linear structure. The number of levels (height) is determined by how many times you can subtract 'c' from 'n' before reaching the base case. The height is $h = \frac{n}{c}$ (for base case T(1)). The total work is the sum of $f(n - i \cdot c)$ across levels $i = 0$ to $h-1$, scaled by $A^i$.

## Recurrence Relations: Root Shrink (sqrt(n))

**Front:** How do you solve a recurrence of the form $T(n) = A \cdot T(\sqrt{n}) + f(n)$?
**Back:**
Use a variable substitution to linearize the shrinking argument. Set $n = 2^k$. Then $\sqrt{n} = 2^{k/2}$. Define a new function $F(k) = T(2^k)$. The recurrence becomes:

$$
F(k) = A \cdot F\left(\frac{k}{2}\right) + f(2^k)
$$

This is now a divide-and-conquer style recurrence in 'k'.

## Height for sqrt(n) Recurrence

**Front:** After substituting $n = 2^k$ for $T(n) = A \cdot T(\sqrt{n}) + f(n)$, what is the height $h_T$ of the original recurrence tree?
**Back:**
The new recurrence is $F(k) = A \cdot F(k/2) + f(2^k)$. Its height in terms of 'k' is $h_F = \log_2(k)$. Since $k = \log_2(n)$, the height of the original recurrence $T(n)$ is:

$$
h_T = \log_2(\log_2(n))
$$

It is **doubly logarithmic** in n.

## Absorption Rule for Fractional Arguments

**Front:** What is the "absorption rule" when simplifying fractional arguments in recurrence relations (e.g., for Master Theorem analysis)?
**Back:**
For large 'n', lower-order terms and additive constants in the numerator and denominator of a fractional argument can be absorbed/ignored. The asymptotic behavior depends only on the dominant terms. Examples:

$$
T\left(\frac{n}{n^{2/3} + 1}\right) \approx T(n^{1/3}), \quad T\left(\frac{n^2 + 3}{\sqrt{n} + 5}\right) \approx T\left(\frac{n^2}{\sqrt{n}}\right) = T(n^{3/2})
$$

## Absorption for Linear Fractions

**Front:** How does the absorption rule apply to arguments like $T(\frac{an+b}{c})$?
**Back:**
Additive constants in the numerator can be absorbed. The constant factor in the denominator is critical and must be kept. Example:

$$
T\left(\frac{2n + 1}{3}\right) \approx T\left(\frac{2n}{3}\right)
$$

The base of the logarithm in the recursion tree height will be $3/2$, not $3$.

## Floors and Ceilings in Recurrences

**Front:** How do floors and ceilings affect the asymptotic analysis of recurrence arguments?
**Back:**
They are typically **effectless** for asymptotic analysis. For example:

$$
T(\lceil n/2 \rceil) \text{ and } T(\lfloor n/2 \rfloor) \text{ are both treated as } T(n/2)
$$

This simplification is standard in theorems like the Master Theorem.

## Generalized Power Shrink Recurrence

**Front:** What is the height of the recursion tree for $T(n) = A \cdot T(n^c) + f(n)$ where $0 < c < 1$?
**Back:**
Use substitution $n = b^k$ (or $n = 2^k$). Then $n^c = (b^k)^c = b^{ck} = b^{k \cdot c}$. Let $F(k) = T(b^k)$. The recurrence becomes $F(k) = A \cdot F(c \cdot k) + f(b^k)$. The argument shrinks multiplicatively by 'c' in 'k'-space. The height $h_F$ is $\log_{1/c}(k)$. Since $k = \log_b(n)$:

$$
h_T = \log_{1/c}(\log_b(n))
$$

For base $b=2$: $h_T = \log_{1/c}(\log_2(n))$.

## Multi-Parameter Recurrence Height

**Front:** For a two-parameter recurrence like $T(n,k) = T(n/2, k) + T(n, k/4) + f(n,k)$, how do you find the maximum recursion tree height?
**Back:**
The height is determined by the branch(es) that divide parameters sequentially until base cases (e.g., $T(1,k)=1, T(n,1)=1$). To find the **maximum** height, follow the path that performs divisions *intermittently* on different parameters to delay reaching the base.

$$
h_{max} = \log_2(n) + \log_4(k)
$$

This represents dividing 'n' by 2 until it becomes 1, and *then* dividing 'k' by 4 until it becomes 1 (or vice-versa, summing the steps).

## Pitfall: Non-Integer Heights

**Front:** What is a common pitfall when calculating recursion tree height for relations like $T(n) = T(n-c) + f(n)$?
**Back:**
Assuming $h = n/c$ gives an integer. In precise analysis, the height is $\lfloor n/c \rfloor$ or $\lceil n/c \rceil$. While asymptotically $( \Theta(n/c) )$, for exact closed forms (especially with small 'n' or specific base cases), the floor/ceiling matters.

## Pitfall: Substitution Domain Mismatch

**Front:** What can go wrong with the substitution $n = 2^k$ for $T(n) = A \cdot T(\sqrt{n}) + f(n)$?
**Back:**
The substitution assumes 'n' is a perfect power of 2 at every recursive step. This is fine for asymptotic analysis but can fail for exact solutions if 'n' is not a power of 2. Always state the assumption $n = 2^{2^m}$ for complete rigor, or rely on smoothness assumptions from the Master Theorem.

## Special Consideration: Absorbing Constants in Denominator

**Front:** Why is absorbing a multiplicative constant in the denominator (e.g., treating $T(n/3)$ as $T(n/2)$) dangerous?
**Back:**
It changes the **base of the logarithm** in the tree height and the critical exponent in Master Theorem case comparisons. $T(n/3)$ yields $\log_3 n$ levels, while $T(n/2)$ yields $\log_2 n$ levels. These are asymptotically different ($\Theta(\log n)$) but the constant matters in the polynomial factor $n^{\log_b a}$.

## Special Consideration: Multi-Parameter Base Cases

**Front:** In multi-parameter recurrences, what must be carefully checked regarding base cases?
**Back:**
The recursion stops when *any* parameter reaches its base condition. The maximum height calculation must find the longest path to *any* base case, not necessarily to a point where *all* parameters are at their base simultaneously. The sequence of divisions that most delays hitting any single base case determines the height.
