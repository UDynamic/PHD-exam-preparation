## Integration: General Power Rule

**Front:** What is the general rule for integrating a power of `n`, i.e., `n^A`? What is the special case?
**Back:**
For `A ≠ -1`: $\int n^A \, dn = \frac{n^{A+1}}{A+1} + C$

For the special case `A = -1`: $\int n^{-1} \, dn = \int \frac{1}{n} \, dn = \log(n) + C$

## Integration: "Multiply by n" Trick

**Front:** What is the quick integral trick for a term of the form `n^A * log(n)^B`, when it's *not* a special `n^{-1}` case? What is the result of $\int n^2 \log(n)$?
**Back:**
The trick: Multiply the integrand by `n` (which effectively integrates `n^A` to `n^{A+1}` while keeping the `log(n)` part).

Result: $\int n^2 \log(n) \approx n^3 \log(n)$.
(Note: This is an approximation/rule-of-thumb for quick asymptotic analysis, not the exact antiderivative).

## Integration: "Multiply by log(n)" Trick

**Front:** What is the quick integral trick for a term containing `n^{-1}`, like `(1/n) * log(n)^B`? What is $\int (1/n) \log^5(n)$?
**Back:**
The trick: Multiply the integrand by `log(n)`. This accounts for the integral of `1/n` being `log(n)`.

Result: $\int \frac{1}{n} \log^5(n) \approx \log^6(n)$.

## Integration: "Multiply by log(log(n))" Trick

**Front:** What is the quick integral trick for a term of the form `(n log(n))^{-1}`? What is $\int \frac{1}{n \log(n)}$?
**Back:**
The trick: Multiply the integrand by `log(log(n))`.

Result: $\int \frac{1}{n \log(n)} \approx \log(\log(n))$.

## Integration: Dominant Term Estimation

**Front:** How do you quickly estimate an integral of a sum, e.g., $\int (1/n + 1/n^2) \, dn$?
**Back:**
Identify the *dominant term* (the one that decays slowest/grows fastest). For large `n`, `1/n` dominates `1/n^2`. Use the trick for the dominant term.

Estimate: $\int (1/n + 1/n^2) \, dn \approx \int (1/n) \, dn \approx \log(n)$.

## Recursion Tree Components

**Front:** In a recurrence $T(n) = A \cdot T(n/b) + f(n)$, what do the constants `A`, `b`, and the function `f(n)` represent in the recursion tree?
**Back:**

* `A`: The *branching factor* (number of children per node).
* `b`: The *size reduction factor* per level.
* `f(n)`: The *cost of the node* at the root level (work done per node at that level).

## Recursion Tree: Level Sum & Height

**Front:** In the recursion tree for $T(n) = A \cdot T(n/b) + f(n)$, what is the sum of costs at level `i` and the tree height `h`?
**Back:**

* **Level Cost:** Level `i` has $A^i$ nodes, each with cost $f(n / b^i)$. Total cost at level `i`: $A^i \cdot f(n / b^i)$.
* **Height:** The recursion stops when the subproblem size hits a constant (e.g., 1). So, $n / b^h = O(1) \Rightarrow h = \Theta(\log_b n)$.

## Master Theorem (Standard Form & Variables)

**Front:** What is the standard form of a recurrence solvable by the Master Theorem? Define the critical exponent `p`.
**Back:**
Standard form: $T(n) = A \cdot T(n/b) + f(n)$, where $A \ge 1$, $b > 1$, and $f(n)$ is asymptotically positive.

Critical exponent: $p = \log_b A$. It represents the *intrinsic growth rate* from the recursion alone.

## Master Theorem (Three Cases)

**Front:** What are the three cases of the Master Theorem, based on comparing $f(n)$ and $n^p$?
**Back:**
Let $p = \log_b A$.

1. **Leaf-heavy:** If $f(n) = O(n^{p - \epsilon})$ for $\epsilon > 0$, then $T(n) = \Theta(n^p)$.
2. **Balanced:** If $f(n) = \Theta(n^p \log^k n)$, then $T(n) = \Theta(n^p \log^{k+1} n)$. (For $k=0$: $T(n)=\Theta(n^p \log n)$).
3. **Root-heavy:** If $f(n) = \Omega(n^{p + \epsilon})$ *and* $A f(n/b) \le c f(n)$ for some $c<1$, then $T(n) = \Theta(f(n))$.

## Master Theorem: Integral Form for Non-Polynomial f(n)

**Front:** When $f(n)$ is not a simple polynomial (e.g., $n/\log n$), what is the general integral formula for $T(n) = A T(n/b) + f(n)$?
**Back:**
$T(n) = \Theta\left( n^p \left( 1 + \int_1^n \frac{f(x)}{x^{p+1}} \, dx \right) \right)$, where $p = \log_b A$.

*Note: The solution cannot be asymptotically smaller than $n^p$, regardless of $f(n)$.*

## Example: Master Theorem with Integral

**Front:** Solve $T(n) = 2 T(n/2) + n / \log n$ using the integral form. ($p=\log_2 2 = 1$)
**Back:**
$T(n) = \Theta\left( n \left( 1 + \int_1^n \frac{x / \log x}{x^{2}} \, dx \right) \right) = \Theta\left( n \left( 1 + \int_1^n \frac{1}{x \log x} \, dx \right) \right)$.

We know $\int \frac{1}{x \log x} dx = \log(\log x)$. Thus, $T(n) = \Theta(n \log(\log n))$.

## Variable Substitution in Recurrences

**Front:** How do you solve $T(n) = 4 T(\sqrt{n} / 4) + \log^2 n$? What substitution is used?
**Back:**
Use the substitution $n = 2^k$ (or $n = b^k$ to simplify the recursive term).

1. Let $n = 2^k$. Then $\sqrt{n} = 2^{k/2}$.
2. Define $F(k) = T(2^k)$.
3. Recurrence becomes: $F(k) = 4 F(k/2 - 2) + k^2$. For asymptotic analysis, we simplify to $F(k) = 4 F(k/2) + k^2$.
4. Solve $F(k)$ using the Master Theorem ($A=4, b=2, f(k)=k^2, p=2$). Since $f(k)=k^2=O(k^{2})$ (Case 2 with $k=0$), $F(k) = \Theta(k^2 \log k)$.
5. Substitute back: $k = \log n \Rightarrow T(n) = \Theta(\log^2 n \cdot \log(\log n))$.

## Dividing by a Product Trick (Non-constant A)

**Front:** How do you solve a recurrence where `A` is not constant, like $T(n) = n \cdot T(n-1) + (n+1)!$?
**Back:**
Divide both sides by a product (often `n!`) to create a telescoping series.

1. Divide by `n!`: $\frac{T(n)}{n!} = \frac{T(n-1)}{(n-1)!} + (n+1)$.
2. Let $F(n) = T(n)/n!$. Then $F(n) = F(n-1) + (n+1)$.
3. This telescopes: $F(n) = \sum_{i=1}^n (i+1) = \Theta(n^2)$.
4. Thus, $T(n) = n! \cdot F(n) = \Theta(n^2 \cdot n!)$.

## Akra-Bazzi Theorem (General Form)

**Front:** What is the general form of a recurrence solvable by the Akra-Bazzi theorem? What equation defines the exponent `p`?
**Back:**
General form: $T(n) = \sum_{i=1}^k a_i T(b_i n + h_i(n)) + f(n)$, with $a_i>0$, $0<b_i<1$, and $|h_i(n)|=O(n/\log^2 n)$.

Exponent `p` is the unique real number satisfying: $\sum_{i=1}^k a_i b_i^p = 1$.

## Akra-Bazzi Theorem (Solution Formula)

**Front:** Given the Akra-Bazzi exponent `p`, what is the formula for $T(n)$?
**Back:**
$T(n) = \Theta\left( n^p \left( 1 + \int_1^n \frac{f(u)}{u^{p+1}} \, du \right) \right)$.

This generalizes the Master Theorem's integral form to multiple recursive terms.

## Akra-Bazzi: Special Case for f(n) = Θ(n^k)

**Front:** If $f(n) = \Theta(n^k)$ in an Akra-Bazzi recurrence, what are the two possible outcomes based on the exponent `p` compared to `k`?
**Back:**
Let `p` be the solution to $\sum a_i b_i^p = 1$.

* If $p = k$, then $T(n) = \Theta(n^k \log n)$.
* If $p < k$, then $T(n) = \Theta(n^k)$.
  *(If $p > k$, the formula gives $T(n)=\Theta(n^p)$, which dominates $n^k$)*.

## 2D Recursion & Backtracking Summation

**Front:** How do you solve a recurrence in two variables like $T(n,k) = T(n/2, k) + T(n, k/4) + nk$ for asymptotic bounds?
**Back:**
Unfold the recursion tree. The cost at the root is `nk`. The children sum costs `(n/2)*k + n*(k/4) = (3/4)nk`. This forms a geometric series.

$T(n,k) = nk \left(1 + \frac{3}{4} + \left(\frac{3}{4}\right)^2 + \dots \right) = nk \cdot \frac{1}{1 - 3/4} = 4nk = \Theta(nk)$.

## Time Complexity from Recursive Calls

**Front:** In a recursive function `F(n, m)`, if the base case is `n <= 1 OR m <= 1`, what determines the recursion depth (height of the tree)?
**Back:**
The recursion stops when *either* variable hits its base constant. Therefore, the depth is determined by the variable that reaches its base *fastest*.
Depth $h = 1 + \min(h_n, h_m)$, where $h_n$ and $h_m$ are the individual recursion depths for `n` and `m` alone.

For `n <= 1 AND m <= 1`, the depth is $1 + \max(h_n, h_m)$.

## Linear Homogeneous Recurrence: Characteristic Equation

**Front:** For a linear homogeneous recurrence $T(n) = a_1 T(n-1) + a_2 T(n-2) + ... + a_k T(n-k)$, how do you find its characteristic equation?
**Back:**
Assume a solution of the form $T(n) = r^n$. Substitute into the recurrence (ignoring base cases):
$r^n = a_1 r^{n-1} + a_2 r^{n-2} + ... + a_k r^{n-k}$.
Divide both sides by $r^{n-k}$:
$r^k = a_1 r^{k-1} + a_2 r^{k-2} + ... + a_k$.
Rearrange to standard form:
$r^k - a_1 r^{k-1} - a_2 r^{k-2} - ... - a_k = 0$. This is the characteristic equation.

## Linear Recurrence: Homogeneous Solution

**Front:** What is the form of the homogeneous solution $T^{(h)}(n)$ for a linear recurrence, given the roots `r_i` of the characteristic equation?
**Back:**

* **Distinct Real Roots:** $T^{(h)}(n) = C_1 r_1^n + C_2 r_2^n + ... + C_k r_k^n$.
* **Repeated Real Root `r` of multiplicity `m`:** The root contributes terms: $r^n, n r^n, n^2 r^n, ..., n^{m-1} r^n$.
  E.g., for root `r` repeated 3 times: $T^{(h)}(n) = (C_1 + C_2 n + C_3 n^2) r^n$.

## Linear Recurrence: Particular Solution Guess

**Front:** How do you guess the form of the particular solution $T^{(p)}(n)$ for a linear recurrence with forcing function $f(n)$? Give an example for $f(n)=3n5^n$.
**Back:**
Guess a form similar to $f(n)$, but with unknown coefficients for a complete polynomial.
General form for $f(n) = (\text{polynomial in } n) \cdot A^n$ is $T^{(p)}(n) = (c_q n^q + ... + c_1 n + c_0) \cdot A^n$.

For $f(n)=3n5^n$: The polynomial part is `3n` (degree 1), and $A=5$. So guess:
$T^{(p)}(n) = (c_1 n + c_0) \cdot 5^n$.

## Linear Recurrence: Solving for Particular Coefficients

**Front:** After guessing $T^{(p)}(n)$ for a linear recurrence, how do you find its unknown coefficients?
**Back:**

1. Substitute $T^{(p)}(n)$ and its shifted versions (e.g., $T^{(p)}(n-1)$) into the original, full recurrence $T(n) = ... + f(n)$.
2. Simplify to get an equation of the form `(Polynomial in n) = f(n)`.
3. Equate the coefficients for each power of `n` on both sides to create a system of linear equations.
4. Solve the system for the unknown coefficients $c_0, c_1,...$.

## Linear Recurrence: General Solution & Initial Conditions

**Front:** What is the final, general solution to a linear recurrence? How are the constants determined?
**Back:**
The general solution is the sum of homogeneous and particular solutions:
$T(n) = T^{(h)}(n) + T^{(p)}(n)$.

The constants $C_1, C_2, ...$ in $T^{(h)}(n)$ are determined by using the *initial conditions* (e.g., $T(0), T(1), ... T(k-1)$) and solving the resulting system of linear equations.

## Classic Tower of Hanoi Recurrence

**Front:** What is the recurrence, initial condition, and closed-form solution for the minimum moves in the classic Tower of Hanoi problem with `n` disks?
**Back:**
Recurrence: $T(n) = 2T(n-1) + 1$, with $T(1) = 1$.
Solution: $T(n) = 2^n - 1$.
*(Derivation: Homogeneous: $C\cdot2^n$. Particular: constant $A=-1$. Apply $T(1)=1$ to find $C=1$.)*

## Restricted Tower of Hanoi Recurrence

**Front:** For a restricted Tower of Hanoi (where direct moves between the first and last peg are forbidden), what is the recurrence and its solution?
**Back:**
Recurrence: $T(n) = 3T(n-1) + 2$, with $T(1) = 2$.
Solution: $T(n) = 3^n - 1$.
*(Derivation: Homogeneous: $C\cdot3^n$. Particular: constant $A=-1$. Apply $T(1)=2$ to find $C=1$.)*

## Pitfalls: Master Theorem Case 3 Regularity Condition

**Front:** What is a common pitfall when applying Case 3 (root-heavy) of the Master Theorem?
**Back:**
Forgetting the *regularity condition*: $A f(n/b) \le c f(n)$ for some constant $c < 1$ and all sufficiently large `n`. This condition ensures the work at the root truly dominates. Some functions like $f(n)=n \log n$ may not satisfy it for certain `A` and `b`.

## Pitfalls: Guessing Particular Solutions with Overlap

**Front:** What special step is needed when guessing the particular solution $T^{(p)}(n)$ for a linear recurrence?
**Back:**
If your guessed form for $T^{(p)}(n)$ is already part of the homogeneous solution $T^{(h)}(n)$, you must multiply the *entire* guessed particular form by `n` (or `n^s` where `s` is the smallest integer to avoid overlap). This prevents the system of equations from becoming unsolvable.

## Special Consideration: Akra-Bazzi vs. Master Theorem

**Front:** When should you use the Akra-Bazzi theorem over the standard Master Theorem?
**Back:**
Use Akra-Bazzi when:

1. The recurrence has *multiple* recursive terms (e.g., $T(n)=T(n/3)+T(2n/3)+n$).
2. The size splits are *not* exactly equal fractions of `n` (e.g., $T(n)=T(\lfloor n/2 \rfloor)+T(\lceil n/2 \rceil)+n$).
3. The `f(n)` term is not a simple polynomial or polylog.
   The Master Theorem is a special case of Akra-Bazzi for a single recursive term with exact division.
