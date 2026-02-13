## 1. Loop Iteration Count - Linear Increment

**Front:** For a loop with initialization `i = a`, condition `i ≤ b` (or `i ≥ b`), and update `i = i + c` (or `i = i - c`), how many times does the loop execute? Assume step size c > 0.
**Back:**
For a loop incrementing by a constant `c`, the iteration count is the number of steps in an arithmetic progression from the start to the end value.
For `for(i = a; i ≤ b; i = i + c)`:

$$
\text{Iterations} = \left\lfloor \frac{b - a}{c} \right\rfloor + 1 \quad \text{or approximately} \quad \frac{b-a}{c} \text{ times for asymptotic analysis.}
$$

For a decrementing loop `for(i = a; i ≥ b; i = i - c)`, the formula is similar: $\frac{a-b}{c}$ times.

## 2. Loop Iteration Count - Geometric Increment

**Front:** For a loop with initialization `i = a`, condition `i ≤ b` (or `i ≥ b`), and update `i = i * c` (or `i = i / c`), where c > 1, how many times does the loop execute?
**Back:**
For a loop where the index multiplies or divides by a constant, the iteration count is the number of steps in a geometric progression, which is logarithmic.
For `for(i = a; i ≤ b; i = i * c)`:

$$
\text{Iterations} = \lfloor \log_c(b/a) \rfloor + 1 \quad \text{or asymptotically} \quad \Theta(\log_c(b/a)) \text{ times.}
$$

For `for(i = a; i ≥ b; i = i / c)`, it's asymptotically $\Theta(\log_c(a/b))$ times.

## 3. Basic Summation for Simple Nested Loops

**Front:** How do you express the time complexity of a simple nested loop like `for(i=1; i≤n; i++) { for(j=1; j≤n; j++) { O(1) work } }` using summation?
**Back:**
The inner loop runs `n` times for *each* iteration of the outer loop. The total work is the sum of the inner loop's iterations across all outer loop iterations.

$$
\sum_{i=1}^{n} \left( \sum_{j=1}^{n} 1 \right) = \sum_{i=1}^{n} n = n \cdot n = n^2
$$

Thus, the time complexity is $\Theta(n^2)$.

## 4. Summation for Dependent Inner Loop Bounds (Linear)

**Front:** How do you analyze `for(i=1; i≤n; i++) { for(j=1; j≤i; j++) { O(1) work } }`?
**Back:**
The bound of the inner loop (`j≤i`) depends on the current value of the outer loop index `i`.

$$
\text{Total Work} = \sum_{i=1}^{n} \left( \sum_{j=1}^{i} 1 \right) = \sum_{i=1}^{n} i = \frac{n(n+1)}{2} = \Theta(n^2)
$$

This is a common pattern resulting in quadratic time.

## 5. Summation for Dependent Inner Loop Step (j = j + i)

**Front:** How do you analyze `for(i=1; i≤n; i++) { for(j=1; j≤n; j = j + i) { O(1) work } }`?
**Back:**
The inner loop's step size depends on `i`. For a fixed `i`, the loop runs approximately `n/i` times.

$$
\text{Total Work} = \sum_{i=1}^{n} \left( \sum_{j=1, \text{ step } i}^{n} 1 \right) \approx \sum_{i=1}^{n} \frac{n}{i} = n \sum_{i=1}^{n} \frac{1}{i} = n \cdot H_n
$$

where $H_n$ is the n-th harmonic number, which is $\Theta(\log n)$. Thus, total complexity is $\Theta(n \log n)$.

## 6. Summation for Triple Nested Loops

**Front:** How do you analyze `for(i=1; i≤n; i++) { for(j=i; j≤n; j++) { for(k=1; k≤n^2; k++) { O(1) work } } }`?
**Back:**
Evaluate the summations from the innermost loop outward.

1. Innermost (k-loop): Independent of i and j. $\sum_{k=1}^{n^2} 1 = n^2$.
2. j-loop: For a given i, it runs from j=i to n. $\sum_{j=i}^{n} 1 = n - i + 1$.
3. i-loop: Sum over all i.

$$
\text{Total Work} = \sum_{i=1}^{n} \left( \sum_{j=i}^{n} \left( \sum_{k=1}^{n^2} 1 \right) \right) = \sum_{i=1}^{n} \left( (n - i + 1) \cdot n^2 \right) = n^2 \sum_{i=1}^{n} (n - i + 1)
$$

Let $m = n-i+1$, then $\sum_{m=1}^{n} m = n(n+1)/2$. Final complexity: $\Theta(n^4)$.

## 7. Complex Nested Summation Example

**Front:** Represent the work for `for(i=1; i≤n; i++) { for(j=1; j≤i; j++) { for(k=j; k≤ i+j; k++) { O(1) } } }` as a triple sum and simplify.
**Back:**
The sum representation is:

$$
\sum_{i=1}^{n} \sum_{j=1}^{i} \sum_{k=j}^{i+j} 1
$$

Simplify step-by-step:

1. Innermost sum: $\sum_{k=j}^{i+j} 1 = (i+j) - j + 1 = i + 1$.
2. Middle sum: $\sum_{j=1}^{i} (i + 1) = i(i + 1)$.
3. Outer sum: $\sum_{i=1}^{n} i(i+1) = \sum_{i=1}^{n} (i^2 + i) = \frac{n(n+1)(2n+1)}{6} + \frac{n(n+1)}{2}$.
   The dominant term is $\frac{n^3}{3}$, so complexity is $\Theta(n^3)$.

## 8. Loop with Modification of Loop Bound (n-- in outer loop)

**Front:** Analyze `for(i=1; i≤n; i++) { for(j=1; j≤n; j++) { O(1) } n--; }`. Does it run $\Theta(n^2)$?
**Back:**
Yes, it is still $\Theta(n^2)$. The outer loop runs approximately `n/2` times because `i` increments and `n` decrements each iteration, meeting around `n/2`. For iteration `t`, inner loop runs about `n - t` times.

$$
\text{Total Work} \approx \sum_{t=1}^{n/2} (n - t) = \Theta(n^2)
$$

The constant factor is halved, but the asymptotic growth remains quadratic.

## 9. Loop with Modification of Inner Loop Bound (n-- inside inner loop)

**Front:** Analyze `for(i=1; i≤n; i++) { for(j=1; j≤n; j++) { O(1); n--; } }`. What is the time complexity?
**Back:**
Complexity is $\Theta(n)$. The inner loop condition `j≤n` is checked each time, and `n` is decremented on each inner iteration. This causes `n` to shrink rapidly.
In the first outer iteration (i=1): The inner loop runs until `j > n`. Since `j++` and `n--` happen together, they meet when `j ≈ n/2`. So the first inner loop runs ~n/2 times and reduces `n` to ~n/2.
The process repeats with the halved `n`. Total work ~ n/2 + n/4 + n/8 + ... = $\Theta(n)$.

## 10. Independent vs. Dependent Nested Loops

**Front:** When analyzing nested loops, when can you simply multiply their individual complexities? Give an example.
**Back:**
You can multiply complexities only when the loops are **independent**—the bounds of the inner loop do not depend on the index of the outer loop, and the inner loop's execution does not alter variables affecting the outer loop's bounds.
**Example (Independent):**
`for(i=1; i≤n; i++) { for(j=1; j≤m; j++) { for(k=1; k≤p; k*=2) { O(1) } } }`
Here, `m` and `p` are constants relative to `i`. Complexity = $\Theta(n \cdot m \cdot \log p)$.
**Counter-Example (Dependent):** `for(i=1; i≤n; i++) { for(j=1; j≤i; j++) { ... } }` requires summation, not simple multiplication.

## Pitfalls and Special Considerations

**Front:** What are key pitfalls when analyzing loop complexity using summations?
**Back:**

1. **Off-by-one errors:** Using `≤` vs `<` changes the iteration count by 1. Use floor/ceiling carefully or rely on asymptotic analysis where constants drop out.
2. **Modifying loop variables:** Changing the loop counter or the bound variable (like `n--`) inside the loop drastically changes behavior. Trace the first few iterations.
3. **Assuming independence:** Do not multiply complexities if inner loop bounds depend on outer indices. Always write the summation to be sure.
4. **Ignoring the cost of the condition check:** The loop condition is evaluated each time. For simple conditions, this is O(1) and doesn't change asymptotic order.
5. **Approximation for multiple-choice:** When asked to select the closest complexity, approximate sums (e.g., $\sum_{i=1}^n 1/i \approx \log n$) and test with small values of `n` to rule out incorrect options.
