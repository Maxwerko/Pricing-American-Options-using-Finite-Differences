# Finite difference solver to price American options

The script in `Solver.py` Implements both PSOR and the operator splitting method presented in the [1]. Note that in the first line of equation (10) the signs in front of $\lambda$ should be the opposite when implementing the operator splitting method
(it’s a typo in the article).



## Sources

[1] S. IKONEN AND J. TOIVANEN: Operator Splitting Methods for American Option
Pricing, Applied Mathematics Letters, Volume 17, Issue 7, 2004, pp. 809–814.

[2] Rüdiger Seydel. Tools for Computational Finance. Springer
Nature, 01 2017. ISBN 978-1-4471-7337-3. doi:
10.1007/978-1-4471-7338-0.