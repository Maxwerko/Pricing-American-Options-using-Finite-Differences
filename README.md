# Finite difference solver to price American options

The script in [Solver.py](Solver.py) Implements both PSOR (from [[2]](#Tools)) and the operator splitting method presented in [[1]](#Ikonen). Note that in the first line of equation (10) the signs in front of $\lambda$ should be the opposite when implementing the operator splitting method (it’s a typo in [[1]](#Ikonen)).

The [Convergence_test.py](Convergence_test.py) script runs convergence tests for both methods by comparing coarse-grid solutions against a fine-grid reference.

Expected plots: In [Convergence_M.png](Convergence_M.png), for a smooth solution you expect $\|e\| \sim C M^{-2}$ (slope about $-2$ in log-log) from the second-order spatial stencil, although the free boundary can reduce this toward $M^{-1}$. In [Convergence_N.png](Convergence_N.png), with $\theta=1/2$ (Crank-Nicolson) you expect $\|e\| \sim C N^{-2}$ (slope about $-2$), while $\theta=1$ would give $N^{-1}$; the obstacle can again reduce the observed slope and PSOR can hit a tolerance floor.

## Summary
The finite difference approach reformulates the American option pricing as a Linear Complementarity Problem (LCP), avoiding explicit tracking of the early-exercise boundary $S_f(t)$ [[2]](#Tools). Using the transformations $S = Ke^x$ and $\tau$-time ($t = T - 2\tau/\sigma^2$), the value function becomes $V(S,t) = K \exp\{-\frac{1}{2}(q_\delta -1)x -(\frac{1}{4}(q_\delta -1)^2 + q)\tau\} y(x,\tau)$, where $q = \frac{2r}{\sigma^2}$ and $q_\delta= \frac{2(r-\delta)}{\sigma^2}$. The problem requires finding $y$ such that:

$$
%\label{eq:y_conditions}
\left(\frac{\partial y}{\partial \tau} - \frac{\partial^2y}{\partial x^2}\right)(y-g) = 0, \quad \left(\frac{\partial y}{\partial \tau} - \frac{\partial^2y}{\partial x^2}\right) \geq 0, \quad (y-g) \geq 0
$$

where $g(x,\tau)$ determines the initial condition (in terms of $\tau$) and is used to calculate the vector $b^{(v)}$ which incorporates boundary conditions at each time step, $v$, which is used to denote the discrete time $\tau_v$. After discretization with spatial step $\Delta x$ and time step $\Delta \tau$, the problem becomes: find $w$ satisfying

$$
Aw-b^{(v)} \geq 0, \quad w\geq g^{(v+1)}, \quad (Aw -b^{(v)})^T(w-g^{(v+1)}) = 0 \quad (1)
$$

where $A$ is a tridiagonal matrix with entries determined by $\lambda = \frac{\Delta \tau}{\Delta x^2}$ and $\theta$ (the time discretization parameter: $\theta = 0$ for explicit, $\theta = 1$ for implicit, $\theta = 1/2$ for Crank-Nicolson). Now the solution vector $w$ in equation (1) denotes an approximation to $y$ at time $\tau_v$.

### Projected SOR (PSOR)
PSOR solves the LCP iteratively using the transformation $\mathrm{x} = w-g$. At iteration $k$, the update is 

$$
\mathrm{x}_{i}^{(k)} = \max \left[0,\mathrm{x}_{i}^{(k-1)} + \omega_R \frac{r_i^{(k)}}{a_{ii}} \right]
$$ 

where

$$
r_i^{(k)} = \hat{b}_i - \sum_{j=1}^{i-1}a_{ij}\mathrm{x}_j^{(k)} - a_{ii}\mathrm{x}_i^{(k-1)} - \sum_{j=i+1}^{n}a_{ij}\mathrm{x}_j^{(k-1)}
$$

with $\hat{b} = b - Ag$ and $\omega_R$ the relaxation parameter chosen to improve convergence. The projection $\max[0, \cdot]$ enforces $\mathrm{x} \geq 0$.

### Operator Splitting
The operator splitting method in [[1]](#Ikonen) decouples the PDE solve from constraint enforcement using an auxiliary variable $\eta$. Each time step consists of two sub-steps:

The first step is to solve equation (2) for $\hat{V}^{(v)}$ using LU decomposition.

$$
\frac{1}{\Delta t} \left(V^{(v+1)}-\hat{V}^{(v)} \right) + A\left((1-\theta)V^{(v+1)} + \theta \hat{V}^{(v)}\right) - \eta^{(v+1)} = 0 \quad (2)
$$

The second step is to project component-wise to enforce the constraints: 

$$
\frac{1}{\Delta t} \left(\hat{V}^{(v)} - V^{(v)}\right) - \eta^{(v+1)} + \eta^{(v)} = 0 
$$
$$
\left[V_i^{(v)} - \Psi(S_i)\right] \eta_i^{(v)} = 0, \quad V_i^{(v)} \geq \Psi(S_i), \quad \eta_i^{(v)} \geq 0
$$

solving for the component pairs $V_i^{(v)}$ and $\eta_i^{(v)}$. This method is significantly more efficient than PSOR while it maintains comparable accuracy [[1]](#Ikonen).


**Note:** The call option appears to behave correctly, but the put option does not. In [American_put.png](American_put.png), the PSOR and operator splitting curves diverge near the strike price K. The cause is unclear. It could be an implementation issue or a mismatch related to the PSOR log-price transformation, although the call case aligns well in [American_call.png](American_call.png).

## Sources

<a id="Ikonen"></a>
[[1]](#Ikonen) S. IKONEN AND J. TOIVANEN: Operator Splitting Methods for American Option Pricing, Applied Mathematics Letters, Volume 17, Issue 7, 2004, pp. 809–814. https://doi.org/10.1016/j.aml.2004.06.010 

<a id="Tools"></a>
[[2]](#Tools) Rüdiger Seydel. Tools for Computational Finance. Springer
Nature, 01 2017. ISBN 978-1-4471-7337-3. https://doi.org/10.1007/978-1-4471-7338-0

