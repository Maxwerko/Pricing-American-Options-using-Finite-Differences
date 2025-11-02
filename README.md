# Finite difference solver to price American options

The script in `Solver.py` Implements both PSOR (from [[2]](#Tools)) and the operator splitting method presented in [[1]](#Ikonen). Note that in the first line of equation (10) the signs in front of $\lambda$ should be the opposite when implementing the operator splitting method (it’s a typo in the article [[1]](#Ikonen)).

## Summary
The finite difference approach reformulates the American option pricing as a Linear Complementarity Problem (LCP), avoiding explicit tracking of the early-exercise boundary $S_f(t)$ [[2]](#Tools). Using the transformations $S = Ke^x$ and $\tau$-time ($t = T - 2\tau/\sigma^2$), the value function becomes $V(S,t) = K \exp\{-\frac{1}{2}(q_\delta -1)x -(\frac{1}{4}(q_\delta -1)^2 + q)\tau\} y(x,\tau)$, where $q = \frac{2r}{\sigma^2}$ and $q_\delta= \frac{2(r-\delta)}{\sigma^2}$. The problem requires finding $y$ such that:

$$
%\label{eq:y_conditions}
\left(\frac{\partial y}{\partial \tau} - \frac{\partial^2y}{\partial x^2}\right)(y-g) = 0, \quad \left(\frac{\partial y}{\partial \tau} - \frac{\partial^2y}{\partial x^2}\right) \geq 0, \quad (y-g) \geq 0
$$

where $g(x,\tau)$ determines the initial condition (in terms of $\tau$) and is used to calculate the vector $b^{(v)}$ which incorporates boundary conditions at each time step, $v$, which is used to denote the discrete time $\tau_v$. After discretization with spatial step $\Delta x$ and time step $\Delta \tau$, the problem becomes: find $w$ satisfying
$$
\begin{equation}
%\label{eq:w_conditions}
Aw-b^{(v)} \geq 0, \quad w\geq g^{(v+1)}, \quad (Aw -b^{(v)})^T(w-g^{(v+1)}) = 0 
\end{equation}
$$
where $A$ is a tridiagonal matrix with entries determined by $\lambda = \frac{\Delta \tau}{\Delta x^2}$ and $\theta$ (the time discretization parameter: $\theta = 0$ for explicit, $\theta = 1$ for implicit, $\theta = 1/2$ for Crank-Nicolson). Now the solution vector $w$ in equation (1) denotes an approximation to $y$ at time $\tau_v$.

### Projected SOR (PSOR)
PSOR solves the LCP iteratively using the transformation $\mathrm{x} = w-g$. At iteration $k$, the update is $x_i^{(k)} = \max\{0, x_i^{(k-1)} + \omega_R \frac{r_i^{(k)}}{a_{ii}}\}$ where
$$
r_i^{(k)} = \hat{b}_i - \sum_{j=1}^{i-1}a_{ij}\mathrm{x}_j^{(k)} - a_{ii}\mathrm{x}_i^{(k-1)} - \sum_{j=i+1}^{n}a_{ij}\mathrm{x}_j^{(k-1)}
$$
with $\hat{b} = b -Ag$ and $\omega_R$ the relaxation parameter chosen to improve convergence. The projection $\max\{0, \cdot\}$ enforces $\mathrm{x} \geq 0$.

### Operator Splitting
The operator splitting method in [[1]](#Ikonen) decouples the PDE solve from constraint enforcement using an auxiliary variable $\eta$. Each time step consists of two sub-steps:

The first step is to solve equation (2) for $\hat{V}^{(v)}$ using LU decomposition.
$$
\begin{equation}
%\label{eq:OS1}
    \frac{1}{\Delta t} \left(V^{(v+1)}-\hat{V}^{(v)} \right) + A\left((1-\theta)V^{(v+1)} + \theta \hat{V}^{(v)}\right) - \eta^{(v+1)} = 0 
\end{equation}
$$

The second step is to project component-wise to enforce the constraints: 
$$
\begin{align}
%\label{eq:OS}

    \frac{1}{\Delta t} \left(\hat{V}^{(v)} - V^{(v)}\right) - \eta^{(v+1)} + \eta^{(v)} &= 0 \\
     \left[V_i^{(v)} - \Psi(S_i)\right] \eta_i^{(v)} = 0, \quad V_i^{(v)} \geq \Psi(S_i), &\quad \eta_i^{(v)} \geq 0
\end{align}
$$
solving for the component pairs $V_i^{(v)}$ and $\eta_i^{(v)}$. This method is significantly more efficient than PSOR while it maintains comparable accuracy [[1]](#Ikonen).

## Sources

<a id="Ikonen"></a>
[[1]](#Ikonen) S. IKONEN AND J. TOIVANEN: Operator Splitting Methods for American Option Pricing, Applied Mathematics Letters, Volume 17, Issue 7, 2004, pp. 809–814. https://doi.org/10.1016/j.aml.2004.06.010 

<a id="Tools"></a>
[[2]](#Tools) Rüdiger Seydel. Tools for Computational Finance. Springer
Nature, 01 2017. ISBN 978-1-4471-7337-3. https://doi.org/10.1007/978-1-4471-7338-0

