"""
=====================================================
Finite Difference solver to price American options
=====================================================

This script Implements both PSOR and the operator splitting method presented 
in the article and compare them. Note that in the first line of equation (10)
the signs in front of λ should be the opposite when implementing the operator 
splitting method (it's a typo in the article). 
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from tqdm import tqdm


class AmericanOptionSolver:
    def __init__(self, K, T, r, sigma, Smax, M, N, delta=0, theta=0.5, call=True, method='PSOR'):
        self.K = K         # Strike price
        self.T = T         # Time to maturity
        self.r = r         # Risk-free interest rate
        self.sigma = sigma # Volatility
        self.Smax = Smax   # Maximum stock price considered
        self.M = M         # Number of price steps
        self.N = N         # Number of time steps
        self.delta = delta # Dividend yield Only for PSOR method

        self.call = call   # Call or put
        self.method = method

        if method == 'PSOR':
            self.q_delta = 2*(r-delta)/sigma**2
            self.q = 2*r/sigma**2
            self.dtau = sigma**2 * T/N  
            self.theta = theta # Theta for Implicit-Explicit scheme

            # Initialization
            #self.sol_vec = np.zeros(M+1) # Solution vector
            self.x_max = np.log(Smax/K)
            self.x_min = - 4
            self.x = np.linspace(self.x_min, self.x_max, M+2)
            self.dx = self.x[1] - self.x[0]
            
            self.lambda_ = self.dtau/self.dx**2

        elif method == 'OperatorSplitting':
            # For operator splitting method, we use a different transformation
            self.S = np.linspace(0, Smax, M)
            self.dt = T / N
            self.v0 = self.payoff(self.S)
        else:
            raise NotImplementedError("Only availabe methods are 'PSOR' or 'OperatorSplitting' as of now.")


    
    def g(self, x, tau):
        if self.call:
            res = np.exp(tau * 0.25 *((self.q_delta -1)**2 + 4*self.q)) * np.maximum(np.e**(x* 0.5 *(self.q_delta + 1)) - np.e**(x* 0.5 *(self.q_delta - 1)),0)
        else:
            res = np.exp(tau * 0.25 *((self.q_delta -1)**2 + 4*self.q)) * np.maximum(np.e**(x* 0.5 *(self.q_delta - 1)) - np.e**(x* 0.5 *(self.q_delta + 1)),0)
        return res

    def setup_coefficients(self):
        if self.method == 'PSOR':
            M, x = self.M, self.x
            # terminal condition for American put option
            self.sol_vec = self.g(x[1:-1], 0)
            # Construct A and B matrices (B calculates b_i in the report)
            Middel_diag = np.ones(M) * (1+2*self.lambda_ * self.theta)
            Lower_Higher_diag = np.ones(M-1) * (-self.lambda_ * self.theta)
            A = diags([Lower_Higher_diag, Middel_diag, Lower_Higher_diag], [-1, 0, 1]).tocsc()
            #print(f"A matrix shape: {A.shape}")
            Middel_diag_B = np.ones(M) * (1-2*self.lambda_ * (1 - self.theta))
            Lower_Higher_diag_B = np.ones(M-1) * (self.lambda_ * (1 - self.theta))
            B = diags([Lower_Higher_diag_B, Middel_diag_B, Lower_Higher_diag_B], [-1, 0, 1]).tocsc()
            #print(f"B matrix shape: {B.shape}")
            return A, B
        else:
            M = self.M
            sigma = self.sigma
            r = self.r

            # Coefficients for the tridiagonal matrix A
            a = 0.5 * (sigma**2 * np.arange(M+1)**2 - r * np.arange(M+1))
            b = - (sigma**2 * np.arange(M+1)**2 + r)
            c = 0.5 * (sigma**2 * np.arange(M+1)**2 + r * np.arange(M+1))

            # Construct the sparse tridiagonal matrix A
            #print(len(a[2:]), len(b[1:]), len(c[:-2]))
            diagonals = [a[2:], b[1:], c[:-2]]
            A = diags(diagonals, [-1, 0, 1]).tocsc()
            #print(f"A matrix shape: {A.shape}")

            return A
    
    # Efficient version for tridiagonal matrices
    def psor_tridiagonal(self, A, b_hat, x, g, omega_R=1.5, max_iter=1000, tol=1e-7):
        """
        Optimized PSOR for tridiagonal A, using x = w - g transformation
        """
        
        # extratract diagonals
        a_diag = A.diagonal(0)  # Main diagonal
        a_lower = A.diagonal(-1)  # Lower diagonal
        a_upper = A.diagonal(1)  # Upper diagonal

        x_old = x.copy()
        n = len(x)
        for k in range(max_iter):
            for i in range(n):
                # Residual for tridiagonal matrix
                r_i = b_hat[i] - a_diag[i] * x_old[i]
                
                if i > 0:
                    r_i -= a_lower[i-1] * x[i-1]  # Updated value
                
                if i < n - 1:
                    r_i -= a_upper[i] * x_old[i+1]  # Old value
                
                # Projection: x >= 0
                x[i] = max(0, x_old[i] + omega_R * r_i / a_diag[i])
            
            # Check convergence
            if np.linalg.norm(x - x_old, np.inf) < tol:
                break
            
            x_old[:] = x
        
        # Transform back to w
        w = x + g
        
        return w
        
    
    def psor_solver(self, A, B, omega=1.3, tol=1e-7, max_iter=10000):
        M, N = self.M, self.N
        # initialize solution vector
        w = self.sol_vec.copy()
        alpha = self.lambda_ * self.theta
        beta = (self.lambda_ * (1-self.theta))

        for v in tqdm(range(N), desc="PSOR Time Steps", unit="step"):
            tau_v = v * self.dtau
            b = B @ w
            # Incorporate boundary conditions
            b[0] +=  beta * self.g(self.x_min, tau_v) + alpha * self.g(self.x_min, tau_v + self.dtau)
            b[-1] += beta * self.g(self.x_max, tau_v) + alpha * self.g(self.x_max, tau_v + self.dtau)

            # Solve using psor the Aw = b componentwise so that w >= g is obeyed
            g_tauv = self.g(self.x[1:-1], tau_v + self.dtau)
            x = w - g_tauv
            b_hat = b - A @ g_tauv
            w_new = self.psor_tridiagonal(A, b_hat, x, g_tauv, omega_R=omega, max_iter=max_iter, tol=tol)

            w = w_new # Update for next time step

        self.sol_vec = w

        # Get option price for all S at t=0
        S_vec = self.K * np.exp(self.x[1:-1])
        self.S = S_vec
        price = self.K * self.sol_vec * np.exp(-self.x[1:-1]* 0.5 *(self.q_delta - 1)) * np.exp(-self.N * self.dtau * (0.25*(self.q_delta - 1)**2 + self.q))
        
        # Test for early exercise
        eps = self.K * 1e-5
        if self.call:
            early_ex = np.abs(self.K-S_vec + price)
            i_f = np.argmin(early_ex)
        else:
            early_ex = np.abs(price + S_vec - self.K)
            i_f = np.argmax(early_ex)
        
        if early_ex[i_f] < eps:
            return price, S_vec[i_f]
        else:
            return price, None
        
        
    def payoff(self, S):
        if self.call:
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def operator_splitting_solver(self, A):
        """
        Operator splitting for Crank-Nicolson scheme for American options
        
        Parameters:
        -----------
        A : ndarray, shape (M, M)
            Black-Scholes matrix
            
        Returns:
        --------
        v : ndarray, shape (n,)
            Option values at t=0
        """
        M, N = self.M, self.N
        v = self.v0.copy()
        eta_v = np.zeros(M)  # Initialize auxiliary variable
        # Create a sparse identity matrix
        I_sparse = csc_matrix(np.eye(M))  # Create a sparse identity matrix
        
        # Backward in time
        for _ in tqdm(range(N), desc="Operator Splitting Time Steps", unit="step"):
            # Solve for intermediate solution
            # Equation  (A - I/dt) v_tilde = eta - (I/dt + A) v 
            
            # Left-hand side matrix: (I/dt - A)
            LHS = A - I_sparse / self.dt  
            
            # Right-hand side: eta - (I/dt + A) v^k 
            # Note: eta^{k+1} = eta^k for the first iteration
            RHS = (eta_v - (I_sparse / self.dt + 0.5 * A) @ v)

            LHS_csc = csc_matrix(LHS)
            lu = splu(LHS_csc)
            v_tilde = lu.solve(RHS)
            eta_tilde = eta_v.copy()
            # Compute payoff
            payoff = self.payoff(self.S)
            
            # Projection step (equation 10)
            # This is solved component-wise
            
            v_new = np.maximum(v_tilde, payoff) # Enforce constraint: v >= payoff
            eta_new = eta_tilde + (v_tilde - v_new) / self.dt 

            # Ensure eta_new is non-negative
            if np.any(eta_new < 0):
                eta_new = np.maximum(eta_new, 0)
            
            # Update for next iteration
            v = v_new
            eta_v = eta_new
        
        return v
    


# Example usage
if __name__ == "__main__":
    K = 10       # Strike price
    T = 1      # Time to maturity
    r = 0.06     # Risk-free interest rate
    sigma = 0.3  # Volatility
    Smax = K*5  # Maximum stock price considered
    M = 500       # Number of price steps
    N = 2000     # Number of time steps
    theta = 0.5    # Theta for Implicit-Explicit scheme
    call = True   # Call option
    if call:
        print("Pricing American Call Option")
    else:        
        print("Pricing American Put Option")
    from time import time

    solver = AmericanOptionSolver(K, T, r, sigma, Smax, M, N, theta=theta, call=call, method='PSOR')
    A, B = solver.setup_coefficients()
    start_time = time()
    # PSOR method
    price, stopping_criteria = solver.psor_solver(A, B)
    end_time = time()
    PSOR_time = end_time - start_time
    # plot the price vector as a function of S
    plt.figure(figsize=(10,6))
    plt.plot(solver.S, price, label='PSOR')


    if stopping_criteria is not None:
        print(f"Early exercise boundary at S = {stopping_criteria:.4f}")

    # Operator Splitting method
    solver_os = AmericanOptionSolver(K, T, r, sigma, Smax, M, N, call=call, method='OperatorSplitting')
    A_os = solver_os.setup_coefficients()   
    start_time = time()
    price_os = solver_os.operator_splitting_solver(A_os)
    end_time = time()
    OS_time = end_time - start_time

    # plot the price vector as a function of S
    plt.plot(solver_os.S, price_os, label='Operator Splitting')
    plt.xlabel('Stock Price S')
    plt.ylabel('Option Price')
    if stopping_criteria is not None:
       plt.axvline(x=stopping_criteria, color='r', linestyle='--', label='Early Exercise Boundary for PSOR')

    if call:
        plt.title('American Call Price using OSP and PSOR Methods')
    else:
        plt.title('American Put Price using OSP and PSOR Methods')

    plt.axvline(x=K, color='g', linestyle='--', label='Strike Price K')
    plt.legend()
    plt.grid()
    if call:
        plt.savefig('American_call.png', dpi=300)
    else:
        plt.savefig('American_put.png', dpi=300)
    print("================= Solver Performance ===================")
    print(f"PSOR Time taken:               {PSOR_time:.4f} seconds")
    print(f"Operator Splitting Time taken: {OS_time:.4f} seconds")
    print("========================================================")
    plt.show()