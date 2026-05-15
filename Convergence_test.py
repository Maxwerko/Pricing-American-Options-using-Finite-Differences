"""
=============================================================================
This script contains convergence tests for the numerical methods implemented.
=============================================================================
"""

import os
import pickle
import hashlib
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from Solver import AmericanOptionSolver


def disk_memoize(cache_dir="cache"):
    """
    Decorator for caching function outputs on disk.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optionally force a fresh computation (ignores cache if True)
            force = kwargs.pop("force_recompute", False)

            # Make sure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Build a unique hash key from the function name and arguments
            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            # Load the cached result if it exists (and recomputation is not forced)
            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # Otherwise: compute the result, then cache it to disk
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result
        
        return wrapper
    return decorator

# Coarsening factors used to define nested grids from the fine grid.
COARSENING_FACTORS = [10, 12, 15, 20, 30, 40, 60]


def _nested_time_values(N_fine, factors, max_fraction, min_time_steps=1):
    max_N = int(np.floor(max_fraction * N_fine))
    values = []
    for factor in factors:
        if N_fine % factor != 0:
            continue
        N = N_fine // factor
        if N <= max_N and N >= min_time_steps:
            values.append(N)
    values = sorted(set(values))
    if not values:
        raise ValueError("No nested N values found. Adjust N_fine or coarsening factors.")
    return values


def _nested_m_values_psor(M_fine, factors, max_fraction):
    interval_count = M_fine + 1
    max_M = int(np.floor(max_fraction * M_fine))
    values = []
    for factor in factors:
        if interval_count % factor != 0:
            continue
        M = interval_count // factor - 1
        if M <= max_M:
            values.append(M)
    values = sorted(set(values))
    if not values:
        raise ValueError("No nested PSOR M values found. Adjust M_fine or coarsening factors.")
    return values


def _nested_m_values_os(M_fine, factors, max_fraction):
    interval_count = M_fine - 1
    max_M = int(np.floor(max_fraction * M_fine))
    values = []
    for factor in factors:
        if interval_count % factor != 0:
            continue
        M = interval_count // factor + 1
        if M <= max_M:
            values.append(M)
    values = sorted(set(values))
    if not values:
        raise ValueError("No nested OS M values found. Adjust M_fine or coarsening factors.")
    return values


def _psor_fine_indices(M_fine, M_coarse):
    interval_fine = M_fine + 1
    interval_coarse = M_coarse + 1
    if interval_fine % interval_coarse != 0:
        raise ValueError("PSOR grids are not nested.")
    step = interval_fine // interval_coarse
    return np.arange(1, M_coarse + 1) * step - 1


def _os_fine_indices(M_fine, M_coarse):
    interval_fine = M_fine - 1
    interval_coarse = M_coarse - 1
    if interval_fine % interval_coarse != 0:
        raise ValueError("OS grids are not nested.")
    step = interval_fine // interval_coarse
    return np.arange(0, M_coarse) * step


@disk_memoize()
def get_true_approx(
    Smax,
    K,
    r,
    delta,
    sigma,
    T,
    call,
    theta,
    M_fine_psor=5039,
    M_fine_os=5041,
    N_fine=10080,
    psor_tol=1e-7,
):
    """Compute fine-grid solutions for both methods for convergence testing."""
    fine_solver_OS = AmericanOptionSolver(Smax=Smax, K=K, r=r, sigma=sigma, theta=theta, T=T, M=M_fine_os, N=N_fine, delta=delta, call=call, method='OperatorSplitting')
    A_OS = fine_solver_OS.setup_coefficients()
    price_OS = fine_solver_OS.operator_splitting_solver(A_OS)

    fine_solver_PSOR = AmericanOptionSolver(Smax=Smax, K=K, r=r, sigma=sigma, theta=theta, T=T, M=M_fine_psor, N=N_fine, delta=delta, call=call, method='PSOR')
    A, B = fine_solver_PSOR.setup_coefficients()
    price_PSOR, __ = fine_solver_PSOR.psor_solver(A, B, tol=psor_tol)

    return {
        'S_PSOR': fine_solver_PSOR.S,
        'Price_PSOR': price_PSOR,
        'S_OS': fine_solver_OS.S,
        'Price_OS': price_OS,
    }
    
@disk_memoize()
def test_convergence(M_fine_psor, M_fine_os, N_fine, true_approx, Smax, K, r, delta, sigma, T, call, theta, max_fraction=0.1, coarsening_factors=None, time_coarsening_factors=None, min_time_steps=1, psor_tol=1e-7):
    """
    Run convergence tests for PSOR and Operator Splitting methods.
    
    The problem we run into here is that we want to compare the numerical
    solutions at different grid resolutions to a "true" solution computed on a fine grid.
    Thus we need the less fine grids to align with the fine grid points.
    1. We first compute the "true" solution on a very fine grid and save it to disk.
    2. For each coarser grid, we use only those grid points that align with the fine grid
       to compute the error norms.
    3. We compute and store the L-2 and L-infinity error norms for each method and grid size.
    4. Finally, we plot the error norms against the grid sizes on a log-log scale to visualize convergence.

    """
    methods = ['PSOR', 'OperatorSplitting']
    errors_M = {method: {'L2': [], 'L_inf': []} for method in methods}
    errors_N = {method: {'L2': [], 'L_inf': []} for method in methods}

    if coarsening_factors is None:
        coarsening_factors = COARSENING_FACTORS
    if time_coarsening_factors is None:
        time_coarsening_factors = coarsening_factors

    M_values = {
        'PSOR': _nested_m_values_psor(M_fine_psor, coarsening_factors, max_fraction),
        'OperatorSplitting': _nested_m_values_os(M_fine_os, coarsening_factors, max_fraction),
    }
    N_values = _nested_time_values(N_fine, time_coarsening_factors, max_fraction, min_time_steps=min_time_steps)

    print(f"Testing convergence for PSOR M values: {M_values['PSOR']}")
    print(f"Testing convergence for OperatorSplitting M values: {M_values['OperatorSplitting']}")
    print(f"Testing convergence for N values: {N_values}")
    # Load the true solution

    Price_fine_PSOR = np.asarray(true_approx['Price_PSOR'])
    Price_fine_OS = np.asarray(true_approx['Price_OS'])

    # Time convergence: vary N, keep M fixed
    for method in methods:
        L2_norms = []
        L_inf_norms = []
        M_fine_method = M_fine_psor if method == 'PSOR' else M_fine_os
        fine_price = Price_fine_PSOR if method == 'PSOR' else Price_fine_OS
        for N in N_values:
            solver = AmericanOptionSolver(Smax=Smax, K=K, r=r, sigma=sigma, theta=theta, T=T, M=M_fine_method, N=N, delta=delta, call=call, method=method)
            AB = solver.setup_coefficients()

            if method == 'PSOR':
                A, B = AB
                price, __ = solver.psor_solver(A, B, tol=psor_tol)
            else:
                A = AB
                price = solver.operator_splitting_solver(A)

            if price.shape != fine_price.shape:
                raise ValueError("Time convergence requires identical spatial grids.")

            # Compute error norms
            error = price - fine_price
            L2_norm = np.sqrt(np.sum(error**2) / len(error))
            L_inf_norm = np.max(np.abs(error))

            L2_norms.append(L2_norm)
            L_inf_norms.append(L_inf_norm)

        errors_N[method]['L2'] = L2_norms
        errors_N[method]['L_inf'] = L_inf_norms

    # Spatial convergence: vary M, keep N fixed
    for method in methods:
        L2_norms = []
        L_inf_norms = []
        M_fine_method = M_fine_psor if method == 'PSOR' else M_fine_os
        fine_price = Price_fine_PSOR if method == 'PSOR' else Price_fine_OS
        for M in M_values[method]:
            solver = AmericanOptionSolver(Smax=Smax, K=K, r=r, sigma=sigma, theta=theta, T=T, M=M, N=N_fine, delta=delta, call=call, method=method)
            AB = solver.setup_coefficients()
            
            if method == 'PSOR':
                A, B = AB
                price, __ = solver.psor_solver(A, B, tol=psor_tol)
                idx = _psor_fine_indices(M_fine_method, M)
                fine_price_sample = fine_price[idx]
            else:
                A = AB
                price = solver.operator_splitting_solver(A)
                idx = _os_fine_indices(M_fine_method, M)
                fine_price_sample = fine_price[idx]

            # Compute error norms
            error = price - fine_price_sample
            L2_norm = np.sqrt(np.sum(error**2) / len(error))
            L_inf_norm = np.max(np.abs(error))

            L2_norms.append(L2_norm)
            L_inf_norms.append(L_inf_norm)

        errors_M[method]['L2'] = L2_norms
        errors_M[method]['L_inf'] = L_inf_norms

    return errors_M, errors_N, M_values, N_values

def plot_convergence(errors_M, errors_N, M_values, N_values):
    def add_reference_slope(ax, x_values, y_values, slope, label, anchor_index=0):
        if len(x_values) == 0 or len(y_values) == 0:
            return
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        if anchor_index < 0 or anchor_index >= len(x):
            anchor_index = 0
        x0 = x[anchor_index]
        y0 = y[anchor_index]
        y_ref = y0 * (x / x0) ** slope
        ax.loglog(x, y_ref, linestyle=":", color="red", linewidth=1.5, label=label)

    method_colors = {
        'PSOR': 'tab:blue',
        'OperatorSplitting': 'tab:orange',
    }
    norm_styles = {
        'L2': '-',
        'L_inf': '--',
    }
    norm_markers = {
        'L2': 'o',
        'L_inf': 'x',
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Spatial convergence (M)
    ax = axes[0]
    for method in errors_M:
        color = method_colors.get(method, 'tab:gray')
        for norm_key in ('L2', 'L_inf'):
            ax.loglog(
                M_values[method],
                errors_M[method][norm_key],
                color=color,
                linestyle=norm_styles[norm_key],
                marker=norm_markers[norm_key],
                markersize=4,
            )
    ref_method = 'PSOR' if 'PSOR' in errors_M else next(iter(errors_M))
    add_reference_slope(ax, M_values[ref_method], errors_M[ref_method]['L2'], slope=-2, label='M^-2 reference')
    ax.set_xlabel('Number of Spatial Steps (M)')
    ax.set_ylabel('Error')
    ax.set_title('Spatial Convergence')

    # Time convergence (N)
    ax = axes[1]
    for method in errors_N:
        color = method_colors.get(method, 'tab:gray')
        for norm_key in ('L2', 'L_inf'):
            ax.loglog(
                N_values,
                errors_N[method][norm_key],
                color=color,
                linestyle=norm_styles[norm_key],
                marker=norm_markers[norm_key],
                markersize=4,
            )
    ref_method = 'PSOR' 
    add_reference_slope(ax, N_values, errors_N[ref_method]['L2'], slope=-2, label='N^-2 reference (CN)')
    add_reference_slope(ax, N_values, errors_N[ref_method]['L_inf'], slope=-2, label='N^-2 reference (CN)')
    ax.set_xlabel('Number of Time Steps (N)')
    ax.set_title('Time Convergence')

    for ax in axes:
        ax.grid(True, which="both", ls="--")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color=method_colors['PSOR'], linestyle='-', label='PSOR'),
        Line2D([0], [0], color=method_colors['OperatorSplitting'], linestyle='-', label='OperatorSplitting'),
        Line2D([0], [0], color='red', linestyle=':', label='(M or N) $^{-2}$ reference'),
    ]
    norm_handles = [
        Line2D([0], [0], color='black', linestyle=norm_styles['L2'], marker=norm_markers['L2'], label='L2'),
        Line2D([0], [0], color='black', linestyle=norm_styles['L_inf'], marker=norm_markers['L_inf'], label='$L_{\\infty}$'),
    ]

    fig.legend(handles=method_handles, loc='upper center', bbox_to_anchor=(0.29, 0.92), ncol=3, frameon=True)
    fig.legend(handles=norm_handles, loc='upper center', bbox_to_anchor=(0.76, 0.92), ncol=2, frameon=True)
    fig.suptitle('Convergence Test for American Option Pricing Methods', y=0.98)
    #fig.subplots_adjust(top=0.8, wspace=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig('convergence_M_N.png', dpi=300)
    plt.show()



def main():
    K = 10              # Strike price
    T = 1               # Time to maturity
    r = 0.06            # Risk-free interest rate
    sigma = 0.3         # Volatility
    Smax = K*5          # Maximum stock price considered
    theta = 0.5         # Theta for Crank-Nicolson scheme
    delta = 0.0         # Dividend yield 
    call = True         # Call option
    
    M_fine_psor = 5039
    M_fine_os = 5041
    N_fine = 10080
    psor_tol = 1e-9
    time_coarsening_factors = COARSENING_FACTORS + [80, 120, 140, 168, 210, 240, 280, 336, 420, 560, 840]
    spatial_coarsening_factors = time_coarsening_factors

    fine_results = get_true_approx(
        Smax,
        K,
        r,
        delta,
        sigma,
        T,
        call,
        theta,
        M_fine_psor=M_fine_psor,
        M_fine_os=M_fine_os,
        N_fine=N_fine,
        psor_tol=psor_tol,
        force_recompute=False,
    )

    errors_M, errors_N, M_values, N_values = test_convergence(
        M_fine_psor,
        M_fine_os,
        N_fine,
        fine_results,
        Smax,
        K,
        r,
        delta,
        sigma,
        T,
        call,
        theta,
        coarsening_factors=spatial_coarsening_factors,
        time_coarsening_factors=time_coarsening_factors,
        min_time_steps=10,
        psor_tol=psor_tol,
        force_recompute=False,
    )

    plot_convergence(errors_M, errors_N, M_values, N_values)



if __name__ == "__main__":
    main()
    

