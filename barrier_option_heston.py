import numpy as np
from scipy.sparse import eye, diags, kron, csr_matrix
from scipy.sparse.linalg import splu


def create_nonuniform_asset_price_grid(S_min, S_max, K, Ns, c=20):
    """
    Create a non-uniform asset price grid for barrier option pricing.

    s_i = K + c * sinh(xi_i)
    """
    if Ns < 2:
        raise ValueError("Number of grid points Ns must be at least 2.")
    if c <= 0:
        raise ValueError("Stretching parameter c must be positive.")

    if S_min >= S_max:
        raise ValueError("S_min must be less than S_max.")

    xi_min = np.arcsinh((S_min - K) / c)
    xi_max = np.arcsinh((S_max - K) / c)
    xi = np.linspace(xi_min, xi_max, Ns)
    S_grid = K + c * np.sinh(xi)

    S_grid[0] = S_min
    S_grid[-1] = S_max

    return S_grid


def create_nonuniform_variance_grid(v_min, v_max, Nv, d=0.05):
    """
    Create a non-uniform variance grid with clustering near v=0.

    v_j = d * sinh(eta_j)
    """
    if Nv < 2:
        raise ValueError("Number of grid points Nv must be at least 2.")
    if d <= 0:
        raise ValueError("Stretching parameter d must be positive.")

    if v_min >= v_max:
        raise ValueError("v_min must be less than v_max.")

    eta_min = np.arcsinh(v_min / d)
    eta_max = np.arcsinh(v_max / d)
    eta = np.linspace(eta_min, eta_max, Nv)
    v_grid = d * np.sinh(eta)

    v_grid[0] = v_min
    v_grid[-1] = v_max

    return v_grid

def initialize_option_price_grid_call(S_grid, v_grid, K):
    N_S = len(S_grid)
    N_v = len(v_grid)
    U = np.zeros((N_S, N_v))
    for i in range(N_S):
        for j in range(N_v):
            U[i, j] = max(S_grid[i] - K, 0)
    return U

def first_derivative_matrix(x):
    """
    Construct the first derivative matrix using central differences on a non-uniform grid.
    """
    N = len(x)
    D = np.zeros((N, N))

    for i in range(1, N - 1):
        h_forward = x[i + 1] - x[i]
        h_backward = x[i] - x[i - 1]
        D[i, i - 1] = -h_forward / (h_backward * (h_forward + h_backward))
        D[i, i] = (h_forward - h_backward) / (h_forward * h_backward)
        D[i, i + 1] = h_backward / (h_forward * (h_forward + h_backward))

    D[0, 0] = -1 / (x[1] - x[0])
    D[0, 1] = 1 / (x[1] - x[0])

    D[-1, -2] = -1 / (x[-1] - x[-2])
    D[-1, -1] = 1 / (x[-1] - x[-2])

    return D


def second_derivative_matrix(x):
    """
    Construct the second derivative matrix using central differences on a non-uniform grid.
    Uses three-point stencil at all points, including boundaries.
    """
    N = len(x)
    D2 = np.zeros((N, N))

    for i in range(1, N - 1):
        h_forward = x[i + 1] - x[i]
        h_backward = x[i] - x[i - 1]
        D2[i, i - 1] = 2 / (h_backward * (h_forward + h_backward))
        D2[i, i] = -2 / (h_forward * h_backward)
        D2[i, i + 1] = 2 / (h_forward * (h_forward + h_backward))

    # Left boundary (i=0): use forward stencil with points 0,1,2
    h0 = x[1] - x[0]
    h1 = x[2] - x[1]
    D2[0, 0] = 2 / (h0 * (h0 + h1))
    D2[0, 1] = -2 / (h0 * h1)
    D2[0, 2] = 2 / (h1 * (h0 + h1))

    # Right boundary (i=N-1): use backward stencil with points N-3,N-2,N-1
    h_n2 = x[-2] - x[-3]
    h_n1 = x[-1] - x[-2]
    D2[-1, -3] = 2 / (h_n2 * (h_n2 + h_n1))
    D2[-1, -2] = -2 / (h_n2 * h_n1)
    D2[-1, -1] = 2 / (h_n1 * (h_n2 + h_n1))

    return D2


def zero_sparse_rows(A, row_indices):
    if len(row_indices) == 0:
        return A
    row_mask = np.ones(A.shape[0], dtype=bool)
    row_mask[np.array(row_indices, dtype=int)] = False
    return A.multiply(row_mask[:, None]).tocsr()


def build_heston_operators(S_grid, v_grid, r, kappa, theta, sigma, rho):
    """
    Build the finite difference operators for the Heston PDE.
    Returns the sparse matrix representing the PDE operator.
    """
    N_S = len(S_grid)
    N_v = len(v_grid)

    D_S = first_derivative_matrix(S_grid)
    D_SS = second_derivative_matrix(S_grid)
    D_v = first_derivative_matrix(v_grid)
    D_vv = second_derivative_matrix(v_grid)

    I_S = eye(N_S, format="csr")
    I_v = eye(N_v, format="csr")

    D_S_sparse = csr_matrix(D_S)
    D_SS_sparse = csr_matrix(D_SS)
    D_v_sparse = csr_matrix(D_v)
    D_vv_sparse = csr_matrix(D_vv)

    S_diag = diags(S_grid, format="csr")
    v_diag = diags(v_grid, format="csr")
    S2_diag = diags(S_grid**2, format="csr")
    kv_diag = diags(kappa * (theta - v_grid), format="csr")

    A0 = rho * sigma * kron(v_diag @ D_v_sparse, S_diag @ D_S_sparse, format="csr")

    A1 = (
        0.5 * kron(v_diag, S2_diag @ D_SS_sparse, format="csr")
        + r * kron(I_v, S_diag @ D_S_sparse, format="csr")
        - 0.5 * r * eye(N_S * N_v, format="csr")
    )

    A2 = (
        0.5 * sigma**2 * kron(v_diag @ D_vv_sparse, I_S, format="csr")
        + kron(kv_diag @ D_v_sparse, I_S, format="csr")
        - 0.5 * r * eye(N_S * N_v, format="csr")
    )

    boundary_rows = np.unique(
        np.concatenate(
            [
                np.arange(N_S),  # v=0 boundary
                np.arange(0, N_S * N_v, N_S),  # S=0 boundary
                np.arange(N_S - 1, N_S * N_v, N_S),  # S=S_max boundary
            ]
        )
    )

    A0 = zero_sparse_rows(A0, boundary_rows)
    A1 = zero_sparse_rows(A1, boundary_rows)
    A2 = zero_sparse_rows(A2, boundary_rows)

    return {
        "A_mixed": A0,
        "A_S": A1,
        "A_v": A2,
    }


def MCS_ADI_step(u_n, dt, theta_ADI, A, A0, A1, A2, lu_M1, lu_M2):
    """
    Perform one time step using the MCS ADI method.
    """

    Y0 = u_n + dt * (A @ u_n)

    rhs1 = Y0 - theta_ADI * dt * (A1 @ u_n)
    Y1 = lu_M1.solve(rhs1)

    rhs2 = Y1 - theta_ADI * dt * (A2 @ u_n)
    Y2 = lu_M2.solve(rhs2)

    # Correction step for the mixed derivative term
    Y0_hat = Y0 + theta_ADI * dt * (A0 @ (Y2 - u_n))
    Y0_tilde = Y0_hat + (0.5 - theta_ADI) * dt * (A @ (Y2 - u_n))

    # Second implicit sweep
    rhs1_tilde = Y0_tilde - theta_ADI * dt * (A1 @ u_n)
    Y1_tilde = lu_M1.solve(rhs1_tilde)

    rhs2_tilde = Y1_tilde - theta_ADI * dt * (A2 @ u_n)
    Y2_tilde = lu_M2.solve(rhs2_tilde)

    return Y2_tilde


def apply_boundary_conditions_vanilla_call(U_mat, S_grid, v_grid, K, r, tau):
    """
    Apply boundary conditions to the solution vector U.
    - For S = S_max: U = S_max - PV(K) (call payoff)
    - For S = 0: U = 0
    - For v = 0 (zero variance), the PDE degenerate.
    - For v = v_max, we can use a Neumann boundary condition (zero flux).
    """
    S_max = S_grid[-1]
    # Apply barrier condition
    U_mat[0, :] = 0.0  # S=0 boundary

    # Apply call payoff at S=S_max
    U_mat[-1, :] = S_max - K * np.exp(-r * tau)

    # Apply high variance condition
    U_mat[:, -1] = U_mat[:, -2]  # Neumann condition at v=v_max

    return U_mat


def enforce_v0_boundary_condition(
    U_mat, S_grid, K, r, tau, dt_local, alpha_v0, lu_v0_local
):
    """
    Enforce the boundary condition at v=0.
    The PDE degenerates at v=0, and we can derive a boundary condition from the PDE itself.
    """
    U0_old = U_mat[:, 0].copy()
    U1 = U_mat[:, 1].copy()

    rhs = U0_old + dt_local * alpha_v0 * U1
    U0_new = lu_v0_local.solve(rhs)

    # Re-impose boundary conditions at v=0
    U_mat[:, 0] = U0_new
    U_mat[0, 0] = 0.0  # S=0 boundary
    U_mat[-1, 0] = S_grid[-1] - K * np.exp(-r * tau)  # S=S_max boundary

    return U_mat


def solve_heston_adi_vanilla_call(
    T, Nt, U, theta_ADI, S_grid, v_grid, r, kappa, theta_heston, sigma, rho, K
):
    """
    Solve the Heston PDE for a vanilla call option using the MCS ADI method.
    """
    dt = T / Nt
    u = U.flatten(order="F")
    I = eye(U.size, format="csr")
    S_diag = diags(S_grid, format="csr")

    # Directional implicit matrices for the ADI method

    operators = build_heston_operators(
        S_grid, v_grid, r, kappa, theta_heston, sigma, rho
    )
    A = operators["A_mixed"] + operators["A_S"] + operators["A_v"]

    A0 = operators["A_mixed"]
    A1 = operators["A_S"]
    A2 = operators["A_v"]

    I = eye(A.shape[0], format="csr")
    M1 = (I - theta_ADI * (T / Nt) * A1).tocsc()
    M2 = (I - theta_ADI * (T / Nt) * A2).tocsc()
    M_be = (I - (dt / 2) * A).tocsc()

    lu_M1 = splu(M1)
    lu_M2 = splu(M2)
    lu_be = splu(M_be)

    if Nt < 2:
        raise ValueError("Number of time steps Nt must be at least 2.")

    # v = 0 reduced operator
    dv0 = v_grid[1] - v_grid[0]
    alpha_v0 = kappa * theta_heston / dv0

    D_S = first_derivative_matrix(S_grid)
    I_S_sparse = eye(U.shape[0], format="csr")
    D_S_sparse = csr_matrix(D_S)

    L_v0 = r * (S_diag @ D_S_sparse) - (alpha_v0 + r) * I_S_sparse
    L_v0 = zero_sparse_rows(L_v0.tocsr(), np.array([0, S_grid.shape[0] - 1], dtype=int)).tocsr()
    M_v0 = (I_S_sparse - dt * L_v0).tocsc()
    lu_v0 = splu(M_v0)

    M_v0_rannacher = (I_S_sparse - 0.5 * dt * L_v0).tocsc()
    lu_v0_rannacher = splu(M_v0_rannacher)

    U_work = U.copy()
    u = U_work.flatten(order="F")
    tau = T

    # Rannacher smoothing: perform two half-steps with backward Euler to dampen high-frequency errors
    for k in range(2):
        tau -= dt / 2
        u = lu_be.solve(u)
        U_work = u.reshape(U.shape, order="F")
        U_work = apply_boundary_conditions_vanilla_call(
            U_work, S_grid, v_grid, K, r, tau
        )
        U_work = enforce_v0_boundary_condition(
            U_work, S_grid, K, r, tau, dt / 2, alpha_v0, lu_v0_rannacher
        )
        u = U_work.flatten(order="F")

    # Main ADI time-stepping loop
    for n in range(1, Nt):
        tau -= dt

        u = MCS_ADI_step(u, dt, theta_ADI, A, A0, A1, A2, lu_M1, lu_M2)

        U_work = u.reshape(U.shape, order="F")
        U_work = apply_boundary_conditions_vanilla_call(
            U_work, S_grid, v_grid, K, r, tau
        )
        U_work = enforce_v0_boundary_condition(
            U_work, S_grid, K, r, tau, dt, alpha_v0, lu_v0
        )
        u = U_work.flatten(order="F")

    return U_work

def price_option_heston(U, S0, v0, S_grid, v_grid):
    i = np.argmin(np.abs(S_grid - S0))
    j = np.argmin(np.abs(v_grid - v0))
    price = U[i, j]

    return price