from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.stats import norm


@dataclass
class KnockOutCallParams:
    S0: float = 100.0
    K: float = 100.0
    B: float = 80.0
    R: float = 0.0
    T: float = 1.0
    r: float = 0.02
    kappa: float = 2.0
    theta: float = 0.04
    sigma: float = 0.5
    rho: float = -0.7
    S_min: float = 0.0
    S_max: float = 400.0
    v_min: float = 0.0
    v_max: float = 0.5
    Ns: int = 200
    Nv: int = 50
    Nt: int = 100
    c: float = 20.0
    d: float = 0.05
    theta_ADI: float = 0.5
    v0: float | None = None
    debug: bool = False

    def initial_variance(self) -> float:
        return self.theta if self.v0 is None else self.v0


def create_nonuniform_asset_grid(
    S_min: float, S_max: float, Ns: int, K: float, c: float, B: float | None = None
) -> np.ndarray:
    """
    Literature-standard sinh asset grid

    s_i = K + c * sinh(xi_i),

    where xi_i is uniformly spaced between the transformed lower/upper bounds.
    The optional barrier lets the PDE domain start at the knock-out boundary instead
    of at S_min, which avoids wasting points below the barrier.
    """
    if Ns < 2:
        raise ValueError("Ns must be at least 2.")
    if c <= 0.0:
        raise ValueError("Asset grid scale c must be strictly positive.")

    lower = S_min if B is None else max(S_min, B)
    if lower >= S_max:
        raise ValueError("Asset grid lower bound must be smaller than S_max.")

    xi_min = np.arcsinh((lower - K) / c)
    xi_max = np.arcsinh((S_max - K) / c)
    xi = np.linspace(xi_min, xi_max, Ns)
    S_grid = K + c * np.sinh(xi)

    # Pin the endpoints exactly to the intended domain to avoid tiny roundoff drift.
    S_grid[0] = lower
    S_grid[-1] = S_max
    return S_grid


def create_nonuniform_variance_grid(
    v_min: float, v_max: float, Nv: int, d: float
) -> np.ndarray:
    xi = np.linspace(0.0, 1.0, Nv)
    alpha = np.arcsinh((v_max - v_min) / d)
    v_grid = v_min + d * np.sinh(alpha * xi)
    return v_grid


def first_derivative_matrix(x: np.ndarray) -> np.ndarray:
    n = len(x)
    D = np.zeros((n, n))

    for i in range(1, n - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        D[i, i - 1] = -hp / (hm * (hm + hp))
        D[i, i] = (hp - hm) / (hm * hp)
        D[i, i + 1] = hm / (hp * (hm + hp))

    D[0, 0] = -1.0 / (x[1] - x[0])
    D[0, 1] = 1.0 / (x[1] - x[0])
    D[-1, -2] = -1.0 / (x[-1] - x[-2])
    D[-1, -1] = 1.0 / (x[-1] - x[-2])
    return D


def second_derivative_matrix(x: np.ndarray) -> np.ndarray:
    n = len(x)
    D2 = np.zeros((n, n))

    for i in range(1, n - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        D2[i, i - 1] = 2.0 / (hm * (hm + hp))
        D2[i, i] = -2.0 / (hm * hp)
        D2[i, i + 1] = 2.0 / (hp * (hm + hp))

    return D2


def _zero_sparse_rows(matrix: csr_matrix, rows: np.ndarray) -> csr_matrix:
    if rows.size == 0:
        return matrix

    row_mask = np.ones(matrix.shape[0], dtype=bool)
    row_mask[rows] = False
    return matrix.multiply(row_mask[:, None]).tocsr()


def _boundary_row_indices(Ns: int, Nv: int) -> np.ndarray:
    rows = set()

    # Asset boundaries for every variance level in Fortran-order vectorization.
    for j in range(Nv):
        rows.add(j * Ns)
        rows.add(j * Ns + Ns - 1)

    # The v=0 line is treated by a dedicated reduced-PDE solve.
    rows.update(range(Ns))

    return np.array(sorted(rows), dtype=int)


def apply_boundary_conditions_matrix(
    U_mat: np.ndarray,
    S_grid: np.ndarray,
    v_grid: np.ndarray,
    K: float,
    B: float,
    r: float,
    tau: float,
    rebate: float,
) -> np.ndarray:
    S_max = S_grid[-1]

    U_mat[S_grid <= B, :] = rebate
    U_mat[0, :] = rebate
    U_mat[-1, :] = S_max - K * np.exp(-r * tau)
    U_mat[:, -1] = U_mat[:, -2]
    U_mat[S_grid <= B, :] = rebate
    return U_mat


def build_v0_line_solver(
    Ns: int,
    S_grid: np.ndarray,
    D_S: np.ndarray,
    dt: float,
    r: float,
    kappa: float,
    theta_heston: float,
    v_grid: np.ndarray,
):
    dv0 = v_grid[1] - v_grid[0]
    alpha_v0 = kappa * theta_heston / dv0

    I_S_sparse = eye(Ns, format="csc")
    S_diag = diags(S_grid, format="csr")
    D_S_sparse = csr_matrix(D_S)
    L_v0 = r * (S_diag @ D_S_sparse) - (alpha_v0 + r) * I_S_sparse
    L_v0 = _zero_sparse_rows(L_v0.tocsr(), np.array([0, Ns - 1], dtype=int)).tocsc()
    M_v0 = (I_S_sparse - dt * L_v0).tocsc()
    return alpha_v0, splu(M_v0)


def enforce_v0_boundary_european_line_solve(
    U_mat: np.ndarray,
    S_grid: np.ndarray,
    K: float,
    B: float,
    r: float,
    tau: float,
    rebate: float,
    alpha_v0: float,
    lu_v0,
    dt: float,
) -> np.ndarray:
    U0_old = U_mat[:, 0].copy()
    U1 = U_mat[:, 1].copy()

    rhs = U0_old + dt * alpha_v0 * U1
    U0_new = lu_v0.solve(rhs)
    U_mat[:, 0] = U0_new

    U_mat[S_grid <= B, 0] = rebate
    U_mat[0, 0] = rebate
    U_mat[-1, 0] = S_grid[-1] - K * np.exp(-r * tau)
    return U_mat


def douglas_adi_step(u_n, dt, theta_adi, A, A1, A2, lu_M1, lu_M2):
    Y0 = u_n + dt * (A @ u_n)
    rhs1 = Y0 - theta_adi * dt * (A1 @ u_n)
    Y1 = lu_M1.solve(rhs1)
    rhs2 = Y1 - theta_adi * dt * (A2 @ u_n)
    Y2 = lu_M2.solve(rhs2)
    return Y2


def bilinear_interpolate(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x0: float,
    y0: float,
) -> float:
    i = np.searchsorted(x_grid, x0)
    j = np.searchsorted(y_grid, y0)

    i = min(max(i, 1), len(x_grid) - 1)
    j = min(max(j, 1), len(y_grid) - 1)

    x1, x2 = x_grid[i - 1], x_grid[i]
    y1, y2 = y_grid[j - 1], y_grid[j]

    q11 = values[i - 1, j - 1]
    q12 = values[i - 1, j]
    q21 = values[i, j - 1]
    q22 = values[i, j]

    tx = 0.0 if x2 == x1 else (x0 - x1) / (x2 - x1)
    ty = 0.0 if y2 == y1 else (y0 - y1) / (y2 - y1)

    return (
        (1.0 - tx) * (1.0 - ty) * q11
        + (1.0 - tx) * ty * q12
        + tx * (1.0 - ty) * q21
        + tx * ty * q22
    )


def first_derivative_coeffs(x, i):
    """
    Coefficients for first derivative on a nonuniform grid using
    the 3-point stencil at node i.
    Returns coefficients for [i-1, i, i+1].
    """
    hm = x[i] - x[i - 1]
    hp = x[i + 1] - x[i]

    a_im1 = -hp / (hm * (hm + hp))
    a_i = (hp - hm) / (hm * hp)
    a_ip1 = hm / (hp * (hm + hp))

    return a_im1, a_i, a_ip1


def second_derivative_coeffs(x, i):
    """
    Coefficients for second derivative on a nonuniform grid using
    the 3-point stencil at node i.
    Returns coefficients for [i-1, i, i+1].
    """
    hm = x[i] - x[i - 1]
    hp = x[i + 1] - x[i]

    b_im1 = 2.0 / (hm * (hm + hp))
    b_i = -2.0 / (hm * hp)
    b_ip1 = 2.0 / (hp * (hm + hp))

    return b_im1, b_i, b_ip1


def greeks_from_surface(U_surface, S_grid, v_grid, S0, v0, D_S, D_SS, D_v):
    i = np.argmin(np.abs(S_grid - S0))
    j = np.argmin(np.abs(v_grid - v0))

    i = min(max(i, 1), len(S_grid) - 2)
    j = min(max(j, 1), len(v_grid) - 2)

    delta_surface = D_S @ U_surface
    gamma_surface = D_SS @ U_surface
    vega_surface = U_surface @ D_v.T

    return {
        "i": i,
        "j": j,
        "S_i": S_grid[i],
        "v_j": v_grid[j],
        "price": bilinear_interpolate(S_grid, v_grid, U_surface, S0, v0),
        "delta": bilinear_interpolate(S_grid, v_grid, delta_surface, S0, v0),
        "gamma": bilinear_interpolate(S_grid, v_grid, gamma_surface, S0, v0),
        "vega_v": bilinear_interpolate(S_grid, v_grid, vega_surface, S0, v0),
    }


def theta_from_surfaces(U0, U_penultimate, S_grid, v_grid, S0, v0, dt):
    i = np.argmin(np.abs(S_grid - S0))
    j = np.argmin(np.abs(v_grid - v0))

    i = min(max(i, 1), len(S_grid) - 2)
    j = min(max(j, 1), len(v_grid) - 2)

    theta_greek = -(U0[i, j] - U_penultimate[i, j]) / dt

    return {
        "i": i,
        "j": j,
        "S_i": S_grid[i],
        "v_j": v_grid[j],
        "theta_greek": theta_greek,
    }


def price_european_down_and_out_call_heston_adi(
    params: KnockOutCallParams,
) -> dict[str, float | np.ndarray | dict[str, float | int] | None]:
    S_grid = create_nonuniform_asset_grid(
        params.S_min, params.S_max, params.Ns, params.K, params.c, B=params.B
    )
    v_grid = create_nonuniform_variance_grid(
        params.v_min, params.v_max, params.Nv, params.d
    )

    U = np.maximum(S_grid[:, None] - params.K, 0.0) * np.ones((1, params.Nv))
    U[S_grid <= params.B, :] = params.R

    D_S = first_derivative_matrix(S_grid)
    D_SS = second_derivative_matrix(S_grid)
    D_v = first_derivative_matrix(v_grid)
    D_vv = second_derivative_matrix(v_grid)

    I_S = eye(params.Ns, format="csr")
    I_v = eye(params.Nv, format="csr")

    D_S_sparse = csr_matrix(D_S)
    D_SS_sparse = csr_matrix(D_SS)
    D_v_sparse = csr_matrix(D_v)
    D_vv_sparse = csr_matrix(D_vv)

    S_diag = diags(S_grid, format="csr")
    S2_diag = diags(S_grid**2, format="csr")
    v_diag = diags(v_grid, format="csr")
    kv_diag = diags(params.kappa * (params.theta - v_grid), format="csr")

    A0 = (
        params.rho
        * params.sigma
        * kron(v_diag @ D_v_sparse, S_diag @ D_S_sparse, format="csr")
    )
    A1 = (
        0.5 * kron(v_diag, S2_diag @ D_SS_sparse, format="csr")
        + params.r * kron(I_v, S_diag @ D_S_sparse, format="csr")
        - 0.5 * params.r * eye(params.Ns * params.Nv, format="csr")
    )
    A2 = (
        0.5 * params.sigma**2 * kron(v_diag @ D_vv_sparse, I_S, format="csr")
        + kron(kv_diag @ D_v_sparse, I_S, format="csr")
        - 0.5 * params.r * eye(params.Ns * params.Nv, format="csr")
    )
    boundary_rows = _boundary_row_indices(params.Ns, params.Nv)
    A0 = _zero_sparse_rows(A0, boundary_rows)
    A1 = _zero_sparse_rows(A1, boundary_rows)
    A2 = _zero_sparse_rows(A2, boundary_rows)
    A = A0 + A1 + A2

    dt = params.T / params.Nt
    I = eye(params.Ns * params.Nv, format="csr")
    theta_adi = params.theta_ADI

    M1 = (I - theta_adi * dt * A1).tocsc()
    M2 = (I - theta_adi * dt * A2).tocsc()
    lu_M1 = splu(M1)
    lu_M2 = splu(M2)

    alpha_v0, lu_v0 = build_v0_line_solver(
        params.Ns,
        S_grid,
        D_S,
        dt,
        params.r,
        params.kappa,
        params.theta,
        v_grid,
    )

    U_work = U.copy()
    u = U_work.flatten(order="F")
    U_penultimate = None

    for n in range(params.Nt):
        tau_np1 = params.T - (n + 1) * dt
        u = douglas_adi_step(u, dt, theta_adi, A, A1, A2, lu_M1, lu_M2)
        U_work = u.reshape((params.Ns, params.Nv), order="F")
        U_work = apply_boundary_conditions_matrix(
            U_work,
            S_grid,
            v_grid,
            params.K,
            params.B,
            params.r,
            tau_np1,
            params.R,
        )
        U_work = enforce_v0_boundary_european_line_solve(
            U_work,
            S_grid,
            params.K,
            params.B,
            params.r,
            tau_np1,
            params.R,
            alpha_v0,
            lu_v0,
            dt,
        )
        if n == params.Nt - 2:
            U_penultimate = U_work.copy()
        if params.debug:
            print(
                n,
                np.nanmin(U_work),
                np.nanmax(U_work),
                np.any(np.isnan(U_work)),
                np.any(np.isinf(U_work)),
            )
        u = U_work.flatten(order="F")

    v0 = params.initial_variance()
    price = bilinear_interpolate(S_grid, v_grid, U_work, params.S0, v0)
    greek_values = greeks_from_surface(
        U_work, S_grid, v_grid, params.S0, v0, D_S, D_SS, D_v
    )
    theta_values = (
        theta_from_surfaces(U_work, U_penultimate, S_grid, v_grid, params.S0, v0, dt)
        if U_penultimate is not None
        else {"theta_greek": np.nan}
    )

    return {
        "price": float(price),
        "surface_t0": U_work,
        "surface_penultimate": U_penultimate,
        "S_grid": S_grid,
        "v_grid": v_grid,
        "v0": float(v0),
        "greeks_pde": {
            "price_grid": float(greek_values["price"]),
            "i": int(greek_values["i"]),
            "j": int(greek_values["j"]),
            "S_i": float(greek_values["S_i"]),
            "v_j": float(greek_values["v_j"]),
            "delta": float(greek_values["delta"]),
            "gamma": float(greek_values["gamma"]),
            "vega_v": float(greek_values["vega_v"]),
            "theta_greek": float(theta_values["theta_greek"]),
        },
    }


def run_experiment_grid(
    base_params: dict,
    experiment_list: list[dict],
    label: str = "experiment",
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []

    for i, overrides in enumerate(experiment_list, start=1):
        started_at = perf_counter()
        params_dict = base_params.copy()
        params_dict.update(overrides)
        if "B" in params_dict and "S_min" in params_dict:
            params_dict["S_min"] = max(params_dict["S_min"], params_dict["B"])
        params = KnockOutCallParams(**params_dict)

        try:
            result = price_european_down_and_out_call_heston_adi(params)
            surface = result["surface_t0"]
            elapsed = perf_counter() - started_at
            rows.append(
                {
                    "group": label,
                    "run": i,
                    "price": result["price"],
                    "min_surface": (
                        float(np.nanmin(surface))
                        if isinstance(surface, np.ndarray)
                        else np.nan
                    ),
                    "max_surface": (
                        float(np.nanmax(surface))
                        if isinstance(surface, np.ndarray)
                        else np.nan
                    ),
                    "has_nan": (
                        bool(np.isnan(surface).any())
                        if isinstance(surface, np.ndarray)
                        else True
                    ),
                    "has_inf": (
                        bool(np.isinf(surface).any())
                        if isinstance(surface, np.ndarray)
                        else True
                    ),
                    "elapsed_sec": elapsed,
                    **overrides,
                }
            )
            if verbose:
                print(
                    f"[{label}] run {i}/{len(experiment_list)} finished "
                    f"in {elapsed:.3f}s with price={result['price']:.12f}"
                )
        except Exception as exc:
            elapsed = perf_counter() - started_at
            rows.append(
                {
                    "group": label,
                    "run": i,
                    "price": np.nan,
                    "min_surface": np.nan,
                    "max_surface": np.nan,
                    "has_nan": True,
                    "has_inf": True,
                    "elapsed_sec": elapsed,
                    "error": str(exc),
                    **overrides,
                }
            )
            if verbose:
                print(
                    f"[{label}] run {i}/{len(experiment_list)} failed "
                    f"in {elapsed:.3f}s with error: {exc}"
                )

    return pd.DataFrame(rows)


def european_call_bsm_price(S0, K, T, r, q, sigma):

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def european_put_bsm_price(S0, K, T, r, q, sigma):

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


def european_down_and_in_call_bsm_price(S0, K, B, T, r, q, sigma):
    lambda_ = (r - q + 0.5 * sigma**2) / (sigma**2)
    y = np.log(B**2 / (S0 * K)) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    x1 = np.log(S0 / B) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    y1 = np.log(B / S0) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    if B < K:
        c_di = S0 * np.exp(-q * T) * (B / S0) ** (2 * lambda_) * norm.cdf(
            y
        ) - K * np.exp(-r * T) * (B / S0) ** (2 * lambda_ - 2) * norm.cdf(
            y - sigma * np.sqrt(T)
        )
    else:
        c_di = (
            european_call_bsm_price(S0, K, T, r, q, sigma)
            - S0
            * (np.exp(-q * T) * norm.cdf(x1) - (B / S0) ** (2 * lambda_) * norm.cdf(y1))
            + K
            * np.exp(-r * T)
            * (
                norm.cdf(y - sigma * np.sqrt(T))
                - (B / S0) ** (2 * lambda_ - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
            )
        )
    return c_di


def european_down_and_out_call_bsm_price(S0, K, B, T, r, q, sigma):
    c_di = european_down_and_in_call_bsm_price(S0, K, B, T, r, q, sigma)
    c_european = european_call_bsm_price(S0, K, T, r, q, sigma)
    c_do = c_european - c_di
    return c_do


def european_up_and_in_call_bsm_price(S0, K, B, T, r, q, sigma):
    lambda_ = (r - q + 0.5 * sigma**2) / (sigma**2)
    y = np.log(B**2 / (S0 * K)) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    x1 = np.log(S0 / B) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    y1 = np.log(B / S0) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    if B > K:
        c_ui = (
            S0 * np.exp(-q * T) * norm.cdf(x1)
            - K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
            - S0
            * np.exp(-q * T)
            * (B / S0) ** (2 * lambda_)
            * ((norm.cdf(-y) - norm.cdf(-y1)))
            + K
            * np.exp(-r * T)
            * (B / S0) ** (2 * lambda_ - 2)
            * (norm.cdf(-y + sigma * np.sqrt(T)) - norm.cdf(-y1 + sigma * np.sqrt(T)))
        )
    else:
        c_ui = european_call_bsm_price(S0, K, T, r, q, sigma)
    return c_ui

def european_up_and_out_call_bsm_price(S0, K, B, T, r, q, sigma):
    c_ui = european_up_and_in_call_bsm_price(S0, K, B, T, r, q, sigma)
    c_european = european_call_bsm_price(S0, K, T, r, q, sigma)
    c_uo = c_european - c_ui
    return c_uo

if __name__ == "__main__":
    params = KnockOutCallParams()
    result = price_european_down_and_out_call_heston_adi(params)
    v0_value = result["v0"]
    v0_float = (
        float(v0_value) if isinstance(v0_value, (float, int, np.floating)) else np.nan
    )
    print(
        f"option price at S0={params.S0}, volatility={np.sqrt(v0_float):.6f} "
        f"is: {result['price']:.12f}"
    )
