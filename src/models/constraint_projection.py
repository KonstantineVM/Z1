# ==============================================================================
# FILE: src/models/constraint_projection.py
# PURPOSE: Equality-constraint projection for Kalman state mean & covariance
# ==============================================================================

from typing import Tuple, Optional
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAVE_SCIPY = True
except Exception:
    sp = None
    spla = None
    _HAVE_SCIPY = False


def _is_sparse(x) -> bool:
    return _HAVE_SCIPY and sp.issparse(x)


def _to_csr(x):
    if _is_sparse(x):
        return x.tocsr()
    return x


def _symmetrize(P: np.ndarray) -> np.ndarray:
    # Numerical symmetry clean-up for covariance
    return 0.5 * (P + P.T)


def _solve_Sy_equals_r(S, r, rtol: float = 1e-10):
    """
    Solve S y = r robustly.
    - If S is sparse: use spla.factorized or least-squares
    - If S is dense: use np.linalg.lstsq
    """
    if _is_sparse(S):
        try:
            # Prefer a direct sparse factorization if available
            factor = spla.factorized(S.tocsc())
            return factor(r)
        except Exception:
            # Fall back to least squares on a (possibly) dense view
            Sd = S.toarray()
            y, *_ = np.linalg.lstsq(Sd, r, rcond=rtol)
            return y
    else:
        y, *_ = np.linalg.lstsq(S, r, rcond=rtol)
        return y


def project_state_and_cov(
    x: np.ndarray,
    P: np.ndarray,
    A,
    b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    rtol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Project (x, P) onto equality constraints A x = b.

    Parameters
    ----------
    x : (n,) state mean
    P : (n,n) state covariance (symmetric PSD)
    A : (k,n) constraint matrix (dense or CSR/CSC)
    b : (k,) right-hand side
    weights : optional (k,) positive weights; if provided, enforce (W A) x = (W b)
    rtol : solver tolerance for least-squares

    Returns
    -------
    x_new : projected mean
    P_new : projected covariance
    max_resid : max absolute post-projection residual ||A x_new - b||_inf
    """
    if A is None or (hasattr(A, "shape") and A.shape[0] == 0):
        return x, P, 0.0

    # Normalize shapes
    x = np.asarray(x).reshape(-1)
    P = np.asarray(P)
    b = np.asarray(b).reshape(-1)
    n = x.shape[0]

    # Convert A to CSR if sparse for efficient multiplies
    A = _to_csr(A)

    # Optionally weight constraints: A_w = W A, b_w = W b
    if weights is not None:
        w = np.asarray(weights).reshape(-1)
        if _is_sparse(A):
            W = sp.diags(w)
            A_w = W @ A
            b_w = w * b
        else:
            A_w = (w[:, None]) * A
            b_w = w * b
    else:
        A_w = A
        b_w = b

    # Compute S = A P A'
    if _is_sparse(A_w):
        AP = A_w @ P
        S = AP @ A_w.T
    else:
        AP = A_w @ P
        S = AP @ A_w.T   # (k,k)

    # Residual r = A x - b  (weighted)
    r = (A_w @ x) - b_w

    # Solve S y = r
    y = _solve_Sy_equals_r(S, r, rtol=rtol)

    # Gain Kc = P A' y_solve with y_solve = S^{-1} r  => Kc = P A' S^{-1}
    # But we only need Kc @ r. Compute Kc_r = P A' y.
    if _is_sparse(A_w):
        Kc_r = P @ (A_w.T @ y)
    else:
        Kc_r = P @ (A_w.T @ y)

    # Project mean
    x_new = x - Kc_r

    # Project covariance: P_new = P - P A' S^{-1} A P
    # Reuse intermediate: U = P A' S^{-1} = (P A') * y_solve for generic RHS.
    # To avoid explicit inverse, compute U = P A' S^{-1} as:
    # Solve S U' = (A P)' for U' columns if needed. For efficiency at scale,
    # do a rank-update using the same factorization. Here, use: UAP = Kc_r on a basis;
    # general form (dense) below for clarity.
    if _is_sparse(A_w):
        # Dense update to keep things simple and stable here
        # (can be optimized to sparse rank-k update later)
        Sd = S.toarray()
        # Solve for U': S U' = (A P)' = P' A'  => U' = S^{-1} (P A')'
        PA_T = (P @ A_w.T)
        # Solve each column of U' (k rhs)
        U_T = np.linalg.solve(Sd, PA_T.T)
        U = U_T.T
        P_new = P - PA_T @ U
    else:
        PA_T = P @ A_w.T
        # Solve S U' = (P A')' for U' with lstsq
        U_T, *_ = np.linalg.lstsq(S, PA_T.T, rcond=rtol)
        U = U_T.T
        P_new = P - PA_T @ U

    P_new = _symmetrize(P_new)

    # Report post-projection residual (unweighted)
    resid = (A @ x_new) - b
    max_resid = float(np.max(np.abs(resid))) if resid.size else 0.0
    return x_new, P_new, max_resid

