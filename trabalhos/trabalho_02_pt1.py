#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Constrained Optimization Benchmark Harness
--------------------------------------------------
Give this file to students. They should:
  1) Implement or import THEIR Augmented Lagrangian (AL) class.
  2) Wire it in the function `build_student_optimizer` below (single place).
  3) Optionally adjust hyperparameters there.

What this script does:
  - Runs several constrained benchmark problems (mix of convex/nonconvex, eq/ineq).
  - Compares THEIR optimizer vs SciPy trust-constr on the same problems.
  - Prints a compact results table + summary.

Conventions:
  - Inequalities: return g(x) with g(x) <= 0 feasible.
  - Equalities:   return h(x) = 0.
  - Objective/constraints are written in PyTorch for auto-diff (double precision).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, Callable

import numpy as np
import torch

# ======================================================================================
# 0) >>>>>>>>>>>>>>>>>>>>>>>>>>>>  STUDENT TODO: PLUG YOUR CLASS HERE  <<<<<<<<<<<<<<<<<<<<<<<<<<
# ======================================================================================

"""
Replace the placeholder below with your own AL optimizer.
Expected call signature after you adapt here:

    optimizer = build_student_optimizer(
        objective=...,                          # Callable[[torch.Tensor], torch.Tensor]
        equality_constraints=...,               # Optional[Callable[[torch.Tensor], torch.Tensor]]
        inequality_constraints=...,             # Optional[Callable[[torch.Tensor], torch.Tensor]]
        device="cpu",                           # str
        **your_params                           # Any extra kwargs you need
    )
    x_opt, info = optimizer.optimize(x0)        # x0: torch.Tensor
    # 'info' should be a dict with at least:
    #   - 'final_objective' (float)
    #   - 'final_constraint_violation' (float, inf-norm of violations)
    #   - 'final_kkt_residual' (float)  [if you don't have it, set to grad norm or 0.0]
    #   - 'iterations' (int)
    #   - 'converged' (bool)
    #   - 'convergence_reason' (str)
"""

# --- Example adapter: replace with your imports / constructor -------------------------
# from my_al_package import MyAugmentedLagrangian  # <-- Your import

def build_student_optimizer(
    objective: Callable[[torch.Tensor], torch.Tensor],
    equality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    inequality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: str = "cpu",
    **kwargs: Any,
):
    """
    STUDENT TODO:
      - Replace the body with construction of YOUR optimizer.
      - Map parameters from **kwargs as you see fit.
      - Ensure it exposes .optimize(x0) -> (x_opt, info dict).

    Minimal adapter example (delete this and plug yours):
        opt = MyAugmentedLagrangian(
            objective=objective,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            device=device,
            penalty_init=kwargs.get("initial_penalty", 10.0),
            tol=kwargs.get("tolerance", 1e-6),
            max_iters=kwargs.get("max_iterations", 50),
            # ... add your own params ...
        )
        return opt
    """
    class _PlaceholderAL:
        def __init__(self, **_):
            raise NotImplementedError(
                "STUDENT: Replace build_student_optimizer() with your AL class.\n"
                "See the docstring above for expected interface."
            )
        def optimize(self, x0: torch.Tensor):
            raise NotImplementedError
    return _PlaceholderAL()  # <-- Replace with your optimizer instance


# ======================================================================================
# 1) SciPy bridge — autograd-backed objective/constraints and violation metrics
# ======================================================================================

def _torch_obj_grad_to_numpy(
    f: Callable[[torch.Tensor], torch.Tensor],
    dtype=np.float64,
) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    def f_np(x_np: np.ndarray) -> float:
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        fx = f(x)
        return float(fx.detach().cpu().numpy())
    def grad_np(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        fx = f(x)
        (gx,) = torch.autograd.grad(fx, x, create_graph=False, retain_graph=False)
        return gx.detach().cpu().numpy().astype(dtype)
    return f_np, grad_np

def _normalize_vec_out(y: torch.Tensor | float) -> torch.Tensor:
    if isinstance(y, torch.Tensor):
        if y.ndim == 0:
            return y.reshape(1)
        return y.reshape(-1)
    return torch.tensor([float(y)], dtype=torch.float64)

def _torch_vec_fun_jac_to_numpy(
    fun: Optional[Callable[[torch.Tensor], torch.Tensor | float]],
    dtype=np.float64,
) -> Tuple[Optional[Callable[[np.ndarray], np.ndarray]], Optional[Callable[[np.ndarray], np.ndarray]]]:
    if fun is None:
        return None, None
    def fun_np(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        gx = _normalize_vec_out(fun(x))
        return gx.detach().cpu().numpy().astype(dtype)
    def jac_np(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        def g_vec(z: torch.Tensor) -> torch.Tensor:
            return _normalize_vec_out(fun(z))
        J = torch.autograd.functional.jacobian(g_vec, x)
        return J.detach().cpu().numpy().astype(dtype)
    return fun_np, jac_np

def _inf_norm_viol(ineq_vals: Optional[np.ndarray], eq_vals: Optional[np.ndarray]) -> float:
    parts = []
    if ineq_vals is not None and ineq_vals.size > 0:
        # inequalities are feasible if <= 0; positive parts are violations
        parts.append(np.maximum(ineq_vals, 0.0).max(initial=0.0))
    if eq_vals is not None and eq_vals.size > 0:
        parts.append(np.abs(eq_vals).max(initial=0.0))
    return float(max(parts) if parts else 0.0)

def _grad_norm_at(f: Callable[[torch.Tensor], torch.Tensor], x_np: np.ndarray) -> float:
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    fx = f(x)
    (gx,) = torch.autograd.grad(fx, x, create_graph=False, retain_graph=False)
    return float(torch.linalg.norm(gx).item())

def run_scipy_baseline(
    x0_torch: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    equality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    inequality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    method: str = "trust-constr",
    verbose: int = 0,
) -> Dict[str, object]:
    try:
        import scipy.optimize as scopt
        from scipy.optimize import Bounds, NonlinearConstraint
    except Exception as e:
        return {"available": False, "reason": f"SciPy unavailable: {e}"}

    x0 = x0_torch.detach().cpu().numpy().astype(np.float64)
    f_np, grad_np = _torch_obj_grad_to_numpy(objective)

    constraints = []
    # Equalities: h(x) = 0
    if equality_constraints is not None:
        eq_fun_np, eq_jac_np = _torch_vec_fun_jac_to_numpy(equality_constraints)
        constraints.append(NonlinearConstraint(eq_fun_np, 0.0, 0.0, jac=eq_jac_np))
    # Inequalities: g(x) <= 0
    if inequality_constraints is not None:
        ineq_fun_np, ineq_jac_np = _torch_vec_fun_jac_to_numpy(inequality_constraints)
        constraints.append(NonlinearConstraint(ineq_fun_np, -np.inf, 0.0, jac=ineq_jac_np))

    sci_bounds = None
    if bounds is not None:
        lb, ub = bounds
        sci_bounds = Bounds(lb, ub)

    options = dict(verbose=3 if verbose else 0, maxiter=2000, xtol=1e-12, gtol=1e-12, barrier_tol=1e-12)

    t0 = time.time()
    res = scopt.minimize(
        fun=f_np,
        x0=x0,
        method=method,
        jac=grad_np,
        constraints=constraints if constraints else None,
        bounds=sci_bounds,
        options=options,
    )
    t1 = time.time()

    # Evaluate violations at SciPy solution
    eq_vals = None
    if equality_constraints is not None:
        x_tmp = torch.tensor(res.x, dtype=torch.float64, requires_grad=False)
        eq_vals = _normalize_vec_out(equality_constraints(x_tmp)).detach().cpu().numpy()
    ineq_vals = None
    if inequality_constraints is not None:
        x_tmp = torch.tensor(res.x, dtype=torch.float64, requires_grad=False)
        ineq_vals = _normalize_vec_out(inequality_constraints(x_tmp)).detach().cpu().numpy()

    viol = _inf_norm_viol(ineq_vals, eq_vals)
    grad_norm = _grad_norm_at(objective, res.x)

    return {
        "available": True,
        "x": res.x,
        "f": float(res.fun),
        "success": bool(res.success),
        "status": int(getattr(res, "status", -1)),
        "message": str(getattr(res, "message", "")),
        "niter": int(getattr(res, "niter", -1)),
        "time": t1 - t0,
        "viol": viol,
        "grad_norm": grad_norm,
    }


# ======================================================================================
# 2) Generic AL config holder (students can ignore or use)
# ======================================================================================

@dataclass
class ALConfig:
    # Students can map these into their optimizer in build_student_optimizer()
    initial_penalty: float = 10.0
    tolerance: float = 1e-6
    practical_tolerance: Optional[float] = None
    max_trust_radius: float = 10.0
    max_cg_iterations: int = 100
    use_second_order_correction: bool = True
    use_envelope_filter: bool = False
    verbose: int = 0
    device: str = "cpu"
    max_iterations: int = 50

    def build(self,
              objective: Callable[[torch.Tensor], torch.Tensor],
              equality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
              inequality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
              **overrides) -> Any:
        """Creates the student optimizer via the single adapter point."""
        params = asdict(self) | overrides
        if params.get("practical_tolerance", None) is None:
            params.pop("practical_tolerance")
        return build_student_optimizer(
            objective=objective,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            **params,
        )


# ======================================================================================
# 3) Test harness
# ======================================================================================

@dataclass
class TestResult:
    name: str
    x_opt: torch.Tensor
    f_opt: float
    constraint_viol: float
    kkt_residual: float
    iterations: int
    converged: bool
    convergence_reason: str
    solve_time: float
    theoretical_optimum: Optional[Tuple[torch.Tensor, float]] = None
    scipy_available: bool = False
    scipy_success: Optional[bool] = None
    scipy_status: Optional[int] = None
    scipy_message: Optional[str] = None
    scipy_x: Optional[np.ndarray] = None
    scipy_f: Optional[float] = None
    scipy_viol: Optional[float] = None
    scipy_grad_norm: Optional[float] = None
    scipy_niter: Optional[int] = None
    scipy_time: Optional[float] = None
    scipy_reason: Optional[str] = None

class OptimizationTestSuite:
    """Benchmark suite for constrained optimization (Student AL vs SciPy)."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float64, al_config: Optional[ALConfig] = None):
        self.device = device
        self.dtype = dtype
        self.results: List[TestResult] = []
        self.al_config = al_config or ALConfig()

    # Unified factory to create the student's optimizer
    def al(self,
           objective: Callable[[torch.Tensor], torch.Tensor],
           equality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
           inequality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
           **overrides) -> Any:
        return self.al_config.build(objective, equality_constraints, inequality_constraints, **overrides)

    # If the student's `info` is missing fields, normalize them here.
    @staticmethod
    def _normalize_info(
        info: Dict[str, Any],
        objective: Callable[[torch.Tensor], torch.Tensor],
        x_opt: torch.Tensor,
        equality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]],
        inequality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> Dict[str, Any]:
        out = dict(info)
        # Objective
        if "final_objective" not in out:
            out["final_objective"] = float(objective(x_opt).item())
        # Violations
        eq_vals = _normalize_vec_out(equality_constraints(x_opt)).detach().cpu().numpy() if equality_constraints else None
        ineq_vals = _normalize_vec_out(inequality_constraints(x_opt)).detach().cpu().numpy() if inequality_constraints else None
        viol = _inf_norm_viol(ineq_vals, eq_vals)
        out.setdefault("final_constraint_violation", float(viol))
        # KKT residual (fallback: grad norm)
        if "final_kkt_residual" not in out:
            (gx,) = torch.autograd.grad(objective(x_opt), x_opt, retain_graph=False, create_graph=False)
            out["final_kkt_residual"] = float(torch.linalg.norm(gx).item())
        out.setdefault("iterations", -1)
        out.setdefault("converged", False)
        out.setdefault("convergence_reason", "N/A")
        return out

    # Attach SciPy to a result
    def _attach_scipy(self,
                      result: TestResult,
                      x0: torch.Tensor,
                      objective: Callable[[torch.Tensor], torch.Tensor],
                      equality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                      inequality_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                      bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      verbose: int = 0) -> TestResult:
        sci = run_scipy_baseline(
            x0_torch=x0,
            objective=objective,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            bounds=bounds,
            method="trust-constr",
            verbose=verbose,
        )
        if not sci.get("available", False):
            result.scipy_available = False
            result.scipy_reason = sci.get("reason", "Unknown")
            return result

        result.scipy_available = True
        result.scipy_success = sci["success"]
        result.scipy_status = sci["status"]
        result.scipy_message = sci["message"]
        result.scipy_x = sci["x"]
        result.scipy_f = sci["f"]
        result.scipy_viol = sci["viol"]
        result.scipy_grad_norm = sci["grad_norm"]
        result.scipy_niter = sci["niter"]
        result.scipy_time = sci["time"]
        return result

    # ---------------------------
    # Benchmark problems
    # ---------------------------
    def test_rosenbrock_equality(self, verbose: int = 0) -> TestResult:
        def objective(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        def equality_constraint(x): return x[0]**2 + x[1]**2 - 2.0

        optimizer = self.al(objective, equality_constraints=equality_constraint, verbose=verbose)
        x0 = torch.tensor([1.2, 1.2], dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, equality_constraint, None)
        result = TestResult(
            name="Rosenbrock with Equality",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0,
            theoretical_optimum=(torch.tensor([1.0, 1.0]), 0.0)
        )
        return self._attach_scipy(result, x0, objective, equality_constraints=equality_constraint, verbose=verbose)

    def test_rosenbrock_inequality(self, verbose: int = 0) -> TestResult:
        def objective(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        def inequality_constraints(x):
            # g(x) <= 0:
            return torch.stack([x[0] + x[1] - 1.5, -x[0] + 0.5, -x[1] + 0.5])

        optimizer = self.al(objective, inequality_constraints=inequality_constraints, verbose=verbose)
        x0 = torch.tensor([0.8, 0.6], dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, None, inequality_constraints)
        result = TestResult(
            name="Rosenbrock with Inequalities",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0
        )
        return self._attach_scipy(result, x0, objective, inequality_constraints=inequality_constraints, verbose=verbose)

    def test_rosenbrock_box_constraints(self, verbose: int = 0) -> TestResult:
        def objective(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        def inequality_constraints(x):
            # Box: 0 <= x <= 0.8  -> as g(x) <= 0
            return torch.stack([-x[0], x[0]-0.8, -x[1], x[1]-0.8])

        optimizer = self.al(objective, inequality_constraints=inequality_constraints, verbose=verbose)
        x0 = torch.tensor([0.5, 0.5], dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, None, inequality_constraints)
        result = TestResult(
            name="Rosenbrock with Box Constraints",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0,
            theoretical_optimum=(torch.tensor([0.8, 0.64]), 0.04)
        )
        return self._attach_scipy(result, x0, objective, inequality_constraints=inequality_constraints, verbose=verbose)

    def test_circle_in_square(self, verbose: int = 0) -> TestResult:
        # Maximize area => minimize negative area
        def objective(v): x, y, r = v[0], v[1], v[2]; return -torch.pi * r**2
        def inequality_constraints(v):
            # Circle of radius r stays inside unit square [0,1]^2
            x, y, r = v[0], v[1], v[2]
            return torch.stack([r - x, r + x - 1.0, r - y, r + y - 1.0, -r])

        optimizer = self.al(objective, inequality_constraints=inequality_constraints, verbose=verbose)
        x0 = torch.tensor([0.4, 0.4, 0.3], dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, None, inequality_constraints)
        theoretical_area = float(np.pi * 0.25)
        result = TestResult(
            name="Circle in Square",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0,
            theoretical_optimum=(torch.tensor([0.5, 0.5, 0.5]), -theoretical_area)
        )
        return self._attach_scipy(result, x0, objective, inequality_constraints=inequality_constraints, verbose=verbose)

    def test_disk_constraint(self, verbose: int = 0) -> TestResult:
        def objective(x): return x[0]**2 + x[1]**2 - x[0] - x[1]
        def inequality_constraints(x): return torch.stack([x[0]**2 + x[1]**2 - 1.0])

        optimizer = self.al(objective, inequality_constraints=inequality_constraints, verbose=verbose)
        x0 = torch.tensor([0.3, 0.3], dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, None, inequality_constraints)
        x_theory = torch.tensor([0.5, 0.5], dtype=self.dtype); f_theory = -0.5
        result = TestResult(
            name="Disk Constraint",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0,
            theoretical_optimum=(x_theory, f_theory)
        )
        return self._attach_scipy(result, x0, objective, inequality_constraints=inequality_constraints, verbose=verbose)

    def test_himmelblau_constrained(self, verbose: int = 0) -> TestResult:
        def objective(x): return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
        def equality_constraint(x): return x[0] + x[1] - 4.0
        def inequality_constraints(x): return torch.stack([-x[0], -x[1]])

        optimizer = self.al(objective, equality_constraints=equality_constraint,
                            inequality_constraints=inequality_constraints, verbose=verbose)
        x0 = torch.tensor([2.0, 2.0], dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, equality_constraint, inequality_constraints)
        result = TestResult(
            name="Constrained Himmelblau",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0
        )
        return self._attach_scipy(result, x0, objective, equality_constraints=equality_constraint,
                                  inequality_constraints=inequality_constraints, verbose=verbose)

    def test_quadratic_with_linear_constraints(self, verbose: int = 0) -> TestResult:
        n = 4
        Q = torch.eye(n, dtype=self.dtype, device=self.device)
        Q[0, 1] = Q[1, 0] = 0.5
        Q[2, 3] = Q[3, 2] = -0.3
        c = torch.tensor([1, -2, 0.5, 1.5], dtype=self.dtype, device=self.device)
        A = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=self.dtype, device=self.device)
        b = torch.tensor([1.0], dtype=self.dtype, device=self.device)
        G = -torch.eye(n, dtype=self.dtype, device=self.device)
        h = torch.full((n,), -0.5, dtype=self.dtype, device=self.device)  # x >= -0.5

        def objective(x): return 0.5 * torch.dot(x, Q @ x) + torch.dot(c, x)
        def equality_constraint(x): return (A @ x - b).squeeze(-1)
        def inequality_constraints(x): return (G @ x + h).reshape(-1)

        optimizer = self.al(objective, equality_constraints=equality_constraint,
                            inequality_constraints=inequality_constraints, verbose=verbose)
        x0 = torch.zeros(n, dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, equality_constraint, inequality_constraints)
        result = TestResult(
            name="Quadratic with Linear Constraints",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0
        )
        return self._attach_scipy(result, x0, objective,
                                  equality_constraints=equality_constraint,
                                  inequality_constraints=inequality_constraints, verbose=verbose)

    def test_nonconvex_equality_constrained(self, verbose: int = 0) -> TestResult:
        def objective(x): return x[0]**4 + x[1]**4 - 4 * x[0]**2 * x[1]**2
        def equality_constraint(x): return x[0]**2 + x[1]**2 - 1.0

        optimizer = self.al(objective, equality_constraints=equality_constraint, verbose=verbose)
        best_result = None; best_f = float('inf'); best_x0 = None
        for ang in [0.1, 0.7, 1.2, 2.0, 3.0]:
            x0 = torch.tensor([np.cos(ang), np.sin(ang)], dtype=self.dtype, device=self.device)
            t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
            info = self._normalize_info(raw, objective, x_opt, equality_constraint, None)
            if info['final_objective'] < best_f:
                best_f = info['final_objective']
                best_result = TestResult(
                    name="Nonconvex Equality Constrained",
                    x_opt=x_opt, f_opt=info['final_objective'],
                    constraint_viol=info['final_constraint_violation'],
                    kkt_residual=info['final_kkt_residual'],
                    iterations=info['iterations'], converged=info['converged'],
                    convergence_reason=info['convergence_reason'], solve_time=t1 - t0,
                    theoretical_optimum=(torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)]), -0.5)
                )
                best_x0 = x0.clone()
        return self._attach_scipy(best_result, best_x0, objective,
                                  equality_constraints=equality_constraint, verbose=verbose)

    def test_portfolio_optimization(self, verbose: int = 0) -> TestResult:
        n_assets = 5
        np.random.seed(42)
        A = np.random.randn(n_assets, n_assets)
        Sigma = torch.tensor(A.T @ A + 0.1 * np.eye(n_assets), dtype=self.dtype, device=self.device)
        mu = torch.tensor([0.08, 0.12, 0.10, 0.09, 0.11], dtype=self.dtype, device=self.device)
        r_target = 0.10

        def objective(w): return w @ Sigma @ w
        def equality_constraint(w): return torch.sum(w) - 1.0
        def inequality_constraints(w):
            # -w <= 0 (w >= 0), and -mu^T w + r_target <= 0  (mu^T w >= r_target)
            return torch.cat([-w, (-torch.dot(mu, w) + r_target).reshape(1)])

        optimizer = self.al(objective, equality_constraints=equality_constraint,
                            inequality_constraints=inequality_constraints, verbose=verbose)
        x0 = torch.full((n_assets,), 1.0 / n_assets, dtype=self.dtype, device=self.device)
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, equality_constraint, inequality_constraints)
        result = TestResult(
            name="Portfolio Optimization",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0
        )
        return self._attach_scipy(result, x0, objective,
                                  equality_constraints=equality_constraint,
                                  inequality_constraints=inequality_constraints, verbose=verbose)

    def test_tensor_problem_high_dim(self, verbose: int = 0) -> TestResult:
        n = 10
        def objective(x): return torch.sum(x**4) - 2 * torch.sum(x**2)
        def equality_constraints(x): return torch.stack([torch.sum(x**2) - n, torch.sum(x)])
        def inequality_constraints(x): return torch.cat([x - 2.0, -x - 2.0])

        optimizer = self.al(objective, equality_constraints=equality_constraints,
                            inequality_constraints=inequality_constraints, verbose=verbose)
        torch.manual_seed(42)
        x0 = torch.randn(n, dtype=self.dtype, device=self.device)
        x0 = x0 / x0.norm() * np.sqrt(n)
        x0 = x0 - x0.mean()
        t0 = time.time(); x_opt, raw = optimizer.optimize(x0); t1 = time.time()
        info = self._normalize_info(raw, objective, x_opt, equality_constraints, inequality_constraints)
        result = TestResult(
            name="High-Dim Tensor Problem",
            x_opt=x_opt, f_opt=info['final_objective'],
            constraint_viol=info['final_constraint_violation'],
            kkt_residual=info['final_kkt_residual'],
            iterations=info['iterations'], converged=info['converged'],
            convergence_reason=info['convergence_reason'], solve_time=t1 - t0
        )
        return self._attach_scipy(result, x0, objective,
                                  equality_constraints=equality_constraints,
                                  inequality_constraints=inequality_constraints, verbose=verbose)

    # ---------------------------
    # Running & reporting
    # ---------------------------
    def run_all_tests(self, verbose: int = 1) -> Dict[str, TestResult]:
        test_problems = [
            self.test_rosenbrock_equality,
            self.test_rosenbrock_inequality,
            self.test_rosenbrock_box_constraints,
            self.test_circle_in_square,
            self.test_disk_constraint,
            self.test_himmelblau_constrained,
            self.test_quadratic_with_linear_constraints,
            self.test_nonconvex_equality_constrained,
            self.test_portfolio_optimization,
            self.test_tensor_problem_high_dim,
        ]

        results: Dict[str, TestResult] = {}
        col_fmt = "{:<6} {:<40} {:<6} {:>12} {:>8} {:>6} {:>7}  ||  {:<6} {:>12} {:>8} {:>6} {:>7} {:>10}"

        print("=" * 120)
        print("COMPREHENSIVE OPTIMIZATION TEST SUITE (Student AL vs SciPy)")
        print("=" * 120)
        headers = col_fmt.format(
            "Test", "Name", "AL", "f*", "θ", "Iter", "t(s)",
            "SciPy", "f*", "θ", "Iter", "t(s)", "|Δf|"
        )
        print(headers)
        print("-" * len(headers))

        for i, test_func in enumerate(test_problems, 1):
            result = test_func(verbose=verbose)
            results[result.name] = result
            self.results.append(result)

            status = "PASS" if result.converged else "FAIL"

            if result.scipy_available:
                sci_status = "ok" if result.scipy_success else "fail"
                sci_f = f"{result.scipy_f:.3e}"
                sci_theta = f"{result.scipy_viol:.1e}"
                sci_iter = str(result.scipy_niter)
                sci_t = f"{result.scipy_time:.3f}"
                obj_diff = abs(result.f_opt - (result.scipy_f or result.f_opt))
            else:
                sci_status, sci_f, sci_theta, sci_iter, sci_t = ["-"] * 5
                obj_diff = 0.0

            row = col_fmt.format(
                f"{i}/{len(test_problems)}",
                result.name[:40],
                status,
                f"{result.f_opt:.3e}",
                f"{result.constraint_viol:.1e}",
                str(result.iterations),
                f"{result.solve_time:.3f}",
                sci_status,
                sci_f,
                sci_theta,
                sci_iter,
                sci_t,
                f"{obj_diff:.2e}"
            )
            print(row)

        print("=" * 120)
        self._print_summary_statistics()
        return results

    def _print_summary_statistics(self):
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.converged)
        failed_tests = total_tests - passed_tests
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)
        print(f"Total tests run: {total_tests}")
        print(f"Tests passed (Student AL): {passed_tests}")
        print(f"Tests failed (Student AL): {failed_tests}")
        if total_tests > 0:
            avg_iterations = np.mean([r.iterations for r in self.results])
            avg_time = np.mean([r.solve_time for r in self.results])
            print(f"Average AL iterations: {avg_iterations:.2f}")
            print(f"Average AL solve time: {avg_time:.3f}s")
            sci = [r for r in self.results if r.scipy_available]
            if sci:
                sci_ok = sum(1 for r in sci if r.scipy_success)
                print(f"SciPy available on {len(sci)}/{total_tests} tests | success={sci_ok}")
                print(f"Average SciPy time: {np.mean([r.scipy_time for r in sci]):.3f}s")
                print(f"Average SciPy viol: {np.mean([r.scipy_viol for r in sci]):.2e}")
        print("=" * 80)


# ======================================================================
# 4) Entry point
# ======================================================================
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Centralized defaults students can map inside build_student_optimizer()
    al_defaults = ALConfig(
        initial_penalty=10.0,
        tolerance=1e-3,
        practical_tolerance=5e-5,
        max_trust_radius=10.0,
        max_cg_iterations=100,
        use_second_order_correction=True,
        use_envelope_filter=True,
        verbose=1,
        max_iterations=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    suite = OptimizationTestSuite(device="cpu", dtype=torch.float64, al_config=al_defaults)
    _ = suite.run_all_tests(verbose=0)
