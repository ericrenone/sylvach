#!/usr/bin/env python3
"""
Sylvach vs Pairwise Gossip Benchmark – Single-file Master Version
=================================================================

Distributed consensus on vectors or symmetric matrices.

Goal:
  All nodes converge to (≈) the global average state via local communication.

Key methods compared:
- Random Pairwise Gossip     : classic sparse, truly distributed baseline
- Sylvach (relaxed over-relaxation): X_i ← X_i + α (global_mean - X_i)

Features:
- Supports scalar (dim=1), vector, or symmetric matrix states
- α parameter sweep (over-/under-relaxation study)
- Noise-controlled initialization
- Reproducible (fixed seed)
- Console summary table
- Designed for large N (10k–500k nodes possible in scalar mode)
- Easy to extend: add gossip topologies, approximate mean estimation, plots, etc.

Quick start examples:
    # Medium scale – good starting point
    python sylvach_vs_pairwise_gossip.py --nodes 10000 --dim 6 --runs 5

    # Fast scaling test (very large n)
    python sylvach_vs_pairwise_gossip.py --nodes 200000 --scalar --runs 8 --max-iter 4000

    # Over-relaxation sweep
    python sylvach_vs_pairwise_gossip.py --nodes 8000 --dim 5 --alpha 0.4,0.7,1.0,1.3,1.7 --runs 6

Expected observations at n=10,000:
- Pairwise gossip               → hundreds to low thousands iterations (≈ O(n log n))
- Sylvach (good α)              → iterations almost independent of n
- Wall-clock: Sylvach more expensive per iter (global .mean()), but fewer iters

Author notes (2026): This is the consolidated master version after several iterations.
                     Ready for GitHub README demo or arXiv supplement.

License: MIT (feel free to use/modify)
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
from tqdm import tqdm


@dataclass
class Config:
    output_dir: str = "sylvach_results"
    n_runs: int = 5
    n_nodes: int = 10000
    dim: int = 6
    scalar_mode: bool = False
    max_iter: int = 1500
    tol: float = 1e-9
    alphas: List[float] = None         # type: ignore
    noise_std: float = 0.4
    seed: int = 43

    def __post_init__(self):
        if self.scalar_mode:
            self.dim = 1
        if self.alphas is None:
            self.alphas = [0.6, 0.8, 1.0, 1.2]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Sylvach vs Random Pairwise Gossip – large-scale benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="sylvach_results")
    parser.add_argument("--nodes", "-n", type=int, default=10000,
                        help="number of nodes")
    parser.add_argument("--dim", "-d", type=int, default=6,
                        help="state dimension (ignored if --scalar)")
    parser.add_argument("--scalar", action="store_true",
                        help="force scalar states (dim=1) – very fast large-n test")
    parser.add_argument("--runs", "-r", type=int, default=5,
                        help="number of independent runs")
    parser.add_argument("--max-iter", type=int, default=1500,
                        help="max iterations per run")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="convergence tolerance")
    parser.add_argument("--alpha", default="0.6,0.8,1.0,1.2",
                        help="comma-separated α values for Sylvach")
    parser.add_argument("--noise", type=float, default=0.4,
                        help="std of initialization noise")
    parser.add_argument("--seed", type=int, default=43,
                        help="random seed for reproducibility")

    args = parser.parse_args()

    try:
        alphas = [float(x.strip()) for x in args.alpha.split(",") if x.strip()]
    except Exception as e:
        parser.error(f"Cannot parse --alpha: {e}")

    return Config(
        output_dir=args.output_dir,
        n_runs=args.runs,
        n_nodes=args.nodes,
        dim=args.dim,
        scalar_mode=args.scalar,
        max_iter=args.max_iter,
        tol=args.tol,
        alphas=alphas,
        noise_std=args.noise,
        seed=args.seed,
    )


# ─── Consensus Methods ───────────────────────────────────────────────────────────


def random_pairwise_gossip(
    states: np.ndarray,
    max_iter: int,
    tol: float,
    dim: int,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """Random pairwise gossip – disjoint pairs each round (approx n/2 pairs)"""
    n = len(states)
    hist = []
    converged = False
    start = time.perf_counter()
    X = states.copy()

    for it in range(max_iter):
        idx = np.random.permutation(n)
        for i in range(0, n - 1, 2):
            a, b = idx[i], idx[i + 1]
            avg = (X[a] + X[b]) / 2
            X[a] = avg
            X[b] = avg

        if dim == 1:
            residual = np.max(np.abs(X - X.mean()))
        else:
            Xbar = X.mean(axis=0)
            residual = np.max(np.linalg.norm(X - Xbar[None], axis=1))

        hist.append(residual)
        if residual < tol:
            converged = True
            break

    duration = time.perf_counter() - start
    return X, converged, {
        "iterations": it + 1 if converged else max_iter,
        "residuals": hist,
        "converged": converged,
        "time_s": duration,
    }


def sylvach_relaxed(
    states: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    dim: int,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """Sylvach: relaxed global-mean update (X += α (mean - X))"""
    n = len(states)
    hist = []
    converged = False
    start = time.perf_counter()
    X = states.copy()

    for it in range(max_iter):
        if dim == 1:
            Xbar = X.mean()
            X += alpha * (Xbar - X)
            residual = np.max(np.abs(X - Xbar))
        else:
            Xbar = X.mean(axis=0)
            X += alpha * (Xbar[None] - X)
            residual = np.max(np.linalg.norm(X - Xbar[None], axis=1))

        hist.append(residual)
        if residual < tol:
            converged = True
            break

    duration = time.perf_counter() - start
    return X, converged, {
        "iterations": it + 1 if converged else max_iter,
        "residuals": hist,
        "converged": converged,
        "time_s": duration,
    }


# ─── Benchmark ───────────────────────────────────────────────────────────────────


class Benchmark:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        np.random.seed(cfg.seed)

    def init_states(self) -> np.ndarray:
        n, d = self.cfg.n_nodes, self.cfg.dim
        scale = self.cfg.noise_std

        if d == 1:
            return np.random.randn(n) * scale

        # Symmetric random matrices
        A = np.random.randn(n, d, d) * scale
        return (A + np.transpose(A, (0, 2, 1))) / 2

    def run(self):
        print(f"\n{'═' * 70}")
        print(f"Sylvach vs Pairwise Gossip Benchmark")
        print(f"  • {self.cfg.n_nodes:,} nodes  •  dim={self.cfg.dim}  •  {self.cfg.n_runs} runs")
        print(f"  • noise σ={self.cfg.noise_std:.2f}  •  tol={self.cfg.tol:.1e}  •  seed={self.cfg.seed}")
        print(f"{'═' * 70}\n")

        methods = {
            "Random Pairwise Gossip": lambda s: random_pairwise_gossip(
                s, self.cfg.max_iter, self.cfg.tol, self.cfg.dim
            )
        }
        for a in self.cfg.alphas:
            name = f"Sylvach α={a:.2f}"
            methods[name] = lambda s, a=a: sylvach_relaxed(
                s, a, self.cfg.max_iter, self.cfg.tol, self.cfg.dim
            )

        results: Dict[str, Dict[str, List[float]]] = {}

        for _ in tqdm(range(self.cfg.n_runs), desc="Runs", ncols=80):
            states = self.init_states()

            for name, func in tqdm(methods.items(), desc="Methods", leave=False, ncols=80):
                _, _, info = func(states)

                if name not in results:
                    results[name] = {
                        "iterations": [],
                        "time_s": [],
                        "converged": [],
                    }

                r = results[name]
                r["iterations"].append(info["iterations"])
                r["time_s"].append(info["time_s"])
                r["converged"].append(info["converged"])

        # Summary table
        print("\nResults summary:")
        print(f"{'Method':26} │ {'mean iter':>10} ± {'std':<5} │ {'mean time':>9} s ± {'std':<5} │ conv rate")
        print("─" * 78)
        for name, r in sorted(results.items()):
            it_mean = np.mean(r["iterations"])
            it_std = np.std(r["iterations"])
            t_mean = np.mean(r["time_s"])
            t_std = np.std(r["time_s"])
            conv = np.mean(r["converged"]) * 100
            print(f"{name:26} │ {it_mean:10.0f} ± {it_std:5.0f} │ {t_mean:9.3f} ± {t_std:5.3f} │ {conv:5.0f}%")

        print(f"\nOutput directory: {self.cfg.output_dir}")
        print("Done.\n")


def main():
    cfg = parse_args()
    print("Configuration:")
    print(cfg)
    benchmark = Benchmark(cfg)
    benchmark.run()


if __name__ == "__main__":
    main()