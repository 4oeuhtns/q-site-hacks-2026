"""Micro-benchmark for Quantum Duel workloads. Run on any machine to compare.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u dev/experiments/bench.py

Reports single-core throughput on the three kernels that dominate every learner,
plus how well the box scales across independent processes (which is what the
experiment sweeps actually need).
"""
import os, sys, time, platform, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_quantum_duel_sdk_0_7_2"))
import numpy as np
from scipy.optimize import minimize
from duelkit.recovery4.quantum import circuit_and_jac, dictionary, unitary
from duelkit.recovery4.legacy_learners import DenseMLE

DIC, _ = dictionary(4)

def bench_jac(iters=3000):
    labels = list(DIC[:12]); x = np.linspace(-1, 1, 12)
    t = time.perf_counter()
    for _ in range(iters):
        circuit_and_jac(labels, x, 4)
    return iters / (time.perf_counter() - t)

def bench_expm(iters=400):
    rng = np.random.default_rng(0)
    h = rng.normal(size=(16, 16)) + 1j*rng.normal(size=(16, 16))
    h = h + h.conj().T
    t = time.perf_counter()
    for _ in range(iters):
        w, v = np.linalg.eigh(h)
        (v * np.exp(-1j*w)) @ v.conj().T
    return iters / (time.perf_counter() - t)

def bench_lbfgs(iters=40):
    labels = list(DIC[:12])
    target = unitary([{'pauli': p, 'angle': 0.7} for p in labels], 4)
    def obj(x):
        u, j, *_ = circuit_and_jac(labels, x, 4)
        d = u - target
        return float(np.vdot(d, d).real), np.array([2*np.vdot(d, jj).real for jj in j])
    t = time.perf_counter()
    for k in range(iters):
        minimize(obj, np.full(12, 0.1*(k % 5)), jac=True, method="L-BFGS-B",
                 options={"maxiter": 120})
    return iters / (time.perf_counter() - t)

if __name__ == "__main__":
    print(f"platform : {platform.platform()}")
    print(f"machine  : {platform.machine()}   cpus(logical): {os.cpu_count()}")
    print(f"numpy    : {np.__version__}")
    print()
    for name, fn, unit in (("circuit_and_jac", bench_jac, "calls/s"),
                           ("eigh+expm 16x16", bench_expm, "calls/s"),
                           ("L-BFGS-B fit   ", bench_lbfgs, "fits/s")):
        r = fn()
        print(f"{name}  {r:10.1f} {unit}")
    print()
    print("Higher is better. Compare single-core numbers between machines;")
    print("core count only helps if sweeps are run as parallel processes.")
