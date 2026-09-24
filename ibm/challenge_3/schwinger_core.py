"""Part 0 answers of the Q-SITE 2026 IBM Schwinger hadron challenge, shared by the Part 1 and Part 2/3 notebooks.

Kept in sync with part0_warmup.ipynb (which shows and verifies the same code).
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import SparsePauliOp, Statevector

from challenge_utils import RXXplus, trotter_step_electric_2q

# model parameters and pre-trained SC-ADAPT-VQE angles (fixed by the challenge)
m = 0.5
g = 0.3
vacuum_prep_theta_OV_1 = 0.30738
vacuum_prep_theta_OV_3 = -0.04059
wave_prep_theta_O_11 = -1.6492
wave_prep_theta_O_22 = -0.3281


# ----- Ex 0.1 -----
def chiral_condensate_observables(L: int) -> list[SparsePauliOp]:
    """chi_j = (-1)^j Z_j + I for j = 0 .. 2L-1 (list of 2L SparsePauliOps on 2L qubits)."""
    n = 2 * L
    obs = []
    for j in range(n):
        obs.append(SparsePauliOp.from_sparse_list([("Z", [j], (-1) ** j), ("", [], 1.0)], num_qubits=n))
    return obs


# ----- Ex 0.2 (2-CNOT circuit of Fig. 5) -----
def RXYplus(theta: float) -> Gate:
    """R^{(XY)}_+(theta) = exp(-i theta/2 (XY + YX))."""
    qc = QuantumCircuit(2)
    qc.s(0); qc.h(0)
    qc.z(1); qc.h(1); qc.s(1)
    qc.cx(0, 1)
    qc.ry(theta, 0); qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(0); qc.sdg(0)
    qc.sdg(1); qc.h(1); qc.z(1)
    return qc.to_gate(label=rf"$R^{{XY}}_{{+}}({theta:.4g})$")


def RXYminus(theta: float) -> Gate:
    """R^{(XY)}_-(theta) = exp(+i theta/2 (XY - YX))."""
    qc = QuantumCircuit(2)
    qc.s(0); qc.h(0)
    qc.z(1); qc.h(1); qc.s(1)
    qc.cx(0, 1)
    qc.ry(-theta, 0); qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(0); qc.sdg(0)
    qc.sdg(1); qc.h(1); qc.z(1)
    return qc.to_gate(label=rf"$R^{{XY}}_{{-}}({theta:.4g})$")


# ----- Ex 0.3 -----
def prep_strong_coupling_vacuum(L: int) -> QuantumCircuit:
    """All sites empty: electrons (even sites) |1>, positrons (odd sites) |0>."""
    qc = QuantumCircuit(2 * L)
    for j in range(0, 2 * L, 2):
        qc.x(j)
    return qc


def vacuum_prep_rotate_OV_1(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O^V_mh(1)) to qc (Fig. 4a of arXiv:2308.04481). [given]"""
    for k in range(L):
        qc.append(RXYminus(theta), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYminus(-theta), [2 * k + 1, 2 * k + 2])
    return qc


def vacuum_prep_rotate_OV_3(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O^V_mh(3)) to qc (Fig. 4b of arXiv:2308.04481, simplified form)."""
    # even-n terms: X-shaped circuits start on qubits 0, 2, 4, ... (inner R+ gates cancel)
    for k in range(L):
        qc.append(RXYplus(-np.pi / 2), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYminus(-theta), [2 * k + 1, 2 * k + 2])
    for k in range(L):
        qc.append(RXYplus(np.pi / 2), [2 * k, 2 * k + 1])
    # odd-n terms: start on qubits 1, 3, 5, ...
    for k in range(L - 1):
        qc.append(RXYplus(-np.pi / 2), [2 * k + 1, 2 * k + 2])
    for k in range(1, L - 1):
        qc.append(RXYminus(theta), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYplus(np.pi / 2), [2 * k + 1, 2 * k + 2])
    return qc


def prep_vacuum(L: int, vacuum_prep_theta_OV_1: float, vacuum_prep_theta_OV_3: float) -> QuantumCircuit:
    """Circuit preparing the 2-step SC-ADAPT-VQE vacuum on L spatial sites (2L qubits)."""
    qc = prep_strong_coupling_vacuum(L)
    qc = vacuum_prep_rotate_OV_1(qc, vacuum_prep_theta_OV_1, L)
    qc = vacuum_prep_rotate_OV_3(qc, vacuum_prep_theta_OV_3, L)
    return qc


# ----- Ex 0.4 -----
def wave_prep_rotate_O_11(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O_mh(1,1)) on qubits L-1, L (Fig. 6, top). [given]"""
    qc.append(RXYminus(theta), [L - 1, L])
    return qc


def wave_prep_rotate_O_22(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O_mh(2,2)) on qubits L-2 .. L+1 (Fig. 6 bottom of arXiv:2401.08044, simplified form)."""
    qc.append(RXYplus(-np.pi / 2), [L - 1, L])
    qc.append(RXYplus(-theta), [L - 2, L - 1])
    qc.append(RXYplus(-theta), [L, L + 1])
    qc.append(RXYplus(np.pi / 2), [L - 1, L])
    return qc


def prep_wave(L: int, vacuum_prep_theta_OV_1: float, vacuum_prep_theta_OV_3: float,
              wave_prep_theta_O_11: float, wave_prep_theta_O_22: float) -> QuantumCircuit:
    """Circuit preparing the initial (centred) wavepacket state on L spatial sites."""
    qc = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
    qc = wave_prep_rotate_O_11(qc, wave_prep_theta_O_11, L)
    qc = wave_prep_rotate_O_22(qc, wave_prep_theta_O_22, L)
    return qc


def prep_vacuum_for_subtraction(L: int, eps: float = 0.9e-4) -> QuantumCircuit:
    """Vacuum circuit with the wavepacket layers at (almost) zero angle, so its structure matches the wavepacket circuit. [given]"""
    qc = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
    qc = wave_prep_rotate_O_11(qc, eps, L)
    qc = wave_prep_rotate_O_22(qc, eps, L)
    return qc


# ----- Ex 0.5 -----
def trotter_step(qc: QuantumCircuit, L: int, time_step: float, m: float, g: float) -> QuantumCircuit:
    """Second-order step: H_kin(t/2)[odd, even]  H_el(t)[Rz layer, barbells]  H_m(t)  H_kin(t/2)[even, odd]  (Fig. 8)."""
    n = 2 * L
    # H_kin over t/2: odd bonds, then even bonds
    for k in range(L - 1):
        qc.append(RXXplus(time_step / 4), [2 * k + 1, 2 * k + 2])
    for k in range(L):
        qc.append(RXXplus(time_step / 4), [2 * k, 2 * k + 1])
    # H_el over t: single-qubit Z part + 4-qubit barbell ZZ part
    for k in range(L // 2 - 1):
        qc.rz(g**2 * time_step, 2 * k)
        qc.rz(0.5 * g**2 * time_step, 2 * k + 1)
    qc.rz(0.5 * g**2 * time_step, L - 2)
    qc.rz(-0.5 * g**2 * time_step, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * time_step, L + 2 * k)
        qc.rz(-g**2 * time_step, L + 2 * k + 1)
    qc = trotter_step_electric_2q(qc, L, time_step, g)
    # H_m over t
    for j in range(n):
        qc.rz((-1) ** j * m * time_step, j)
    # H_kin over t/2, mirrored: even bonds, then odd bonds
    for k in range(L):
        qc.append(RXXplus(time_step / 4), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXXplus(time_step / 4), [2 * k + 1, 2 * k + 2])
    return qc


def evolve_circuits(qc_init: QuantumCircuit, L: int, t: float, m: float, g: float, n_steps: int | None = None):
    """Physics circuit (n_steps forward) and calibration circuit (n_steps/2 forward, n_steps/2 backward)."""
    if n_steps is None:
        n_steps = int(2 * np.ceil(t / 2))
    assert n_steps % 2 == 0, "n_steps must be even for the forward/backward calibration circuit"
    time_step = t / n_steps
    qc = qc_init.copy()
    qc_mitig = qc_init.copy()
    for _ in range(n_steps):
        qc = trotter_step(qc, L, time_step, m, g)
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, time_step, m, g)
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, -time_step, m, g)
    return qc, qc_mitig


# ----- small shared helper -----
def chi_from_state(psi, L: int) -> np.ndarray:
    """<chi_j> for all j from a statevector (numpy array or Statevector)."""
    sv = Statevector(psi)
    return np.array([np.real(sv.expectation_value(o)) for o in chiral_condensate_observables(L)])
