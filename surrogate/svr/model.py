"""High-level SVR model creation for the adaptive workflow."""

import math
from dataclasses import dataclass

import numpy as np

from .kernel_utils import compute_kernel, expand_theta
from .train_internal import svr_train_internal


@dataclass
class OptimizationState:
    D: np.ndarray
    ne: np.ndarray
    lo: np.ndarray
    up: np.ndarray
    perf: np.ndarray
    nv: int


def svr_model(
    X: np.ndarray,
    Y: np.ndarray,
    hyperparameters: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    covariance: str,
):
    """Optimize SVR hyperparameters via the box-min algorithm implementation."""

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    m, _ = X.shape

    MInput = np.mean(X, axis=0)
    SInput = np.std(X, axis=0, ddof=0)
    SInput[SInput == 0.0] = 1.0
    Xn = (X - MInput) / SInput

    MOutput = np.mean(Y, axis=0)
    SOutput = np.std(Y, axis=0, ddof=0)
    SOutput[SOutput == 0.0] = 1.0
    Yn = (Y - MOutput) / SOutput
    Yn = Yn.flatten()

    input_moment = np.vstack([MInput, SInput])
    output_moment = np.vstack([MOutput, SOutput])

    row_idx, col_idx = np.triu_indices(m, k=1)
    ij = np.column_stack([row_idx, col_idx])
    d = Xn[row_idx, :] - Xn[col_idx, :]

    par = {
        "D": d,
        "ij": ij,
        "Y": Yn,
        "X": Xn,
        "Inputmoment": input_moment,
        "Outputmoment": output_moment,
    }

    init = np.asarray(hyperparameters, dtype=float)
    lo = np.asarray(lb, dtype=float)
    up = np.asarray(ub, dtype=float)

    t, _, _ = _boxmin(init, lo, up, par, covariance)
    model = svr_train_internal(par, t, covariance)
    return model


def _boxmin(t0: np.ndarray, lo: np.ndarray, up: np.ndarray, par: dict, covariance: str):
    t, f, state = _start(t0, lo, up, par, covariance)
    if not math.isinf(f):
        p = len(t)
        kmax = min(max(p, 2), 4)
        for _ in range(kmax):
            th = t.copy()
            t, f, state = _explore(t, f, state, par, covariance)
            t, f, state = _move(th, t, f, state, par, covariance)
    perf = {"nv": state.nv, "perf": state.perf[:, : state.nv]}
    return t, f, perf


def _svr_likelihood(hyp: np.ndarray, par: dict, covariance: str) -> float:
    model = svr_train_internal(par, hyp, covariance)
    kernel1 = model["Kernelmatrix1"]
    kernel = model["Kernelmatrix"]
    parameter = model["parameter"]
    fmp = kernel @ parameter
    bias = model["bias"]
    delta = model["Output"] - fmp - bias

    C = hyp[0]
    epsilon = model["epsilon"]
    idx = np.where(np.abs(delta) > epsilon)[0]
    if idx.size:
        loss = np.abs(delta[idx]) - epsilon
        loss_sum = float(np.sum(loss))
    else:
        loss_sum = 0.0

    sv = model["SV"]
    if sv.size:
        active_regularized_kernel = kernel1[np.ix_(sv, sv)]
        sign, logdet = np.linalg.slogdet(active_regularized_kernel)
        if sign <= 0:
            logdet = np.log(np.finfo(float).tiny)
    else:
        logdet = 0.0
        sv = np.array([])

    numsv = sv.size
    term1 = 0.5 * parameter @ (kernel @ parameter)
    term2 = C * loss_sum
    term3 = kernel.shape[0] * math.log(math.sqrt(2.0 * math.pi / C) + 2.0 * epsilon)
    term4 = 0.5 * (numsv * math.log(C) + logdet)
    return term1 + term2 + term3 + term4


def _start(t0: np.ndarray, lo: np.ndarray, up: np.ndarray, par: dict, covariance: str):
    t = t0.astype(float).copy()
    lo = lo.astype(float)
    up = up.astype(float)
    p = t.size
    D = 2.0 ** (np.arange(1, p + 1) / (p + 2))

    ee = np.where(up == lo)[0]
    if ee.size:
        D[ee] = 1.0
        t[ee] = up[ee]

    ng = np.where((t < lo) | (t > up))[0]
    if ng.size:
        t[ng] = (lo[ng] * up[ng] ** 7) ** (1.0 / 8.0)

    ne = np.where(D != 1.0)[0]
    f = _svr_likelihood(t, par, covariance)
    perf = np.zeros((p + 2, 200 * p))
    perf[:, 0] = np.concatenate([t, [f, 1]])
    state = OptimizationState(D=D, ne=ne, lo=lo, up=up, perf=perf, nv=1)

    if math.isinf(f):
        return t, f, state

    if ng.size > 1:
        d0 = 16.0
        d1 = 2.0
        q = ng.size
        th = t.copy()
        fh = f
        jdom = int(ng[0])
        for j in ng:
            fk = fh
            tk = th.copy()
            DD = np.ones(p)
            DD[ng] = 1.0 / d1
            DD[j] = 1.0 / d0
            alpha = np.min(np.log(lo[ng] / th[ng]) / np.log(DD[ng])) / 5.0
            v = DD ** alpha
            tk = th.copy()
            for _ in range(4):
                tt = tk * v
                ff = _svr_likelihood(tt, par, covariance)
                state.nv += 1
                state.perf[:, state.nv - 1] = np.concatenate([tt, [ff, 1]])
                if ff <= fk:
                    tk = tt
                    fk = ff
                    if ff <= f:
                        t = tt
                        f = ff
                        jdom = int(j)
                else:
                    state.perf[-1, state.nv - 1] = -1
                    break
        if jdom > 0:
            D[[0, jdom]] = D[[jdom, 0]]
            state.D = D
    return t, f, state


def _explore(t: np.ndarray, f: float, state: OptimizationState, par: dict, covariance: str):
    nv = state.nv
    ne = state.ne
    for j in ne:
        tt = t.copy()
        DD = state.D[int(j)]
        atbd = False
        if t[j] == state.up[j]:
            atbd = True
            tt[j] = t[j] / math.sqrt(DD)
        elif t[j] == state.lo[j]:
            atbd = True
            tt[j] = t[j] * math.sqrt(DD)
        else:
            tt[j] = min(state.up[j], t[j] * DD)
        ff = _svr_likelihood(tt, par, covariance)
        nv += 1
        state.perf[:, nv - 1] = np.concatenate([tt, [ff, 2]])
        if ff < f:
            t = tt
            f = ff
        else:
            state.perf[-1, nv - 1] = -2
            if not atbd:
                tt = t.copy()
                tt[j] = max(state.lo[j], t[j] / DD)
                ff = _svr_likelihood(tt, par, covariance)
                nv += 1
                state.perf[:, nv - 1] = np.concatenate([tt, [ff, 2]])
                if ff < f:
                    t = tt
                    f = ff
                else:
                    state.perf[-1, nv - 1] = -2
    state.nv = nv
    return t, f, state


def _move(th: np.ndarray, t: np.ndarray, f: float, state: OptimizationState, par: dict, covariance: str):
    nv = state.nv
    p = t.size
    v = t / th
    if np.allclose(v, 1.0):
        state.D = np.roll(state.D, -1)
        state.D = state.D**0.2
        return t, f, state

    rept = True
    while rept:
        tt = np.minimum(state.up, np.maximum(state.lo, t * v))
        ff = _svr_likelihood(tt, par, covariance)
        nv += 1
        state.perf[:, nv - 1] = np.concatenate([tt, [ff, 3]])
        if ff < f:
            t = tt
            f = ff
            v = v**2
        else:
            state.perf[-1, nv - 1] = -3
            rept = False
        if np.any((tt == state.lo) | (tt == state.up)):
            rept = False
    state.nv = nv
    state.D = np.roll(state.D, -1)
    state.D = state.D**0.25
    return t, f, state
