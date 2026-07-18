import numpy as np
from numba import njit
from numpy.linalg import norm
from numpy import int64
from numpy import float64
from numpy import sqrt


@njit(cache=True)
def appr_local_sor_queue(n, indptr, indices, degree, s, alpha, eps, omega, opt_x):
    front = int64(0)
    rear = int64(0)
    queue = np.zeros(n + 1, dtype=int64)
    q_mark = np.zeros(n + 1, dtype=np.bool_)
    xt = np.zeros(n, dtype=float64)
    r = np.zeros_like(xt)
    eps_vec = 2. * alpha * eps * degree / (1. + alpha)
    r[:] = 2. * alpha * s / (1. + alpha)
    for u in np.arange(n):
        if eps_vec[u] <= r[u]:
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
    rear = (rear + 1) % n
    queue[rear] = n  # super epoch flag
    q_mark[n] = True
    oper = float64(0.)
    errs = []
    opers = []
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:  # one local super-iteration
            rear = (rear + 1) % n
            queue[rear] = n
            opers.append(oper)
            oper = 0.
            if opt_x is not None:
                errs.append(norm(xt - opt_x, 1))
            else:
                errs.append(0.0)
            continue
        oper += degree[u]
        delta = (1. - alpha) * r[u] * omega / (1. + alpha)
        xt[u] += omega * r[u]
        r[u] = (1. - omega) * r[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += delta / degree[u]
            if not q_mark[v] and eps_vec[v] <= np.abs(r[v]):
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
    return xt, errs, opers

@njit(cache=True)
def appr_local_sor_vec(n, indptr, indices, degree, s, alpha, eps, omega, opt_x):
    xt = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    xt_pre = np.zeros_like(xt)
    sq_deg = np.sqrt(degree)
    const = (2. * alpha) / (1. + alpha)
    r[:] = const * (s / sq_deg)
    eps_vec = const * sq_deg * eps
    errs = []
    opers = []
    while True:
        list_nodes = []
        num_oper = 0.
        for u in np.arange(n):  # needs O(n) time
            if eps_vec[u] <= np.abs(r[u]):
                list_nodes.append(u)
        if len(list_nodes) == 0:
            break
        for uu in list_nodes:
            num_oper += degree[uu]
            delta = (1. - alpha) * r[uu] * omega / (1. + alpha)
            xt[uu] += omega * r[uu]
            r[uu] = (1. - omega) * r[uu]
            for v in indices[indptr[uu]:indptr[uu + 1]]:
                r[v] += delta / (sq_deg[uu] * sq_deg[v])
        opers.append(num_oper)
        if opt_x is not None:
            errs.append(norm(sq_deg * xt - opt_x, 1))
        xt_pre[:] = xt
    return sq_deg * xt, errs, opers
