import numpy as np
from numba import njit
from numpy.linalg import norm
from numpy import float64
from numpy import bool_
from numpy import sqrt
from numpy import int64


def appr_local_gd(n, indptr, indices, degree, b, alpha, eps, opt_x):
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    sq_deg = sqrt(degree)
    eps_vec = eps * sq_deg
    rt[:] = b
    const = (1. - alpha) / (1. + alpha)
    delta_st = np.nonzero(b)[0]

    rear = 0
    queue = np.zeros(n, dtype=int64)
    q_mark = np.zeros(n, dtype=bool_)

    errs = [norm(xt - opt_x, 1) if opt_x is not None else 0.]
    opers = [0.]
    while True:
        delta_vl = np.zeros(len(delta_st), dtype=float64)
        delta_vl[:] = rt[delta_st]
        xt[delta_st] += delta_vl
        rt[delta_st] -= delta_vl
        opers.append(len(delta_st))
        for u, val in zip(delta_st, delta_vl):
            val = const * val / sq_deg[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                rt[v] += val / sq_deg[v]
                if not q_mark[v] and eps_vec[v] <= rt[v]:
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
            opers[-1] += degree[u]
        errs.append(norm(xt - opt_x, 1) if opt_x is not None else 0.)
        if rear == 0:
            break
        # updates for next
        delta_st = np.zeros(rear, dtype=int64)
        delta_st[:] = queue[:rear]
        front = 0
        while rear != front:
            q_mark[queue[front]] = False
            front += 1
        rear = 0
    return xt, errs, opers


def sdd_local_gd(n, indptr, indices, degree, b, alpha, eps, opt_x):
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    rt[:] = b
    sq_deg = sqrt(degree)
    eps_vec = eps * sq_deg
    const = (1. - alpha) / (1. + alpha)
    delta_st = np.nonzero(b)[0]

    rear = 0
    queue = np.zeros(n, dtype=int64)
    q_mark = np.zeros(n, dtype=bool_)

    errs = [norm(xt - opt_x, 1) if opt_x is not None else 0.]
    opers = [0.]
    while True:
        delta_vl = np.zeros(len(delta_st), dtype=float64)
        delta_vl[:] = rt[delta_st]
        xt[delta_st] += delta_vl
        rt[delta_st] -= delta_vl
        opers.append(len(delta_st))
        for u, val in zip(delta_st, delta_vl):
            val = const * val / sq_deg[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                rt[v] += val / sq_deg[v]
                if not q_mark[v] and eps_vec[v] <= rt[v]:
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
            opers[-1] += degree[u]
        errs.append(norm(xt - opt_x, 1) if opt_x is not None else 0.)
        if rear == 0:
            break
        # updates for next
        delta_st = np.zeros(rear, dtype=int64)
        delta_st[:] = queue[:rear]
        front = 0
        while rear != front:
            q_mark[queue[front]] = False
            front += 1
        rear = 0
    return xt, errs, opers
