import time
import numpy as np
from numpy import bool_
from numpy import sqrt
from numpy import int64
from numpy import float64
from numba import njit
from numba import objmode
from numpy.linalg import norm

"""
This module is for symmetric diagonally dominant solvers:
        
        Qx=b, where
        
        Q is symmetric diagonally dominant
        Q = I - ((1-alpha)/(1+alpha))*D^{-1/2} A D^{-1/2}
        b = 2*alpha/(1+alpha) D^{-1/2} e_s
"""


@njit(cache=True)
def sdd_get_opt(n, indptr, indices, degree, source, alpha, eps):
    with objmode(start='f8'):
        start = time.perf_counter()
    xt = np.zeros(n, dtype=float64)
    grad = np.zeros(n, dtype=float64)
    const = (1. - alpha) / (1. + alpha)
    sq_deg = sqrt(degree)
    ind = 0
    while True:
        grad[:] = xt
        for u in np.arange(n):
            val = const * xt[u] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                grad[v] -= val
        grad[source] += -(2. * alpha) / (1. + alpha)
        xt[:] = xt - grad
        err = np.linalg.norm(grad, 1)
        ind += 1
        if err < eps:
            break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    print('get true ppv with error: ', err, 'in ', ind, ' iterations  and ', run_time, 'seconds')
    return xt / sq_deg


@njit(cache=True)
def sdd_global_gd(n, indptr, indices, degree, b, alpha, eps, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()

    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    tmp = np.zeros(n, dtype=float64)
    sq_deg = sqrt(degree)
    rt[:] = sq_deg * b
    eps_vec = eps * degree
    const = (1. - alpha) / (1. + alpha)

    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    while True:
        xt += rt
        tmp[:] = 0.
        for u in np.arange(n):
            val = const * rt[u] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                tmp[v] += val
        rt[:] = tmp

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        if opt_x is not None:
            err = norm(xt / sq_deg - opt_x, 1)
            errs.append(err)
        else:
            errs.append(np.infty)
        opers.append(np.sum(degree))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        gamma_t.append(1.)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------

        if np.sum(eps_vec <= np.abs(rt)) <= 0. or np.abs(errs[-1]) <= 0.:
            break
        if l1_err is not None and errs[-1] <= l1_err:
            break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sq_deg, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_local_gd(n, indptr, indices, degree, b, alpha, eps, opt_x):
    with objmode(start='f8'):
        start = time.perf_counter()

    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    eps_vec = eps * degree
    const = (1. - alpha) / (1. + alpha)

    # queue data structure
    s = np.nonzero(b)[0]
    rt[s[0]] = b[s[0]] * np.sqrt(degree[s[0]])
    queue = np.zeros(n, dtype=int64)
    queue[:len(s)] = s
    q_mark = np.zeros(n, dtype=bool_)
    q_mark[s] = True
    rear = len(s)
    st = np.zeros(n, dtype=int64)
    vl = np.zeros(n, dtype=float64)
    st_len = rear

    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    while True:

        # updates for current iteration from queue
        if st_len < n / 4:
            st[:st_len] = queue[:st_len]
        else:  # use continuous memory
            st[:st_len] = np.nonzero(q_mark)[0]
        vl[:st_len] = rt[st[:st_len]]
        q_mark[st[:st_len]] = False
        rear = 0

        # --- debug ---
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        sq_deg = sqrt(degree)
        num = np.linalg.norm(rt[st[:st_len]] / sq_deg[st[:st_len]], 1)
        dem = np.linalg.norm(rt / sq_deg, 1)
        gamma_t.append(num / dem)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # -------------
        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        if opt_x is not None:
            errs.append(norm(xt / sq_deg - opt_x, 1))
        else:
            errs.append(np.infty)  # fakes
        opers.append(np.sum(degree[st[:st_len]]))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[st[:st_len]]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------

        xt[st[:st_len]] += vl[:st_len]
        rt[st[:st_len]] -= vl[:st_len]
        for ind in range(st_len):
            u = st[ind]
            val = const * vl[ind] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                rt[v] += val
                if not q_mark[v] and eps_vec[v] <= rt[v]:
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
        st_len = rear
        # queue is empty now, quit
        if rear == 0:
            break

    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sq_deg, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_global_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()
    # --- initialization ---
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    sq_deg = sqrt(degree)
    rt[:] = sq_deg * b
    eps_vec = eps * degree
    const = (1. - alpha) / (1. + alpha)

    # results
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    oper = 0.
    with objmode(debug_start='f8'):
        debug_start = time.perf_counter()
    gamma_pre = np.linalg.norm(b, 1)
    gamma = 0.
    with objmode(op_time='f8'):
        op_time += (time.perf_counter() - debug_start)
    while True:

        for u in range(n):
            oper += degree[u]
            gamma += np.abs(rt[u]) / sq_deg[u]
            delta = omega * rt[u]
            xt[u] += delta
            rt[u] -= delta
            val = const * delta / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                rt[v] += val

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        # minimal l1-err meets
        err = norm(xt / sq_deg - opt_x, 1)

        if opt_x is not None:
            errs.append(err)
        else:
            errs.append(np.infty)  # fakes
        opers.append(oper)
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        gamma_t.append(gamma / gamma_pre)
        oper = 0.
        gamma = 0.
        gamma_pre = np.linalg.norm(rt / sq_deg, 1)

        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)

        # all nodes are inactive or get exact solution
        if np.sum(eps_vec <= np.abs(rt)) <= 0. or np.abs(errs[-1]) <= 0.:
            break
        if l1_err is not None and err <= l1_err:
            break

    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sq_deg, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def local_appr(n, indptr, indices, degree, s, alpha, eps, opt_x=None):
    with objmode(start='f8'):
        start = time.perf_counter()
    # --- initialization ---
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    rt[:] = s
    eps_vec = eps * degree
    # ----------------------
    # queue data structure
    st = np.nonzero(s)[0]
    front = int64(0)
    queue = np.zeros(n + 1, dtype=int64)
    queue[:len(st)] = st
    q_mark = np.zeros(n + 1, dtype=bool_)
    q_mark[st] = True
    rear = len(st)
    queue[rear] = n  # iteration flag
    q_mark[n] = True
    rear += 1

    # results
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    oper = 0.
    with objmode(debug_start='f8'):
        debug_start = time.perf_counter()
    gamma_pre = np.linalg.norm(s, 1)
    gamma = 0.
    with objmode(op_time='f8'):
        op_time += (time.perf_counter() - debug_start)

    while True:
        u = queue[front]
        q_mark[u] = False
        front = (front + 1) % n
        if u == n:  # one local iteration
            # ------ debug time ------
            with objmode(debug_start='f8'):
                debug_start = time.perf_counter()
            if opt_x is not None:
                errs.append(norm(xt - opt_x, 1))
            else:
                errs.append(np.infty)  # fakes
            opers.append(oper)
            cd_xt.append(np.count_nonzero(xt))
            cd_rt.append(np.count_nonzero(rt))
            vol_st.append(oper)
            vol_it.append(np.sum(degree[np.nonzero(rt)]))
            gamma_t.append(gamma / gamma_pre)
            oper = 0.
            gamma = 0.
            gamma_pre = np.linalg.norm(rt, 1)
            with objmode(op_time='f8'):
                op_time += (time.perf_counter() - debug_start)
            # ------------------------

            queue[rear] = n
            rear = (rear + 1) % n
            continue

        oper += degree[u]
        gamma += np.abs(rt[u])

        delta = .5 * (1. - alpha) * rt[u]
        xt[u] += alpha * rt[u]
        rt[u] = delta
        for v in indices[indptr[u]:indptr[u + 1]]:
            rt[v] += delta / degree[u]
            if not q_mark[v] and eps_vec[v] <= rt[v]:
                queue[rear] = v
                q_mark[v] = True
                rear = (rear + 1) % n
        # only iteration flag left, quit
        if (rear - front) == 1:
            if len(errs) != 0:
                break
            # ------ debug time ------
            with objmode(debug_start='f8'):
                debug_start = time.perf_counter()
            if opt_x is not None:
                errs.append(norm(xt - opt_x, 1))
            else:
                errs.append(np.infty)  # fakes
            opers.append(oper)
            cd_xt.append(np.count_nonzero(xt))
            cd_rt.append(np.count_nonzero(rt))
            vol_st.append(oper)
            vol_it.append(np.sum(degree[np.nonzero(rt)]))
            gamma_t.append(gamma / gamma_pre)
            with objmode(op_time='f8'):
                op_time += (time.perf_counter() - debug_start)
            break
            # ------------------------
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_local_appr(n, indptr, indices, degree, s, alpha, eps, opt_x=None):
    with objmode(start='f8'):
        start = time.perf_counter()
    # --- initialization ---
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    rt[:] = s
    eps_vec = eps * degree
    # ----------------------
    # queue data structure
    st = np.nonzero(s)[0]
    front = int64(0)
    queue = np.zeros(n + 1, dtype=int64)
    queue[:len(st)] = st
    q_mark = np.zeros(n + 1, dtype=bool_)
    q_mark[st] = True
    rear = len(st)
    queue[rear] = n  # iteration flag
    q_mark[n] = True
    rear += 1

    # results
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    oper = 0.
    with objmode(debug_start='f8'):
        debug_start = time.perf_counter()
    gamma_pre = np.linalg.norm(s, 1)
    gamma = 0.
    with objmode(op_time='f8'):
        op_time += (time.perf_counter() - debug_start)

    while True:
        u = queue[front]
        q_mark[u] = False
        front = (front + 1) % n
        if u == n:  # one local iteration
            # ------ debug time ------
            with objmode(debug_start='f8'):
                debug_start = time.perf_counter()
            if opt_x is not None:
                errs.append(norm(xt / np.sqrt(degree) - opt_x, 1))
            else:
                errs.append(np.infty)  # fakes
            opers.append(oper)
            cd_xt.append(np.count_nonzero(xt))
            cd_rt.append(np.count_nonzero(rt))
            vol_st.append(oper)
            vol_it.append(np.sum(degree[np.nonzero(rt)]))
            gamma_t.append(gamma / gamma_pre)
            oper = 0.
            gamma = 0.
            gamma_pre = np.linalg.norm(rt, 1)
            with objmode(op_time='f8'):
                op_time += (time.perf_counter() - debug_start)
            # ------------------------

            queue[rear] = n
            rear = (rear + 1) % n
            continue

        oper += degree[u]
        gamma += np.abs(rt[u])

        delta = .5 * (1. - alpha) * rt[u]
        xt[u] += alpha * rt[u]
        rt[u] = delta
        for v in indices[indptr[u]:indptr[u + 1]]:
            rt[v] += delta / degree[u]
            if not q_mark[v] and eps_vec[v] <= rt[v]:
                queue[rear] = v
                q_mark[v] = True
                rear = (rear + 1) % n
        # only iteration flag left, quit
        if (rear - front) == 1:
            if len(errs) != 0:
                break
            # ------ debug time ------
            with objmode(debug_start='f8'):
                debug_start = time.perf_counter()
            if opt_x is not None:
                errs.append(norm(xt / np.sqrt(degree) - opt_x, 1))
            else:
                errs.append(np.infty)  # fakes
            opers.append(oper)
            cd_xt.append(np.count_nonzero(xt))
            cd_rt.append(np.count_nonzero(rt))
            vol_st.append(oper)
            vol_it.append(np.sum(degree[np.nonzero(rt)]))
            gamma_t.append(gamma / gamma_pre)
            with objmode(op_time='f8'):
                op_time += (time.perf_counter() - debug_start)
            break
            # ------------------------
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x):
    with objmode(start='f8'):
        start = time.perf_counter()
    # --- initialization ---
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    eps_vec = eps * alpha * degree
    const = (1. - alpha) / (1. + alpha)
    # ----------------------
    # queue data structure
    s = np.nonzero(b)[0]
    rt[s[0]] = b[s[0]] * np.sqrt(degree[s[0]])
    front = int64(0)
    queue = np.zeros(n + 1, dtype=int64)
    queue[:len(s)] = s
    q_mark = np.zeros(n + 1, dtype=bool_)
    q_mark[s] = True
    rear = len(s)
    queue[rear] = n  # iteration flag
    q_mark[n] = True
    rear += 1

    # results
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    oper = np.float64(0.)
    with objmode(debug_start='f8'):
        debug_start = time.perf_counter()
    gamma_pre = np.linalg.norm(b, 1)
    gamma = np.float64(0.)
    with objmode(op_time='f8'):
        op_time += (time.perf_counter() - debug_start)

    while True:
        u = queue[front]
        q_mark[u] = False
        front = (front + 1) % n
        if u == n:  # one local iteration
            # ------ debug time ------
            with objmode(debug_start='f8'):
                debug_start = time.perf_counter()
            sq_deg = sqrt(degree)
            if opt_x is not None:
                errs.append(norm(xt / sq_deg - opt_x, 1))
            else:
                errs.append(np.infty)  # fakes
            opers.append(oper)
            cd_xt.append(np.count_nonzero(xt))
            cd_rt.append(np.count_nonzero(rt))
            vol_st.append(oper)
            vol_it.append(np.sum(degree[np.nonzero(rt)]))
            gamma_t.append(gamma / gamma_pre)
            oper = 0.
            gamma = 0.
            gamma_pre = np.linalg.norm(rt / sq_deg, 1)
            with objmode(op_time='f8'):
                op_time += (time.perf_counter() - debug_start)
            # ------------------------

            queue[rear] = n
            rear = (rear + 1) % n
            continue
        oper += degree[u]
        gamma += np.abs(rt[u])
        delta = omega * rt[u]
        xt[u] += delta
        rt[u] -= delta
        val = const * delta / degree[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            rt[v] += val
            if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                queue[rear] = v
                q_mark[v] = True
                rear = (rear + 1) % n
        # only iteration flag left, quit
        if (rear - front) == 1:
            if len(errs) != 0:
                break
            # ------ debug time ------
            with objmode(debug_start='f8'):
                debug_start = time.perf_counter()
            sq_deg = np.sqrt(degree)
            if opt_x is not None:
                errs.append(norm(xt / sq_deg - opt_x, 1))
            else:
                errs.append(np.infty)  # fakes
            opers.append(oper)
            cd_xt.append(np.count_nonzero(xt))
            cd_rt.append(np.count_nonzero(rt))
            vol_st.append(oper)
            vol_it.append(np.sum(degree[np.nonzero(rt)]))
            gamma_t.append(gamma / gamma_pre)
            with objmode(op_time='f8'):
                op_time += (time.perf_counter() - debug_start)
            break
            # ------------------------
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sqrt(degree), rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_global_heavy_ball(n, indptr, indices, degree, b, alpha, eps, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()
    # ----------------------
    xt_tilde = np.zeros(n, dtype=float64)
    delta_cur = np.zeros(n, dtype=float64)
    delta_pre = np.zeros(n, dtype=float64)
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    sq_deg = sqrt(degree)
    rt[:] = sq_deg * b
    eps_vec = eps * alpha * degree
    const = (1. - alpha) / (1. + alpha)
    tmp_r = np.zeros_like(rt)
    beta1 = (1. + ((1. - sqrt(alpha)) / (1. + sqrt(alpha))) ** 2.)
    beta2 = ((1. - sqrt(alpha)) / (1. + sqrt(alpha))) ** 2

    # ----------------------
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)

    while True:

        delta_cur[:] = beta1 * rt + beta2 * xt_tilde
        xt[:] += delta_cur
        tmp_r[:] = 0.
        for u in np.arange(n):
            val = const * delta_cur[u] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                tmp_r[v] += val
        rt[:] += (tmp_r - delta_cur)
        xt_tilde[:] += (delta_cur - delta_pre)
        delta_pre[:] = delta_cur

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        # minimal l1-err meets
        err = norm(xt / sq_deg - opt_x, 1)
        if opt_x is not None:
            errs.append(err)
        else:
            errs.append(np.infty)  # fakes
        opers.append(np.sum(degree))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        gamma_t.append(1.)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        # all nodes are inactive or get exact solution
        if np.sum(eps_vec <= np.abs(rt)) <= 0. or np.abs(errs[-1]) <= 0.:
            break
        if l1_err is not None and err <= l1_err:
            break

    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sq_deg, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_local_heavy_ball(n, indptr, indices, degree, b, alpha, eps, opt_x):
    with objmode(start='f8'):
        start = time.perf_counter()
    # ----------------------
    xt_tilde = np.zeros(n, dtype=float64)
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    eps_vec = eps * alpha * degree
    const = (1. - alpha) / (1. + alpha)
    beta1 = (1. + ((1. - sqrt(alpha)) / (1. + sqrt(alpha))) ** 2.)
    beta2 = ((1. - sqrt(alpha)) / (1. + sqrt(alpha))) ** 2

    # ----------------------
    # queue data structure
    s = np.nonzero(b)[0]
    rt[s[0]] = b[s[0]] * np.sqrt(degree[s[0]])
    queue = np.zeros(n, dtype=int64)
    queue[:len(s)] = s
    q_mark = np.zeros(n, dtype=bool_)
    q_mark[s] = True
    rear = len(s)
    st1 = np.zeros(n, dtype=int64)
    vl1 = np.zeros(n, dtype=float64)
    st1_len = 0
    st2 = np.zeros(n, dtype=int64)
    vl2 = np.zeros(n, dtype=float64)
    st2_len = rear

    # ----------------------
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)

    while True:

        # updates for current iteration from queue
        if st2_len < n / 4:
            st2[:st2_len] = queue[:st2_len]
        else:  # continuous memory
            st2[:st2_len] = np.nonzero(q_mark)[0]
        # st2[:st2_len] = queue[:st2_len]

        vl2[:st2_len] = beta1 * rt[st2[:st2_len]] + beta2 * xt_tilde[st2[:st2_len]]
        q_mark[st2[:st2_len]] = False

        # --- debug ---
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        sq_deg = sqrt(degree)
        num = np.linalg.norm(rt[st2[:st2_len]] / sq_deg[st2[:st2_len]], 1)
        dem = np.linalg.norm(rt / sq_deg, 1)
        gamma_t.append(num / dem)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # -------------

        rear = 0
        xt[st2[:st2_len]] += vl2[:st2_len]
        rt[st2[:st2_len]] -= vl2[:st2_len]
        for ind in range(st2_len):
            u = st2[ind]
            val = const * vl2[ind] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                rt[v] += val
                if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
        xt_tilde[st2[:st2_len]] += vl2[:st2_len]
        xt_tilde[st1[:st1_len]] -= vl1[:st1_len]

        st1[:st2_len] = st2[:st2_len]
        vl1[:st2_len] = vl2[:st2_len]
        st1_len = st2_len
        st2_len = rear

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        # minimal l1-err meets
        if opt_x is not None:
            err = norm(xt / sq_deg - opt_x, 1)
            errs.append(err)
        else:
            errs.append(np.infty)  # fakes
        opers.append(np.sum(degree[st1[:st1_len]]))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        # queue is empty now, quit
        if rear == 0:
            break

    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sq_deg, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_global_cheby(n, indptr, indices, degree, b, alpha, eps, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()
    # ----------------------
    xt_tilde = np.zeros(n, dtype=float64)
    delta_cur = np.zeros(n, dtype=float64)
    delta_pre = np.zeros(n, dtype=float64)
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    sq_deg = sqrt(degree)
    rt[:] = sq_deg * b
    xt_tilde[:] = rt
    xt[:] = rt
    delta_pre[:] = rt
    eps_vec = eps * degree
    const = (1. - alpha) / (1. + alpha)
    delta_t = const

    tmp_r = np.zeros(n, dtype=float64)
    tmp_r[:] = 0.
    for u in np.arange(n):
        val = const * delta_pre[u] / degree[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            tmp_r[v] += val
    rt[:] = tmp_r

    # ----------------------
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)

    while True:

        delta_t = 1. / (2. / const - delta_t)
        beta = 2. * delta_t / const

        delta_cur[:] = beta * rt + (beta - 1.) * xt_tilde
        xt[:] += delta_cur
        tmp_r[:] = 0.
        for u in np.arange(n):
            val = const * delta_cur[u] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                tmp_r[v] += val
        rt[:] += (tmp_r - delta_cur)
        xt_tilde[:] += (delta_cur - delta_pre)
        delta_pre[:] = delta_cur

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        # minimal l1-err meets
        err = norm(xt / sq_deg - opt_x, 1)
        if opt_x is not None:
            errs.append(err)
        else:
            errs.append(np.infty)  # fakes
        opers.append(np.sum(degree))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        gamma_t.append(1.)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        # all nodes are inactive or get exact solution
        if np.sum(eps_vec <= np.abs(rt)) <= 0. or np.abs(errs[-1]) <= 0.:
            break
        if l1_err is not None and err <= l1_err:
            break

    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sq_deg, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_local_cheby(n, indptr, indices, degree, b, alpha, eps, opt_x):
    with objmode(start='f8'):
        start = time.perf_counter()
    # ----------------------
    xt_tilde = np.zeros(n, dtype=float64)
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    # ----------------------
    # queue data structure
    s = np.nonzero(b)[0]
    rt[s[0]] = b[s[0]] * np.sqrt(degree[s[0]])
    xt_tilde[s[0]] = rt[s[0]]
    xt[s[0]] = rt[s[0]]
    queue = np.zeros(n, dtype=int64)
    queue[:len(s)] = s
    q_mark = np.zeros(n, dtype=bool_)
    q_mark[s] = True
    rear = len(s)
    eps_vec = eps * degree
    const = (1. - alpha) / (1. + alpha)
    delta_t = const

    st1 = np.zeros(n, dtype=int64)
    vl1 = np.zeros(n, dtype=float64)
    st1_len = rear
    st1[:st1_len] = queue[:st1_len]
    vl1[:st1_len] = rt[st1[:st1_len]]
    q_mark[st1[:st1_len]] = False

    rear = 0
    rt[st1[:st1_len]] = 0.
    for ind in range(st1_len):
        u = st1[ind]
        val = const * vl1[ind] / degree[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            rt[v] += val
            if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                queue[rear] = v
                q_mark[v] = True
                rear += 1
    st2 = np.zeros(n, dtype=int64)
    vl2 = np.zeros(n, dtype=float64)
    st2_len = rear
    # ----------------------
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)

    while True:

        delta_t = 1. / (2. / const - delta_t)
        beta = 2. * delta_t / const

        # updates for current iteration from queue
        if st2_len < n / 4:
            st2[:st2_len] = queue[:st2_len]
        else:  # continuous memory
            st2[:st2_len] = np.nonzero(q_mark)[0]
        # st2[:st2_len] = queue[:st2_len]

        vl2[:st2_len] = beta * rt[st2[:st2_len]] + (beta - 1.) * xt_tilde[st2[:st2_len]]
        q_mark[st2[:st2_len]] = False

        # --- debug ---
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        sq_deg = sqrt(degree)
        num = np.linalg.norm(rt[st2[:st2_len]] / sq_deg[st2[:st2_len]], 1)
        dem = np.linalg.norm(rt / sq_deg, 1)
        gamma_t.append(num / dem)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # -------------

        rear = 0
        xt[st2[:st2_len]] += vl2[:st2_len]
        rt[st2[:st2_len]] -= vl2[:st2_len]
        for ind in range(st2_len):
            u = st2[ind]
            val = const * vl2[ind] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                rt[v] += val
                if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
        xt_tilde[st2[:st2_len]] += vl2[:st2_len]
        xt_tilde[st1[:st1_len]] -= vl1[:st1_len]

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        # minimal l1-err meets
        if opt_x is not None:
            err = norm(xt / sq_deg - opt_x, 1)
            errs.append(err)
        else:
            errs.append(np.infty)  # fakes
        opers.append(np.sum(degree[st1[:st1_len]]))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        st1[:st2_len] = st2[:st2_len]
        vl1[:st2_len] = vl2[:st2_len]
        st1_len = st2_len
        st2_len = rear

        # queue is empty now, quit
        if rear == 0:
            break

    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt / sq_deg, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_global_cgm(n, indptr, indices, degree, b, alpha, eps, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()
    xt = np.zeros(n, dtype=float64)
    tmp_v = np.zeros(n, dtype=float64)
    rt = np.zeros_like(xt)
    p = np.zeros_like(xt)
    ap = np.zeros_like(p)
    sq_deg = sqrt(degree)

    eps_vec = eps * sq_deg
    rt[:] = b
    rt_pre = np.dot(rt, rt)
    p[:] = rt  # conjugate direction

    # ----------------------
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    while True:
        if rt_pre <= 0.:
            break
        tmp_v *= 0.
        for u in np.arange(n):
            tmp = p[u] / sq_deg[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                tmp_v[v] += tmp / sq_deg[v]
        ap[:] = p - ((1. - alpha) / (1. + alpha)) * tmp_v
        alpha_t = rt_pre / np.dot(p, ap)
        xt[:] = xt + alpha_t * p
        rt[:] = rt - alpha_t * ap
        r_cur = np.dot(rt, rt)
        beta_t = r_cur / rt_pre
        rt_pre = r_cur
        p[:] = rt + beta_t * p

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        if opt_x is not None:
            err = norm(xt - opt_x, 1)
        else:
            err = np.infty
        if opt_x is not None:
            errs.append(err)
        else:
            errs.append(np.infty)  # fakes
        opers.append(np.sum(degree))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        gamma_t.append(1.)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        # all nodes are inactive or get exact solution
        if np.sum(eps_vec <= np.abs(rt)) <= 0. or np.abs(errs[-1]) <= 0.:
            break
        # minimal l1-err meets
        if l1_err is not None and err <= l1_err:
            break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_local_ista(n, indptr, indices, degree, b, alpha, eps, rho_tilde, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()
    # queue to maintain active nodes per-epoch
    queue = np.zeros(n, dtype=int64)
    q_mark = np.zeros(n, dtype=bool_)
    rear = int64(0)
    # approximated solution
    xt = np.zeros(n, dtype=float64)
    delta_xt = np.zeros(n, dtype=float64)
    grad_xt = np.zeros(n, dtype=float64)
    # initialize to avoid redundant calculation
    sq_deg = sqrt(degree)
    rho = rho_tilde / (1. + eps)
    eps_vec = rho * alpha * sq_deg
    # calculate grad of xt = 0 and S0
    # this part is has large run time O(n)

    for u in np.nonzero(b)[0]:
        grad_xt[u] = -(1. + alpha) * b[u] / 2.
        if (xt[u] - grad_xt[u]) >= eps_vec[u]:
            queue[rear] = u
            rear = rear + 1
            q_mark[u] = True

    # results
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    while True:
        oper = 0
        st = queue[:rear]
        delta_xt[st] = -(grad_xt[st] + eps_vec[st])
        xt[st] = (xt[st] - grad_xt[st]) - eps_vec[st]

        # --- debug ---
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        num = np.linalg.norm(grad_xt[st[:rear]], 1)
        dem = np.linalg.norm(grad_xt, 1)
        gamma_t.append(num / dem)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # -------------

        for u in st:
            grad_xt[u] += .5 * (1. + alpha) * delta_xt[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sq_deg[v] * sq_deg[u]
                grad_xt[v] -= .5 * (1. - alpha) * delta_xt[u] / demon
                # new active nodes added into st
                if not q_mark[v] and (xt[v] - grad_xt[v]) >= eps_vec[v]:
                    queue[rear] = v
                    rear = rear + 1
                    q_mark[v] = True
            oper += degree[u]

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        if opt_x is not None:
            errs.append(norm(xt - opt_x, 1))
        else:
            errs.append(np.infty)  # fakes
        opers.append(oper)
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(grad_xt))
        vol_st.append(oper)
        vol_it.append(np.sum(degree[np.nonzero(grad_xt)]))
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------

        st = queue[:rear]
        cond = np.max(np.abs(-grad_xt[st] / degree[st]))
        if cond <= (1. + eps) * rho:
            break
        # minimal l1-err meets
        if l1_err is not None and errs[-1] <= l1_err:
            break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt, grad_xt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def sdd_local_fista(n, indptr, indices, degree, b, alpha, eps, rho_tilde, mome_fixed, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()
    # queue to maintain active nodes per-epoch
    queue = np.zeros(n, dtype=np.int64)
    q_mark = np.zeros(n, dtype=np.bool_)
    q_len = np.int64(0)
    # approximated solution
    qt = np.zeros(n, dtype=np.float64)
    yt = np.zeros(n, dtype=np.float64)
    grad_yt = np.zeros(n, dtype=np.float64)
    # initialize to avoid redundant calculation
    sq_deg = sqrt(degree)
    rho = rho_tilde / (1. + eps)
    eps_vec = rho * alpha * sq_deg
    for u in np.nonzero(b)[0]:
        grad_yt[u] = -(1. + alpha) * b[u] / 2.
        if (qt[u] - grad_yt[u]) >= eps_vec[u]:
            queue[q_len] = u
            q_len += 1
            q_mark[u] = True

    # results
    errs = []
    opers = []
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    # parameter for momentum
    t1 = 1
    beta = (1. - np.sqrt(alpha)) / (1. + np.sqrt(alpha))
    while True:
        for ind in range(q_len):
            q_mark[queue[ind]] = False
        rear = 0
        oper = 0.
        # --- debug ---
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        num = np.linalg.norm(grad_yt[queue[:q_len]], 1)
        dem = np.linalg.norm(grad_yt, 1)
        gamma_t.append(num / dem)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # -------------
        for ind in range(q_len):
            u = queue[ind]
            if (yt[u] - grad_yt[u]) >= eps_vec[u]:
                delta_qi = yt[u] - grad_yt[u] - eps_vec[u] - qt[u]
            elif (yt[u] - grad_yt[u]) <= - eps_vec[u]:
                delta_qi = yt[u] - grad_yt[u] + eps_vec[u] - qt[u]
            else:
                delta_qi = -qt[u]
            qt[u] += delta_qi
            if mome_fixed:
                delta_yi = qt[u] + beta * delta_qi - yt[u]
            else:
                t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
                beta = (t1 - 1.) / t_next
                delta_yi = qt[u] + beta * delta_qi - yt[u]
                t1 = t_next

            yt[u] += delta_yi
            grad_yt[u] += .5 * (1. + alpha) * delta_yi
            for j in indices[indptr[u]:indptr[u + 1]]:
                demon = sq_deg[j] * sq_deg[u]
                ratio = .5 * (1 - alpha) / demon
                grad_yt[j] += (- ratio * delta_yi)
                if not q_mark[j] and np.abs(grad_yt[j]) > eps_vec[j] * (1. + eps):
                    queue[rear] = j
                    rear += 1
                    q_mark[j] = True
            oper += degree[u]
        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        if opt_x is not None:
            errs.append(norm(yt - opt_x, 1))
        else:
            errs.append(np.infty)  # fakes
        opers.append(oper)
        cd_xt.append(np.count_nonzero(yt))
        cd_rt.append(np.count_nonzero(grad_yt))
        vol_st.append(oper)
        vol_it.append(np.sum(degree[np.nonzero(grad_yt)]))
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        q_len = rear
        if q_len == 0:
            break
        # minimal l1-err meets
        if l1_err is not None and errs[-1] <= l1_err:
            break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return yt, grad_yt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


@njit(cache=True)
def __apgd(indptr, indices, sqrt_deg, alpha, queue, rear, x0, t, s, rho):
    at_pre = 0.
    at = 1.
    yt = np.copy(x0)
    zt = np.copy(x0)
    kappa = 1. / alpha
    st = queue[:rear]
    num_oper = 0
    for _ in np.arange(t):
        at_next = at_pre + at
        xt = (at_pre / at_next) * yt + (at / at_next) * zt
        coeff_1 = (kappa - 1. + at_pre) / (kappa - 1. + at_next)
        coeff_2 = at / (kappa - 1. + at_next)
        # calculate the gradient
        grad_xt = alpha * (rho * sqrt_deg - s / sqrt_deg)
        for u in st:
            grad_xt[u] += .5 * (1. + alpha) * xt[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sqrt_deg[v] * sqrt_deg[u]
                grad_xt[v] -= .5 * (1. - alpha) * xt[u] / demon
            num_oper += sqrt_deg[u] ** 2.
        tmp_zt = coeff_1 * zt + coeff_2 * (xt - grad_xt / alpha)
        zt = np.zeros(len(xt))
        for u in st:
            if tmp_zt[u] > 0:
                zt[u] = tmp_zt[u]
        yt = (at_pre / at_next) * yt + (at / at_next) * zt
        at = at_next * (2. * kappa / (2. * kappa + 1. - np.sqrt(1. + 4 * kappa)) - 1.)
        at_pre = at_next
    return yt, num_oper


@njit(cache=True)
def sdd_local_aspr(n, indptr, indices, degree, s, alpha, eps, rho, opt_x):
    xt = np.zeros(n, dtype=np.float64)
    queue = np.zeros(n, dtype=np.int64)
    q_mark = np.zeros(n, dtype=np.bool_)
    sqrt_deg = np.sqrt(degree)
    rear = 0
    new_counts = 0
    for i in np.arange(n):
        if s[i] > rho * degree[i]:
            queue[rear] = i
            q_mark[i] = True
            rear += 1
            new_counts += 1
    l1_error = []
    nonzero_list = []
    st_list = []
    num_opers = []
    num_oper = 0.
    while new_counts != 0:
        # calculate the gradient
        grad_xt = alpha * (rho * sqrt_deg - s / sqrt_deg)
        st = queue[:rear]

        for u in st:
            grad_xt[u] += .5 * (1. + alpha) * xt[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sqrt_deg[v] * sqrt_deg[u]
                grad_xt[v] -= .5 * (1. - alpha) * xt[u] / demon
            num_oper += degree[u]

        delta_t = np.sqrt((eps * alpha) / (1. + rear))
        eps_t_hat = (alpha * (delta_t ** 2.)) / 2.
        num = (1. - alpha) * np.sum(grad_xt[st] ** 2.)
        dem = 2. * eps_t_hat * (alpha ** 2.)
        t = 1. + np.ceil(2. * np.sqrt(1. / alpha) * np.log(num / dem))
        xt_bar, num_oper_ = __apgd(
            indptr, indices, sqrt_deg, alpha, queue, rear, xt, t, s, rho)
        num_oper += num_oper_
        for u in st:
            if (xt_bar[u] - delta_t) > 0.:
                xt[u] = xt_bar[u] - delta_t
            else:
                xt[u] = 0.
        # calculate the gradient
        grad_xt = alpha * (rho * sqrt_deg - s / sqrt_deg)
        for u in st:
            grad_xt[u] += .5 * (1. + alpha) * xt[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sqrt_deg[v] * sqrt_deg[u]
                grad_xt[v] -= .5 * (1. - alpha) * xt[u] / demon

        st_count_old = rear
        for i in np.arange(n):
            if grad_xt[i] < 0. and not q_mark[i]:
                queue[rear] = i
                rear += 1
                q_mark[i] = True
        new_counts = rear - st_count_old
        if opt_x is not None:
            nonzero_list.append(np.count_nonzero(xt))
            l1_error.append(np.linalg.norm(xt - opt_x, 1))
            st_list.append(rear)
            num_opers.append(num_oper)
    return xt, l1_error, nonzero_list, st_list, num_opers


def test_sdd_gd(dataset="ogbn-papers100M"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)

    eps = 1e-8
    re_ = sdd_local_gd(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print(errs[-1])
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="Local-SOR")
    ax[1].plot(gamma_t, label="Local-SOR")

    re_ = sdd_global_gd(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=errs[-1])
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="SOR")
    ax[1].plot(gamma_t, label="SOR")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_sdd_sor(dataset="com-dblp"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)

    omega = 1.
    eps = 1e-19
    re_ = sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x=opt_x)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="Local-SOR")
    ax[1].plot(gamma_t, label="Local-SOR")

    re_ = sdd_global_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x=opt_x, l1_err=errs[-1])
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="SOR")
    ax[1].plot(gamma_t, label="SOR")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_sdd_heavy_ball(dataset="com-dblp"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)
    print(np.linalg.norm(sqrt(degree) * opt_x, 1))
    eps = 1e-9

    re_ = sdd_local_heavy_ball(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print(np.linalg.norm(xt - opt_x, 1), len(errs))
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="Local-HB")
    ax[1].plot(gamma_t, label="Local-HB")
    print(len(errs), errs[-1])
    re_ = sdd_global_heavy_ball(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=errs[-1])
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print(np.linalg.norm(xt - opt_x, 1), len(errs))
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="HB")
    ax[1].plot(gamma_t, label="HB")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_sdd_cheby(dataset="com-dblp"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)
    print(np.linalg.norm(sqrt(degree) * opt_x, 1))
    eps = 1e-11
    re_ = sdd_local_cheby(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('local-cheby', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))

    ax[0].plot(np.cumsum(opers), np.log10(errs), label="Local-Cheby")
    ax[1].plot(gamma_t, label="Local-Cheby")

    re_ = sdd_global_cheby(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=1e-6 * errs[-1])
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('global-cheby', np.linalg.norm(xt - opt_x, 1), len(errs))

    ax[0].plot(np.cumsum(opers), np.log10(errs), label="Cheby")
    ax[1].plot(gamma_t, label="Cheby")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_sdd_cheby_gd(dataset="com-dblp"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.05  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-5)
    eps = 1. / m
    re_ = sdd_local_cheby(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('local-cheby', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))
    print(np.sum(opers))

    ax[0].plot(np.cumsum(opers), np.log(errs), label="LocCH")
    ax[1].plot(gamma_t, label="LocCH")

    re_ = sdd_local_gd(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('local-gd', np.linalg.norm(xt - opt_x, 1), len(errs))
    print(np.sum(opers))

    ax[0].plot(np.cumsum(opers), np.log(errs), label="LocGD")
    ax[1].plot(gamma_t, label="LocGD")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_sdd_cgm(dataset="com-dblp"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)
    print(np.linalg.norm(sqrt(degree) * opt_x, 1))
    eps = 1e-7
    re_ = sdd_global_cgm(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=1e-9)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('global-cgm', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))

    ax[0].plot(np.cumsum(opers), np.log10(errs), label="CGM")
    ax[1].plot(gamma_t, label="CGM")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_sdd_ista(dataset="com-dblp"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)
    print(np.linalg.norm(sqrt(degree) * opt_x, 1))

    eps = 1e-9
    re_ = sdd_local_sor(
        n, indptr, indices, degree, b, alpha, eps, 1., opt_x)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('local-sor', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="SOR")
    ax[1].plot(gamma_t, label="SOR")
    assert errs[-1] > 0
    re_ = sdd_local_ista(
        n, indptr, indices, degree, b, alpha, .1, eps, opt_x, errs[-1])
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('local-ista', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="ISTA")
    ax[1].plot(gamma_t, label="ISTA")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_sdd_fista(dataset="com-dblp"):
    from utils import load_graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n, m, indptr, indices, degree = load_graph(dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)
    print(np.linalg.norm(sqrt(degree) * opt_x, 1))

    eps = 1e-9
    re_ = sdd_local_ista(
        n, indptr, indices, degree, b, alpha, .1, eps, opt_x, None)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('local-ista', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="SOR")
    ax[1].plot(gamma_t, label="SOR")
    assert errs[-1] > 0
    re_ = sdd_local_fista(
        n, indptr, indices, degree, b, alpha, .1, eps, True, opt_x, None)
    xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    print('local-fista', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))
    ax[0].plot(np.cumsum(opers), np.log10(errs), label="FISTA")
    ax[1].plot(gamma_t, label="FISTA")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def test_opt_ppv(dataset="com-lj"):
    from utils import load_graph
    n, m, indptr, indices, degree = load_graph(dataset=dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    for eps in np.logspace(-5., -1., num=5, base=10):
        opt_ppv = sdd_get_opt(n, indptr, indices, degree, source, alpha, eps / m)
        l1_error = 1. - np.linalg.norm(opt_ppv * np.sqrt(degree), 1)
        print(f'l1-errr of ppv: {l1_error:.6e} with eps: {eps / m:.6e}')


def test_local_appr(dataset="com-dblp"):
    from utils import load_graph
    n, m, indptr, indices, degree = load_graph(dataset=dataset)
    source = 0  # source node
    b = np.zeros(n, dtype=np.float64)
    alpha = 0.1  # dumping factor
    b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
    aver1 = []
    aver2 = []
    for eps in [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]:
        opt_ppv = sdd_get_opt(n, indptr, indices, degree, source, alpha, eps / m)
        s = np.zeros_like(b)
        s[source] = 1.
        re_ = local_appr(n, indptr, indices, degree, s, alpha, eps, opt_x=opt_ppv)
        aver1.append(re_[-2] - re_[-1])
        print(re_[-2] - re_[-1], re_[2][-1])
        mu = (1. - alpha) / (1. + alpha)
        opt_omega = 1. + (mu / (1. + np.sqrt(1. - mu ** 2.))) ** 2.
        re_ = sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, opt_omega, opt_x=opt_ppv)
        print(re_[-2] - re_[-1], re_[2][-1])
        aver2.append(re_[-2] - re_[-1])
    print(np.mean(aver1), np.mean(aver2))


def main(dataset="com-dblp"):
    test_local_appr(dataset="com-dblp")


if __name__ == '__main__':
    main()
