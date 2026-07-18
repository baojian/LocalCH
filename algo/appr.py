import numpy as np
from numba import njit
from numpy import int64
from numpy import float64


@njit(cache=True)
def appr_queue(n, indptr, indices, degree, s, alpha, eps, opt_x):
    front = int64(0)
    rear = int64(0)
    q_max = n + 1
    queue = np.zeros(q_max, dtype=int64)
    q_mark = np.zeros(q_max, dtype=np.bool_)
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    eps_vec = eps * degree
    rt[:] = s
    queue[rear % q_max] = n  # super epoch flag
    rear += 1
    q_mark[n] = True
    for u in np.arange(n):
        if eps_vec[u] <= rt[u]:
            queue[rear % q_max] = u
            rear += 1
            q_mark[u] = True
    oper = float64(0.)
    errs = []
    vol_st = []
    vol_it = []
    gamma_t = []
    st_residual = []
    card_it = []
    while True:
        u = queue[front % q_max]
        front += 1
        q_mark[u] = False
        if u == n:  # one local super-iteration
            vol_st.append(oper)
            vol_it.append(np.sum(degree[np.nonzero(rt)[0]]))
            card_it.append(len(np.nonzero(rt)[0]))
            oper = 0.
            demon = np.linalg.norm(rt, 1)
            gamma_t.append(demon)
            st_residual.append(0.)
            if opt_x is not None:
                errs.append(np.linalg.norm(xt - opt_x, 1))
            else:
                errs.append(0.0)
            if (rear - front) == 0:
                break
            queue[rear % q_max] = n
            rear += 1
            q_mark[u] = True
            continue
        oper += degree[u]
        st_residual[-1] += rt[u]
        delta = .5 * (1. - alpha) * rt[u]
        xt[u] += alpha * rt[u]
        rt[u] = delta
        for v in indices[indptr[u]:indptr[u + 1]]:
            rt[v] += delta / degree[u]
            if not q_mark[v] and eps_vec[v] <= rt[v]:
                queue[rear % q_max] = v
                rear += 1
                q_mark[v] = True
    for i in range(len(st_residual)):
        gamma_t[i] = st_residual[i] / gamma_t[i]
    vol_st = vol_st[1:]
    vol_it = vol_it[:-1]
    gamma_t = gamma_t[:-1]
    return xt, errs, vol_st, vol_it, gamma_t, card_it


def test_appr_queue():
    from utils import data_path
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    from algo.sor import appr_local_sor_queue
    root = 'XXXX'
    adj_m = sp.load_npz(root + f'{data_path()}/wiki-en21/wiki-en21_csr-mat.npz')
    degree = adj_m.sum(1).A.flatten()
    indices = adj_m.indices
    indptr = adj_m.indptr
    n = len(degree)
    print(n, len(indices))
    s_node = 0  # source node
    s = np.zeros(n, dtype=np.float64)
    s[s_node] = 1.
    alpha = 0.1  # dumping factor
    eps = 1e-10
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_ylabel("logarithmic Number of operations")
    ax.set_xlabel("iteration t")
    xt1, errs1, opers1 = appr_queue(
        n, indptr, indices, degree, s, alpha, eps, None)
    ax.plot(opers1, label="APPR")
    print('appr', np.sum(opers1), np.sum(xt1))
    mu = (1. - alpha) / (1. + alpha)
    opt_omega = 1. + (mu / (1. + np.sqrt(1. - mu ** 2.))) ** 2.
    xt2, errs2, opers2 = appr_local_sor_queue(
        n, indptr, indices, degree, s, alpha, eps, opt_omega, None)
    ax.plot(opers2, label="APPR-SOR")
    plt.show()
    print('appr-sor', np.sum(opers2), np.sum(xt2))


def test_appr():
    from algo.gd import appr_fixed_point
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    root = 'XXXX'
    adj_m = sp.load_npz(root + 'datasets/com-dblp/com-dblp_csr-mat.npz')
    degree = adj_m.sum(1).A.flatten()
    indices = adj_m.indices
    indptr = adj_m.indptr
    n = len(degree)
    s_node = 0  # source node
    s = np.zeros(n, dtype=np.float64)
    s[s_node] = 1.
    alpha = 0.1  # dumping factor
    eps = 1e-5  # precision parameter

    fig, ax = plt.subplots(1, 3, figsize=(5, 15))
    font = {'size': 12}
    plt.matplotlib.rc('font', **font)
    ax[0].set_ylabel("$\log\|x^{t}-x^*\|$")
    ax[1].set_ylabel("$\log\|x^{t}-x^*\|$")
    ax[2].set_ylabel("Operations per-iteration")
    ax[0].set_xlabel("iteration t")
    ax[1].set_xlabel("# operations")
    ax[2].set_xlabel("iteration t")

    # number of iterations for fixed point iterations
    t = int(np.log(eps * 1e-5) / np.log(1. - alpha))
    opt_x, _, _ = appr_fixed_point(
        n, indptr, indices, degree, s, alpha, t, None)

    xt_fp, l1_error, num_opers = appr_queue(
        n, indptr, indices, degree, s, alpha, eps, opt_x)
    ax[0].plot(np.log10(l1_error), linestyle="-.", label="APPR-FP")
    ax[1].plot(np.cumsum(num_opers), np.log10(l1_error), linestyle="-.", label="APPR-FP")
    ax[2].plot(num_opers, linestyle="-.", label="APPR-FP")
    print('Forward-Push: ', np.linalg.norm(xt_fp - opt_x, 1),
          np.sum(num_opers), np.count_nonzero(xt_fp))

    xt_fp_fast, l1_error, num_opers = appr_queue_fast(
        n, indptr, indices, degree, s, alpha, eps, opt_x)
    ax[0].plot(np.log10(l1_error), linestyle="--", label="APPR-FP-Fast")
    ax[1].plot(np.cumsum(num_opers), np.log10(l1_error), linestyle="--", label="APPR-FP-Fast")
    ax[2].plot(num_opers, linestyle="--", label="APPR-FP-Fast")
    print('Forward-Push-Fast: ', np.linalg.norm(xt_fp_fast - opt_x, 1),
          np.sum(num_opers), np.count_nonzero(xt_fp_fast))

    for i in range(3):
        ax[i].legend()
    plt.show()


def test_appr_cgm():
    from algo.gd import appr_fixed_point
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    from algo.cg import appr_cg
    root = 'XXXX'
    adj_m = sp.load_npz(root + 'datasets/com-dblp/com-dblp_csr-mat.npz')
    degree = adj_m.sum(1).A.flatten()
    indices = adj_m.indices
    indptr = adj_m.indptr
    n = len(degree)
    s_node = 0  # source node
    s = np.zeros(n, dtype=np.float64)
    m = len(indices)
    s[s_node] = 1.
    alpha = 0.15  # dumping factor
    eps = 1.0 / m  # precision parameter

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    font = {'size': 18}
    plt.matplotlib.rc('font', **font)
    ax.set_ylabel("$\log_{10}\|x^{t}-x^*\|$", fontsize=18)
    ax.set_xlabel("# operations", fontsize=18)

    # number of iterations for fixed point iterations
    t = int(np.log(eps * 1e-5) / np.log(1. - alpha))
    opt_x, _, _ = appr_fixed_point(
        n, indptr, indices, degree, s, alpha, t, None)

    xt_fp, l1_error, num_opers = appr_queue(
        n, indptr, indices, degree, s, alpha, eps, opt_x)
    ax.plot(np.cumsum(num_opers), np.log10(l1_error),
            linestyle="-", label="APPR", linewidth=2.5)
    print('Forward-Push: ', np.linalg.norm(xt_fp - opt_x, 1),
          np.sum(num_opers), np.count_nonzero(xt_fp))
    # n, indptr, indices, degree, s, alpha, eps, opt_x
    xt_fp_fast, l1_error, num_opers = appr_cg(
        n, indptr, indices, degree, s, alpha, eps, opt_x)
    ax.plot(np.cumsum(num_opers), np.log10(l1_error),
            linestyle="--", label="CGM", linewidth=2.5)
    print('CGM: ', np.linalg.norm(xt_fp_fast - opt_x, 1),
          np.sum(num_opers), np.count_nonzero(xt_fp_fast))
    ax.legend()
    plt.savefig(f'figs/fwd-push-cgm-alpha-{alpha:e}-eps-{eps:e}.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    test_appr_queue()
