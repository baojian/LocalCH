import time
import numpy as np
from numpy import sqrt
from algo.sdd_solver import sdd_get_opt
from utils import load_graph
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
import pickle
from utils import data_path
import os
import argparse
from numpy import bool_
from numpy import sqrt
from numpy import int64
from numpy import float64
from numba import njit
from numba import objmode
from numpy.linalg import norm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.system("export OMP_NUM_THREADS=1")
os.system("export OPENBLAS_NUM_THREADS=1")
os.system("export MKL_NUM_THREADS=1")
os.system("export VECLIB_MAXIMUM_THREADS=1")
os.system("export NUMEXPR_NUM_THREADS=1")


@njit(cache=True)
def sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x):
    with objmode(start='f8'):
        start = time.perf_counter()
    # --- initialization ---
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    eps_vec = eps * degree
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
    errs = [1.]
    opers = [0.]
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
    errs = [1.]
    opers = [0.]
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
        num = np.linalg.norm(rt[st2[:st2_len]], 1)
        dem = np.linalg.norm(rt, 1)
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
            err = norm(xt - opt_x, 1)
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
    return xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time


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
    errs = [1.0]
    opers = [0.]
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
            err = norm(np.sqrt(degree) * xt - opt_x, 1)
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


def test_single_method(list_datasets):
    # Set the darkgrid style
    sns.set(font_scale=1.5, rc={'text.usetex': True})
    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale=1.)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    # Customize the background and grid line colors
    for dataset in list_datasets:
        n, m, indptr, indices, degree = load_graph(dataset)
        with plt.rc_context({'axes.facecolor': '0.9',  # Very light gray background
                             'grid.color': 'white',  # White grid lines
                             'grid.linestyle': '-',  # You can customize the line style as well
                             'grid.linewidth': 1.0}):  # Adjust the width of the grid lines as needed
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            for source in range(1):
                b = np.zeros(n, dtype=np.float64)
                alpha = 0.1  # dumping factor
                b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
                opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-6)
                eps = 1. / m
                omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.

                re_1 = sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x=opt_x)
                xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_1
                print('local-sor', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))

                if source == 0:
                    ax.plot(np.cumsum(opers), np.log(errs), c='r', label=r'$\displaystyle \textsc{LocalSOR}$',
                            linewidth=2.5)
                else:
                    ax.plot(np.cumsum(opers), np.log(errs), c='r', linewidth=2.5)

                re_2 = sdd_global_cgm(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=errs[-1])
                xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_2
                print('global-cgm', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))
                if source == 0:
                    ax.plot(np.cumsum(opers), np.log(errs), c='g', linestyle="--",
                            label=r'$\displaystyle \textsc{CGM}$', linewidth=2.5)
                else:
                    ax.plot(np.cumsum(opers), np.log(errs), c='g', linestyle="--", linewidth=2.5)
                pickle.dump([re_1, re_2], open(f'figs/fig1-demo-example-{dataset}-{eps}.pkl', 'wb'))
            ax.set_ylabel(r'$\displaystyle \ln\|\hat{\bm x} - {\bm x}^*\|_1$', fontsize=20)
            ax.set_xlabel(r"Operations", fontsize=20)
            fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
            ax.xaxis.set_major_locator(MaxNLocator(5))  # Approximately 5 grid lines on x-axis
            ax.yaxis.set_major_locator(MaxNLocator(5))
            # n, m, indptr, indices, degree = load_graph('com-dblp')
            ax.legend()
            fig.savefig(f'figs/fig1-demo-example-{dataset}-{eps}-updated.pdf')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            for source in range(1):
                b = np.zeros(n, dtype=np.float64)
                alpha = 0.1  # dumping factor
                b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
                opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-6)
                eps = 1e-1 / m
                omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.

                re_1 = sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x=opt_x)
                xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_1
                print('local-sor', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))

                if source == 0:
                    ax.plot(np.cumsum(opers), np.log(errs), c='r', label=r'$\displaystyle \textsc{LocalSOR}$',
                            linewidth=2.5)
                else:
                    ax.plot(np.cumsum(opers), np.log(errs), c='r', linewidth=2.5)

                re_2 = sdd_global_cgm(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=errs[-1])
                xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_2
                print('global-cgm', np.linalg.norm(xt - opt_x, 1), np.linalg.norm(sqrt(degree) * xt, 1), len(errs))
                if source == 0:
                    ax.plot(np.cumsum(opers), np.log(errs), c='g', linestyle="--",
                            label=r'$\displaystyle \textsc{CGM}$', linewidth=2.5)
                else:
                    ax.plot(np.cumsum(opers), np.log(errs), c='g', linestyle="--", linewidth=2.5)
                pickle.dump([re_1, re_2], open(f'figs/fig1-demo-example-{dataset}-{eps}.pkl', 'wb'))
            ax.set_ylabel(r'$\displaystyle \ln\|\hat{\bm x} - {\bm x}^*\|_1$', fontsize=20)
            ax.set_xlabel(r"Operations", fontsize=20)
            fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
            ax.xaxis.set_major_locator(MaxNLocator(5))  # Approximately 5 grid lines on x-axis
            ax.yaxis.set_major_locator(MaxNLocator(5))
            # n, m, indptr, indices, degree = load_graph('com-dblp')
            ax.legend()
            fig.savefig(f'figs/fig1-demo-example-{dataset}-{eps}-updated.pdf')
            plt.close(fig)


def test_demo_small_scale():
    list_datasets = ['as-skitter', 'cit-patent', 'com-dblp', 'com-lj', 'com-orkut',
                     'com-youtube', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-products', 'ogbn-proteins',
                     'soc-lj1', 'soc-pokec', 'wiki-talk', 'ogbl-ppa', 'wiki-en21',
                     'com-friendster', 'ogbn-papers100M', 'ogb-mag240m']
    list_datasets = ['com-dblp']
    test_single_method(list_datasets)


def demo_sdd(args):
    dataset = args.dataset
    np.random.seed(seed=17)
    n, m, indptr, indices, degree = load_graph(dataset=dataset)
    rand_nodes = np.random.permutation(n)[:50]
    for source in rand_nodes:
        b = np.zeros(n, dtype=np.float64)
        alpha = 0.1  # dumping factor
        b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))

        f_name = f'{data_path()}/{dataset}/sdd-opt-x_source-{source}-alpha-{alpha}.pkl'
        print(f_name)
        if os.path.exists(f_name):
            opt_x = pickle.load(open(f_name, 'rb'))
        else:
            opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-4)
            with open(f_name, "wb") as file:
                pickle.dump(opt_x, file)
        eps = 1. / m
        omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.
        re_1 = sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x=opt_x)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_1
        re_2 = sdd_global_cgm(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=errs[-1])
        pickle.dump([re_1[2:], re_2[2:]],
                    open(f"results/demo-example/demo-example-"
                         f"large-scale_dataset-{dataset}_source-{source}_eps-{eps:.6e}.pkl", 'wb'))


def demo_ppr(args):
    dataset = args.dataset
    np.random.seed(seed=17)
    n, m, indptr, indices, degree = load_graph(dataset=dataset)
    rand_nodes = np.random.permutation(n)[:50]
    for ind, source in enumerate(rand_nodes):
        b = np.zeros(n, dtype=np.float64)
        alpha = 0.1  # dumping factor
        b[source] = 2. * alpha / ((1. + alpha) * sqrt(degree[source]))
        f_name = f'{data_path()}/{dataset}/sdd-opt-x_source-{source}-alpha-{alpha}.pkl'
        if os.path.exists(f_name):
            opt_x = pickle.load(open(f_name, 'rb'))
        else:
            opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-4)
            with open(f_name, "wb") as file:
                pickle.dump(opt_x, file)
        opt_x = np.sqrt(degree) * opt_x
        eps = .1 / n
        omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.
        re_1 = sdd_local_sor(n, indptr, indices, degree, b, alpha, eps, omega, opt_x=opt_x)
        re_2 = sdd_local_cheby(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x)
        err = np.min([re_1[2][-1], re_1[2][-1]])
        re_3 = sdd_global_cgm(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x, l1_err=err)
        result = [n, m, degree, source, re_1[2:], re_2[2:], re_3[2:]]
        pickle.dump(result, open(f"results/demo-figure1/demo-ppr_dataset-{dataset}-{ind}.pkl", 'wb'))


def main(args):
    demo_ppr(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDD Solver')
    parser.add_argument('--dataset', type=str, default='com-dblp',
                        required=False, help='Dataset name')
    parser.add_argument('--alpha', type=float, default=0.1,
                        required=False, help='Alpha value')
    parser.add_argument('--seed', type=int, default=17,
                        required=False, help='Seed for random')
    parser.add_argument('--num_sources', type=int, default=50,
                        required=False, help='Number of source nodes tried')
    parser.add_argument('--output', type=str,
                        default='results/com-dblp_gd_alpha-0.1_seed-17.npz',
                        required=False, help='Save results to file.')
    main(parser.parse_args())
