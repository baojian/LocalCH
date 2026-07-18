import os.path
import time

import numpy as np
import scipy.sparse as sp


def root_path():
    """Repository root. Override with the LOCALCH_ROOT environment variable if
    you keep datasets outside the repo."""
    return os.environ.get(
        'LOCALCH_ROOT', os.path.dirname(os.path.abspath(__file__))) + os.sep


def data_path():
    return root_path() + 'datasets'


def load_graph(dataset='com-dblp'):
    start = time.time()
    graph_file = f'{data_path()}/{dataset}/{dataset}_csr-mat.npz'
    adj_m = sp.load_npz(graph_file)
    degree = adj_m.sum(1).A.flatten()
    indices = adj_m.indices
    indptr = adj_m.indptr
    n = len(degree)
    m = len(indices)
    load_time = time.time() - start
    print(f'graph-{dataset} loaded in {n} nodes and {m} edges with {load_time:.2f} seconds')
    return n, m, indptr, indices, degree


def load_graph_ids(dataset='com-dblp'):
    graph_file = f'{data_path()}/{dataset}/{dataset}_csr-mat.npz'
    id_file = f'{data_path()}/{dataset}/{dataset}_id-mapping.npz'
    adj_m = sp.load_npz(graph_file)
    degree = adj_m.sum(1).A.flatten()
    indices = adj_m.indices
    indptr = adj_m.indptr
    n = len(degree)
    m = len(indices)
    unique_ids = np.load(id_file)['arr_0']
    return n, m, indptr, indices, degree, unique_ids


def list_datasets():
    datasets = ['as-skitter', 'cit-patent', 'com-dblp', 'com-lj', 'com-orkut',
                'com-youtube', 'ogbl-ppa', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-products',
                'ogbn-proteins', 'soc-lj1', 'soc-pokec', 'wiki-talk', 'wiki-en21',
                'com-friendster', 'ogbn-papers100M', 'ogb-mag240m']
    return datasets


def load_seed_nodes(n, rand_size=50):
    np.random.seed(n)
    x = np.random.permutation(n)
    return x[:rand_size]


def load_seed_nodes_from_cc(n, largest_cc, rand_size=50):
    np.random.random(n)
    largest_cc = np.asarray(largest_cc)
    x = np.random.permutation(len(largest_cc))
    return largest_cc[list(x[:rand_size])]


def _l1pr_ista_cm(n, indptr, indices, degree, s, supp_s, rho, eps, alpha, true_vec):
    """
    s is a distribution. with support supp_s
    use continues memory when rho is very small. for example rho <= 1./m.
    This could be 2 times faster than _appr_ista
    """
    # queue to maintain active nodes per-epoch
    queue = np.zeros(n, dtype=np.int64)
    q_mark = np.zeros(n, dtype=np.bool_)
    rear = np.int64(0)
    # approximated solution
    xt = np.zeros(n, dtype=np.float64)
    delta_xt = np.zeros(n, np.float64)
    grad = np.zeros(n, dtype=np.float64)
    # initialize to avoid redundant calculation
    sqrt_deg = np.zeros(n, dtype=np.float64)
    eps_vec = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sqrt_deg[i] = np.sqrt(degree[i])
        eps_vec[i] = rho * alpha * np.sqrt(degree[i])
    # calculate active nodes for first epoch
    for u in supp_s:
        grad[u] = -alpha * s[u] / sqrt_deg[u]
        if xt[u] - grad[u] >= eps_vec[u]:
            queue[rear] = u
            rear = rear + 1
            q_mark[u] = True
    num_oper = np.float64(0.)
    l1_error = []
    opers_list = []
    if rear == 0:
        # don't do anything
        return xt, l1_error, opers_list
    while True:
        st = queue[:rear]
        delta_xt[st] = -(grad[st] + eps_vec[st])
        xt[st] = xt[st] + delta_xt[st]
        if len(st) < n / 4:
            for ii in st:
                grad[ii] += .5 * (1. + alpha) * delta_xt[ii]
                for v in indices[indptr[ii]:indptr[ii + 1]]:
                    demon = sqrt_deg[v] * sqrt_deg[ii]
                    grad[v] -= .5 * (1. - alpha) * delta_xt[ii] / demon
                    if not q_mark[v] and (xt[v] - grad[v]) >= eps_vec[v]:
                        queue[rear] = v
                        rear = rear + 1
                        q_mark[v] = True
                num_oper += degree[ii]
        else:
            for ii in range(n):
                if not q_mark[ii]:
                    continue
                grad[ii] += .5 * (1. + alpha) * delta_xt[ii]
                for v in indices[indptr[ii]:indptr[ii + 1]]:
                    demon = sqrt_deg[v] * sqrt_deg[ii]
                    grad[v] -= .5 * (1. - alpha) * delta_xt[ii] / demon
                    if not q_mark[v] and (xt[v] - grad[v]) >= eps_vec[v]:
                        queue[rear] = v
                        rear = rear + 1
                        q_mark[v] = True
                num_oper += degree[ii]
        opers_list.append(num_oper)
        if len(true_vec) != 1:
            pr_vec = np.sqrt(degree) * xt
            l1_err = np.linalg.norm(pr_vec - true_vec, 1)
            l1_error.append(l1_err)
        check = -grad[queue[:rear]] > (1 + eps) * eps_vec[queue[:rear]]
        if np.sum(check) == 0:
            break
    st = queue[:rear]
    p = sqrt_deg[st] * xt[st]
    print(len(st))
    return p, l1_error, opers_list


def _l1pr_fpc(n, indptr, indices, degree, s, rho, eps, alpha, momentum_fixed):
    """
    Fixed-Point Continuation
    """
    # approximated solution
    xt = np.zeros(n, dtype=np.float64)
    xt_pre = np.zeros_like(xt)
    yt = np.zeros_like(xt)
    mat_yt_vec = np.zeros_like(xt)
    # initialize to avoid redundant calculation
    sqrt_deg = np.zeros(n, dtype=np.float64)
    eps_vec = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sqrt_deg[i] = np.sqrt(degree[i])
        eps_vec[i] = rho * alpha * np.sqrt(degree[i])
        # calculate active nodes for first epoch
    # parameter for momentum
    t1 = 1
    beta = (1. - np.sqrt(alpha)) / (1. + np.sqrt(alpha))

    gap_list = []
    nonzero_list = []
    nonzero_posi_list = []
    nonzero_nega_list = []

    touched_nodes = np.zeros(n, dtype=np.float64)
    optimal_nodes = np.zeros(n, dtype=np.float64)
    while True:
        for ii in np.arange(n):
            mat_yt_vec[ii] = 0.
            for jj in indices[indptr[ii]:indptr[ii + 1]]:
                demon = sqrt_deg[jj] * sqrt_deg[ii]
                mat_yt_vec[ii] += yt[jj] / demon
        delta_yt = .5 * (1. - alpha) * (yt + mat_yt_vec) + alpha * s / sqrt_deg
        xt = np.sign(delta_yt) * np.maximum(np.abs(delta_yt) - eps_vec, 0.)

        touched_nodes[np.nonzero(xt)[0]] = 1

        pr_vec = np.sqrt(degree) * xt
        pr_vec_pre = np.sqrt(degree) * xt_pre
        gap = np.linalg.norm(pr_vec - pr_vec_pre, 1)
        nonzero = np.count_nonzero(xt)
        nonzero_posi = np.count_nonzero(xt > 0.)
        nonzero_nega = np.count_nonzero(xt < 0.)
        print(nonzero, nonzero_posi, nonzero_nega, gap)
        gap_list.append(gap)
        nonzero_list.append(nonzero)
        nonzero_posi_list.append(nonzero_posi)
        nonzero_nega_list.append(nonzero_nega)
        if gap < eps:
            break
        if momentum_fixed:
            yt = xt + beta * (xt - xt_pre)
        else:
            t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
            beta = (t1 - 1.) / t_next
            yt = xt + beta * (xt - xt_pre)
            t1 = t_next

        xt_pre = xt
    optimal_nodes[np.nonzero(xt)[0]] = 1
    print(np.sum(optimal_nodes), np.sum(touched_nodes))
    expand_opt_nodes = np.zeros(n, dtype=np.float64)
    for ii in np.nonzero(xt)[0]:
        expand_opt_nodes[ii] = 1
        for jj in indices[indptr[ii]:indptr[ii + 1]]:
            expand_opt_nodes[jj] = 1
    total = 0
    for ii in np.nonzero(touched_nodes)[0]:
        if expand_opt_nodes[ii] == 0:
            total += 1
    print(f'{total} is not in expand beta:', beta)

    return pr_vec, gap_list, nonzero_list, nonzero_posi_list, nonzero_nega_list


def test_conjugate_gradient():
    dataset = './datasets/undirected/com-dblp/com-dblp_csr-graph.npz'
    n, indptr, indices, degree = load_graph(dataset=dataset)
    alpha = 0.1  # dumping factor
    root = f'./datasets/undirected/com-dblp'
    from algo.cg import appr_cg
    for s_node in load_seed_nodes(n=n, rand_size=50):
        s = np.zeros(n, dtype=np.float64)
        s[s_node] = 1.
        f_name = f'true-appr_alpha-{alpha}_s-{s_node}.npz'
        opt_x = np.load(f'{root}/{f_name}')['arr_0']
        eps = 1e-08
        xt, l1_error, num_opers = appr_cg(
            n, indptr, indices, degree, s, alpha, eps, opt_x)
        print(l1_error[-1])
        break
