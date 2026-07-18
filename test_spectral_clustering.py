import os
import time
import pickle
import argparse
import multiprocessing
import numpy as np
from utils import load_graph
from algo.appr import appr_queue
from algo.sdd_solver import local_appr
from algo.sdd_solver import sdd_local_gd
from algo.sdd_solver import sdd_local_ista
from algo.sdd_solver import sdd_local_fista
from algo.sdd_solver import sdd_local_cheby
from algo.sdd_solver import sdd_local_sor
from algo.sdd_solver import sdd_local_heavy_ball

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


def single_local_clustering(para):
    ind, alpha, eps, source, graph, algo = para
    n, m, indptr, indices, degree = graph
    s = np.zeros(n, dtype=np.float64)
    s[source] = 1.
    b = np.zeros(n, dtype=np.float64)
    b[source] = 2. * alpha / ((1. + alpha) * np.sqrt(degree[source]))

    if algo == 'appr':
        re_ = local_appr(
            n, indptr, indices, degree, s, alpha, eps, opt_x=None)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
    elif algo == 'opt-sor':
        mu = (1. - alpha) / (1. + alpha)
        opt_omega = 1. + (mu / (1. + np.sqrt(1. - mu ** 2.))) ** 2.
        re_ = sdd_local_sor(
            n, indptr, indices, degree, b, alpha, eps, omega=opt_omega, opt_x=None)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
        xt = xt * np.sqrt(degree)
    elif algo == 'hb':
        re_ = sdd_local_heavy_ball(
            n, indptr, indices, degree, b, alpha, eps, opt_x=None)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
        xt = xt * np.sqrt(degree)
    elif algo == 'cheby':
        re_ = sdd_local_cheby(
            n, indptr, indices, degree, b, alpha, eps, opt_x=None)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
        xt = xt * np.sqrt(degree)
    elif algo == 'fista':
        re_ = sdd_local_fista(
            n, indptr, indices, degree, b, alpha,
            eps=.5, rho_tilde=eps, mome_fixed=False, opt_x=None, l1_err=None)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
        xt = xt * np.sqrt(degree)
    elif algo == 'ista':
        re_ = sdd_local_ista(
            n, indptr, indices, degree, b, alpha,
            eps=.5, rho_tilde=eps, opt_x=None, l1_err=None)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
        xt = xt * np.sqrt(degree)
    else:
        re_ = local_appr(
            n, indptr, indices, degree, s, alpha, eps, opt_x=None)
        xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = re_
        xt = xt * np.sqrt(degree)
    # ---- spectral clustering procedure ----
    supp_xt = np.nonzero(xt)[0]
    sorted_vec = sorted([(np.abs(xt[_] / degree[_]), _) for _ in supp_xt], reverse=True)
    list_conductance = []
    num_cut_edges = 0.
    set_s = dict()
    vol_s = 0.
    for u in [_[1] for _ in sorted_vec]:
        for v in indices[indptr[u]:indptr[u + 1]]:
            # ignore self-loops (indeed happened in soc-lj1)
            if u == v:
                continue
            if v not in set_s:
                num_cut_edges += 1.
            else:
                num_cut_edges -= 1.
        set_s[u] = ''
        vol_s += degree[u]
        vol_t = m - vol_s
        list_conductance.append(num_cut_edges / min(vol_s, vol_t))
        if list_conductance[-1] <= 0.:
            print(list_conductance[-1])
    # for the empty set case, just make it trivial.
    if len(list_conductance) == 0:
        list_conductance = [1.]
    min_conduct = min(list_conductance)
    # ---------------------------------------
    step = int(len(list_conductance) / 100.)
    if step == 0:
        step = 1
    cluster_re = [list_conductance[::step], np.arange(len(sorted_vec))[::step]]
    print(ind, algo, min_conduct, len(sorted_vec), np.sum(degree[supp_xt]), np.sum(opers), run_time)
    return min_conduct, cluster_re, len(sorted_vec), np.sum(degree[supp_xt]), np.sum(opers), run_time, op_time


def main(args):
    np.random.seed(args.seed)
    if args.model == 'fixed':
        n, m, indptr, indices, degree = load_graph(dataset=args.dataset)
        eps = args.eps
        graph = (n, m, indptr, indices, degree)
        rand_nodes = np.random.permutation(n)[:args.num_sources]
        # just for compiling
        single_local_clustering([0, args.alpha, eps, 0, graph, args.algo])
        para_space = []
        for ind, source in enumerate(rand_nodes):
            para_space.append([ind, args.alpha, eps, source, graph, args.algo])
        pool = multiprocessing.Pool(processes=args.num_cpus)
        results = pool.map(func=single_local_clustering, iterable=para_space)
        pool.close()
        pool.join()
        print(np.mean([_[-2] - _[-1] for _ in results]))
        pickle.dump(results, open(args.output, 'wb'))
    elif args.model == 'dynamic':
        n, m, indptr, indices, degree = load_graph(dataset=args.dataset)
        eps = 1. / m
        graph = (n, m, indptr, indices, degree)
        rand_nodes = np.random.permutation(n)[:args.num_sources]
        # just for compiling
        single_local_clustering([0, args.alpha, eps, 0, graph, args.algo])
        para_space = []
        for ind, source in enumerate(rand_nodes):
            para_space.append([ind, args.alpha, eps, source, graph, args.algo])
        pool = multiprocessing.Pool(processes=args.num_cpus)
        results = pool.map(func=single_local_clustering, iterable=para_space)
        pool.close()
        pool.join()
        print(np.mean([_[-2] - _[1] for _ in results]))
        pickle.dump(results, open(args.output, 'wb'))


if __name__ == "__main__":
    datasets = ['as-skitter', 'cit-patent', 'com-dblp', 'com-lj', 'com-orkut',
                'com-youtube', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-products', 'ogbn-proteins',
                'soc-lj1', 'soc-pokec', 'wiki-talk', 'ogbl-ppa', 'wiki-en21']
    parser = argparse.ArgumentParser(description='Local Graph Clustering')
    parser.add_argument('--algo', type=str, default='appr',
                        required=False, help='method for clustering')
    parser.add_argument('--model', type=str, default='fixed',
                        required=False, help='Alpha value')
    parser.add_argument('--dataset', type=str, default='as-skitter',
                        required=False, help='Dataset name')
    parser.add_argument('--alpha', type=float, default=0.1,
                        required=False, help='Alpha value')
    parser.add_argument('--eps', type=float, default=1e-6,
                        required=False, help='Epsilon value')
    parser.add_argument('--seed', type=int, default=17,
                        required=False, help='Epsilon value')
    parser.add_argument('--num_cpus', type=int, default=70,
                        required=False, help='Number of cpus')
    parser.add_argument('--num_sources', type=int, default=10000,
                        required=False, help='Number of source nodes tried')
    parser.add_argument('--output', type=str,
                        default='results/spectral-clustering/test-algo-appr_'
                                'dataset-com-dblp_model-fixed_alpha-0.1_eps-1e-5_seed-17_num-sources-10000.pkl',
                        required=False, help='Save results to file.')
    main(parser.parse_args())
