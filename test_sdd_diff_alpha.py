import os
import time
import pickle
import argparse
import multiprocessing
import numpy as np
from utils import load_graph, data_path
from algo.sdd_solver import (sdd_get_opt,
                             sdd_global_gd, sdd_local_gd,
                             sdd_global_sor, sdd_local_sor,
                             sdd_global_heavy_ball, sdd_local_heavy_ball,
                             sdd_local_cheby, sdd_global_cheby,
                             sdd_global_cgm, sdd_local_ista, sdd_local_fista)

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


def single_sdd_solver_large_scale(para):
    ind, alpha, eps, source, graph, algo, dataset = para
    n, m, indptr, indices, degree = graph
    b = np.zeros(n, dtype=np.float64)
    b[source] = 2. * alpha / ((1. + alpha) * np.sqrt(degree[source]))

    opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-3)
    start_time = time.time()
    result_local = None
    # gd sor opt-sor hb cheby cgm ista fista
    if algo == 'gd':
        result_local = sdd_local_gd(
            n, indptr, indices, degree, b, alpha, eps, opt_x)
    elif algo == 'sor':
        omega = 1.
        result_local = sdd_local_sor(
            n, indptr, indices, degree, b, alpha, eps, omega, opt_x)
    elif algo == 'opt-sor':
        opt_omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.
        result_local = sdd_local_sor(
            n, indptr, indices, degree, b, alpha, eps, opt_omega, opt_x)
    elif algo == 'hb':
        result_local = sdd_local_heavy_ball(
            n, indptr, indices, degree, b, alpha, eps, opt_x)
    elif algo == 'cheby':
        result_local = sdd_local_cheby(
            n, indptr, indices, degree, b, alpha, eps, opt_x)
    elif algo == 'cgm':
        opt_omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.
        result_local = sdd_local_sor(
            n, indptr, indices, degree, b, alpha, eps, opt_omega, opt_x)
        l1_err = result_local[2][-1]
        result_local = sdd_global_cgm(
            n, indptr, indices, degree, b, alpha, eps, opt_x, l1_err)
    elif algo == 'ista':
        opt_omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.
        result_local = sdd_local_sor(
            n, indptr, indices, degree, b, alpha, eps, opt_omega, opt_x)
        l1_err = result_local[2][-1]
        result_local = sdd_local_ista(
            n, indptr, indices, degree, b, alpha, eps=.5, rho_tilde=eps, opt_x=opt_x, l1_err=l1_err)
    elif algo == 'fista':
        opt_omega = 2. * (1. + alpha) / (1. + np.sqrt(alpha)) ** 2.
        result_local = sdd_local_sor(
            n, indptr, indices, degree, b, alpha, eps, opt_omega, opt_x)
        l1_err = result_local[2][-1]
        result_local = sdd_local_fista(
            n, indptr, indices, degree, b, alpha, eps=.5, rho_tilde=eps, mome_fixed=True, opt_x=opt_x, l1_err=l1_err)
    else:
        print('Unknown algorithm {}'.format(algo))
        exit(-1)
    run_time = time.time() - start_time
    print('Algorithm: {} on dataset: {}, run-time: {}'.format(algo, dataset, run_time))
    return ind, alpha, eps, source, algo, dataset, result_local[2:]


def get_eps_space_large_scale(alpha, n, m, source_deg):
    eps1 = .5 * (2. * alpha) / (source_deg * (1. + alpha))
    eps2 = 1. / n
    eps3 = 1. / m
    return [eps1, eps2, eps3]


def main(args):
    np.random.seed(args.seed)
    n, m, indptr, indices, degree = load_graph(dataset=args.dataset)
    graph = (n, m, indptr, indices, degree)
    rand_nodes = np.random.permutation(n)[:args.num_sources]
    source = rand_nodes[args.source_id]
    eps_space = get_eps_space_large_scale(args.alpha, n, m, degree[source])
    # do nothing, just for compiling
    single_sdd_solver_large_scale([0, args.alpha, eps_space[0], source, graph, args.algo, args.dataset])
    results = []
    alpha_list = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for alpha in alpha_list:
        result1 = single_sdd_solver_large_scale(
            [args.source_id, alpha, eps_space[1], source, graph, args.algo, args.dataset])
        result2 = single_sdd_solver_large_scale(
            [args.source_id, alpha, eps_space[2], source, graph, args.algo, args.dataset])
        results.append([result1, result2])
    pickle.dump(results, open(args.output, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SDD Solver')
    parser.add_argument('--algo', type=str, default='appr',
                        required=False, help='method for clustering')
    parser.add_argument('--dataset', type=str, default='as-skitter',
                        required=False, help='Dataset name')
    parser.add_argument('--alpha', type=float, default=0.1,
                        required=False, help='Alpha value')
    parser.add_argument('--seed', type=int, default=17,
                        required=False, help='Seed for random')
    parser.add_argument('--num_sources', type=int, default=50,
                        required=False, help='Number of source nodes tried')
    parser.add_argument('--source_id', type=int, default=1,
                        required=False, help='The source node id')
    parser.add_argument('--data_size', type=str, default='medium-scale',
                        required=False, help='The source node id')
    parser.add_argument('--task', type=str, default='diff-alpha',
                        required=False, help='The source node id')
    parser.add_argument('--output', type=str,
                        default='results/com-dblp_gd_alpha-0.1_seed-17.npz',
                        required=False, help='Save results to file.')
    main(parser.parse_args())
