# LocalCH

Code for **[Iterative Methods via Locally Evolving Set Process](https://arxiv.org/abs/2410.15020)**
(NeurIPS 2024).

> Baojian Zhou, Yifan Sun, Reza Babanezhad Harikandeh, Xingzhi Guo, Deqing Yang, Yanghua Xiao.
> *Iterative Methods via Locally Evolving Set Process.* NeurIPS 2024.

Given the damping factor `α` and precision `ε`, Andersen et al.'s **APPR** is the de-facto local method for
approximating the Personalized PageRank vector, with runtime `Θ(1/(αε))` independent of graph size. This work
observes that **APPR is a local variant of Gauss–Seidel**, and introduces the *locally evolving set process* to
characterize algorithm locality — showing that many standard iterative solvers can be effectively **localized**.

With `vol(S_t)` and `γ_t` the running averages of the active set's volume and residual ratio, we prove
`vol(S_t)/γ_t ≤ 1/ε` and give APPR the tighter runtime bound `Õ(vol(S_t)/(α·γ_t))`, which mirrors actual
performance. Furthermore, when the geometric mean of residual reduction is `Θ(√α)`, the **local Chebyshev method**
runs in `Õ(vol(S_t)/(√α(2−c)))` for some `c ∈ (0,2)`, **without the monotonicity assumption**. Numerical results
show up to a hundredfold speedup over the corresponding standard solvers on real-world graphs.

## Install

```bash
pip install -r requirements.txt
```

⚠️ **`numpy < 2.0` is required** — the code uses `np.infty`, which NumPy 2.0 removed. `numba 0.60.0` is the newest
release compatible with `numpy 1.26`.

## Quickstart

A demo graph (`com-dblp`) ships with the repo, so this runs out of the box:

```python
import numpy as np
from utils import load_graph
from algo.sdd_solver import sdd_get_opt, sdd_local_appr, sdd_local_cheby

n, m, indptr, indices, degree = load_graph('com-dblp')
source, alpha, eps = 0, 0.1, 1e-10

# right-hand side of the normalized SDD system
b = np.zeros(n)
b[source] = 2. * alpha / ((1. + alpha) * np.sqrt(degree[source]))

# ground truth
opt_x = sdd_get_opt(n, indptr, indices, degree, source, alpha, 1e-10)

# APPR takes an indicator vector; the other solvers take b
s = np.zeros(n); s[source] = 1.
res_appr  = sdd_local_appr (n, indptr, indices, degree, s, alpha, eps, opt_x=opt_x)
res_cheby = sdd_local_cheby(n, indptr, indices, degree, b, alpha, eps, opt_x=opt_x)

# every solver returns the same tuple
xt, rt, errs, opers, cd_xt, cd_rt, vol_st, vol_it, gamma_t, run_time, op_time = res_cheby
print('total operations:', np.sum(opers))   # = sum_t vol(S_t)
```

Or run the paper's own experiment scripts:

```bash
python test_sdd_solvers.py          # solver comparison
python test_sdd_diff_alpha.py       # sensitivity to alpha
python test_sdd_diff_large_scale.py # large-scale graphs
python test_spectral_clustering.py  # downstream local clustering
```

## The system being solved

All solvers in `algo/sdd_solver.py` target the symmetric diagonally dominant system

```
Q x = b,   Q = I − ((1−α)/(1+α)) · D^{-1/2} A D^{-1/2}
           b = (2α/(1+α)) · D^{-1/2} e_s
```

Locality comes from restricting each update to an **active set** — nodes whose residual satisfies
`|r_v| ≥ ε · d_v`. Work per iteration is `vol(S_t) = Σ_{u∈S_t} d_u`, and the reported `opers` is exactly this, so
`np.sum(opers)` is the total number of node accesses (counted with multiplicity).

## Layout

```
algo/
  sdd_solver.py   all SDD solvers (numba-jitted): local/global variants of
                  APPR, GD, SOR, heavy-ball, Chebyshev, CGM, ISTA, FISTA, ASPR,
                  plus sdd_get_opt for ground truth
  appr.py         standalone APPR implementations
  local_gd.py     local gradient descent
  sor.py          successive over-relaxation
utils.py          graph loading helpers
test_*.py         experiment scripts from the paper
*.ipynb           notebooks (spectral clustering, ASPR experiments)
datasets/
  com-dblp/       demo graph in CSR .npz form
```

### Solvers in `algo/sdd_solver.py`

| Local | Global | Method |
|---|---|---|
| `local_appr`, `sdd_local_appr` | — | APPR (Andersen et al.) |
| `sdd_local_gd` | `sdd_global_gd` | gradient descent |
| `sdd_local_sor` | `sdd_global_sor` | SOR (optimal `ω = 2(1+α)/(1+√α)²`) |
| `sdd_local_heavy_ball` | `sdd_global_heavy_ball` | heavy-ball |
| **`sdd_local_cheby`** | `sdd_global_cheby` | **local Chebyshev** — the accelerated method |
| `sdd_local_ista`, `sdd_local_fista` | — | ℓ1-regularized proximal gradient |
| `sdd_local_aspr` | — | ASPR |
| — | `sdd_global_cgm` | conjugate gradient |

## Datasets

`com-dblp` is included. The paper additionally uses `as-skitter`, `cit-patent`, `com-lj`, `com-orkut`,
`com-youtube`, `ogbl-ppa`, `ogbn-arxiv`, `ogbn-mag`, `ogbn-products`, `ogbn-proteins`, `soc-lj1`, `soc-pokec`,
`wiki-talk`, `wiki-en21`, `ogbn-papers100M` and `com-friendster` — see `utils.list_datasets()`. These are available
from [SNAP](https://snap.stanford.edu/data/) and [OGB](https://ogb.stanford.edu/); convert them to CSR `.npz` using
the same `{name}_csr-mat.npz` naming as the demo graph.

Set `LOCALCH_ROOT` if you keep datasets outside the repository.

## Known limitations

- The demo functions at the bottom of `algo/appr.py` (`test_appr_queue` and friends) import `algo.gd` / `algo.cg`,
  which are not part of this release, and reference datasets not shipped here. They are leftovers and will not run.
  The maintained entry points are `algo/sdd_solver.py` and the top-level `test_*.py` scripts.
- The first call to any solver pays a one-off **numba JIT compilation** cost (tens of seconds for this module).

## Citation

```bibtex
@inproceedings{zhou2024iterative,
  title     = {Iterative Methods via Locally Evolving Set Process},
  author    = {Zhou, Baojian and Sun, Yifan and Babanezhad Harikandeh, Reza and
               Guo, Xingzhi and Yang, Deqing and Xiao, Yanghua},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

## License

MIT — see [LICENSE](LICENSE).
