# SAP-ADMM

This repository uses **SAP-ADMM to solve signal denoising and image denoising problems** from *A Safeguarded Accelerated Proximal ADMM Algorithm for Solving Structured Cardinality Penalized Optimization Problems*, by Wei Bian, Hongyuan Tao, and Fan Wu.

The main experiments use $\alpha=15$ and $t=1.5$. The signal comparison also uses SAP-ADMM with $\alpha=2$, $t=1$, and one restart after $1000$ effective updates. This parameter setting is labeled SAP-ADMM<sup>H</sup> in the output files. The function `sap_admm_halpern` selects this setting.

## Optimization problems

Let $b$ denote a clean signal or a vectorized clean image, $\hat b$ its noisy observation, and $D$ the difference operator. The positive parameters $\lambda_0$ and $\lambda_p$ weight the capped penalty and the mismatch penalty, respectively. For $\nu>0$ and $y\in\mathbb{R}^{s}$, where $s$ is the number of rows of $D$, define

```math
\Phi_\nu(y)=\sum_{i=1}^{s}\min\lbrace 1,\frac{\lvert y_i\rvert}{\nu}\rbrace.
```

The paper denotes this function by $\Phi(y)$. The mismatch penalty below uses the Euclidean norm.

### Piecewise-constant signal denoising

```math
\min_{x\in\mathbb{R}^{n},\,y\in\mathbb{R}^{n-1}}
\frac{\lVert x-\hat b\rVert_1}{n}
+\lambda_p\lVert Dx-y\rVert_2+\lambda_0\Phi_\nu(y).
```

Here $n=1000$ and $D\in\mathbb{R}^{999\times1000}$ is the first-order difference matrix:

```math
D_{i,i}=-1,\qquad D_{i,i+1}=1,\qquad i=1,\ldots,n-1.
```

All other entries are zero. Signal segment lengths and levels are integer draws from $[50,150]$ and $[-5,10]$, respectively, with adjacent levels differing by more than $2$. The last segment is truncated to the signal length.

Gaussian noise with standard deviation $0.5$ is added, followed by additive impulse noise. Each selected entry receives an independent uniform perturbation on $[-5,5]$. We use $50$ trials for each impulse probability:

```math
\pi\in\lbrace 0,0.05,0.10,0.15,0.20\rbrace.
```

### Signal experiment with a general matrix

```math
\min_{x\in\mathbb{R}^{n},\,y\in\mathbb{R}^{n-1}}
\frac{\lVert Ax-\hat b\rVert_1}{m}
+\lambda_p\lVert Dx-y\rVert_2+\lambda_0\Phi_\nu(y).
```

Here $m=800$, $n=1000$, and $D$ is the same signal difference matrix. The entries of $A\in\mathbb{R}^{800\times1000}$ are independent standard Gaussian draws before normalization to $\lVert A\rVert_2=1$.

The observation is generated from $Ab$ with Gaussian standard deviation $0.2$ and impulse probability $0.05$. Impulse perturbations are uniform on $[-5,5]$. Each of the $20$ trial instances is reused for all five safeguard limits:

```math
N_{\mathrm{max}}\in\lbrace 200,300,400,500,600\rbrace.
```

### MNIST image denoising

```math
\min_{x\in\mathbb{R}^{n},\,y\in\mathbb{R}^{2n}}
\frac{\lVert x-\hat b\rVert_1}{n}
+\lambda_p\lVert Dx-y\rVert_2+\lambda_0\Phi_\nu(y).
```

The images have size $28\times28$, so $n=784$. They are vectorized columnwise. The image difference operator is

```math
D=[D_h^{\top},D_v^{\top}]^{\top},\qquad
D_h=L_{28}\otimes I_{28},\qquad D_v=I_{28}\otimes L_{28}.
```

The first $27$ rows of $L_{28}$ contain $-1$ and $1$ forward differences, and its final row is zero. Thus $D\in\mathbb{R}^{1568\times784}$ does not have full row rank.

One fixed clean image is used for each digit $0,\ldots,9$, with $20$ noisy observations per digit. A Bernoulli mask with probability $0.10$ replaces selected pixels by independent uniform samples on $[0,1]$. Gaussian noise with standard deviation $0.2$ is then added. Observations and restored images used for quality evaluation are clipped to $[0,1]$.

## Continuation parameter

The continuation parameter is $\nu_k$, and its lower bound is denoted by $\nu_{\min}$. The counter $k$ counts effective committed proximal outputs. Accepted accelerated outputs advance $k$. Rejected acceleration trials neither advance $k$ nor commit their trial value of $\nu$.

### Signal denoising

For both SAP-ADMM parameter settings in the identity-matrix signal experiment, define

```math
\nu_0=2,\qquad \nu_{\min}=0.99\,\frac{\lambda_0}{3\lambda_p}.
```

The update is

```math
\nu_k=\max\lbrace \nu_{k-1}\gamma_\nu(k),\nu_{\min}\rbrace,\qquad k\ge1,
```

with

```math
\gamma_\nu(k)=
\begin{cases}
0.999,&1\le k\le1500,\\
0.95,&k\ge1501.
\end{cases}
```

### Signal experiment with general $A$

Define

```math
\nu_0=2,\qquad \nu_{\min}=0.99\,\frac{\lambda_0}{3\lambda_p}.
```

The update is

```math
\nu_k=\max\lbrace \nu_{k-1}\gamma_\nu(k),\nu_{\min}\rbrace,\qquad k\ge1,
```

with

```math
\gamma_\nu(k)=
\begin{cases}
0.9995,&1\le k\le4000,\\
0.95,&k\ge4001.
\end{cases}
```

The code condition is `bar_count < 4000` in `SAP_ADMM/src/sap_admm/general_a.py`. Because this counter starts at zero, the first $4000$ effective updates use $0.9995$.

### MNIST image denoising

Define

```math
\nu_0=0.1,\qquad \nu_{\min}=0.999\,\frac{\lambda_0}{3\lambda_p}.
```

The update is

```math
\nu_k=\max\lbrace 0.99\,\nu_{k-1},\nu_{\min}\rbrace
=\max\lbrace \nu_0\,0.99^k,\nu_{\min}\rbrace,\qquad k\ge1.
```

In all three experiments, $\nu_k$ remains at $\nu_{\min}$ after reaching it.

## Paper parameters

| Parameter | Signal denoising | General $A$ | MNIST |
| --- | --- | --- | --- |
| Dimensions | $n=1000$ | $m=800,\ n=1000$ | $n=784$ |
| $L_f$ | $1/\sqrt{n}$ | $1/\sqrt{m}$ | $1/\sqrt{n}$ |
| $\lambda_p$ | $500L_f$ | $500L_f$ | $2L_f$ |
| $\lambda_0$ | $0.016$ | $0.0018$ | $8\times10^{-5}$ |
| $\rho$ | $2/n$ | $0.6/n$ | $0.5/n$ |
| $\beta$ | $6\rho+10^{-8}$ | $6\rho+10^{-8}$ | $10\rho+10^{-8}$ |
| Main $(\alpha,t)$ | $(15,1.5)$ | $(15,1.5)$ | $(15,1.5)$ |
| Additional signal setting | $(\alpha,t)=(2,1)$ with one restart after $1000$ updates | - | - |
| $N_{\mathrm{max}}$ | $500$; $2000$ for illustrative recovery | $200,300,400,500,600$ | $500$ |
| Stopping tolerance $\varepsilon_{\mathrm{stop}}$ | $2\times10^{-4}$ | $2\times10^{-4}$ | $4\times10^{-4}$ |
| Maximum effective updates $K_{\mathrm{max}}$ | $20000$ | $20000$ | $20000$ |
| Additional updates after reaching $\nu_{\min}$ | $50$ | $50$ | $50$ |
| Support threshold $\varepsilon_{\mathrm{supp}}$ | $10^{-6}$ | $10^{-6}$ | - |

For the $(\alpha,t)=(2,1)$ signal setting, the restart occurs after update $1000$ and before update $1001$. The current state becomes the new anchor and the local averaging index is reset. The global effective-update counter and the continuation schedule continue without resetting.

## Initialization and stopping criterion

For identity-matrix signal denoising and MNIST, the initial points are

```math
x^0=\hat b,\quad y^0=\mathbf{0},\quad p^0=x^0,\quad
q^0=Dx^0-y^0,\quad \eta^0=\mathbf{0},\quad \mu^0=\mathbf{0}.
```

For general $A$, use

```math
x^0=A^{\top}\hat b,\quad y^0=\mathbf{0},\quad p^0=Ax^0,\quad
q^0=Dx^0-y^0,\quad \eta^0=\mathbf{0},\quad \mu^0=\mathbf{0}.
```

The returned signal and jump variables are the proximal outputs $\bar x$ and $\bar y$. The code variable `lam` represents the manuscript variable $\eta$.

The stopping residual is

```math
\begin{aligned}
\Delta_k=\max\lbrace
&\lVert\bar p^k-p^k\rVert_2,\lVert\bar q^k-q^k\rVert_2,\lVert\bar x^k-x^k\rVert_2,\\
&\lVert\bar y^k-y^k\rVert_2,\lVert\bar\eta^k-\eta^k\rVert_2,\lVert\bar\mu^k-\mu^k\rVert_2
\rbrace.
\end{aligned}
```

For the paper's initial values, $\nu_0>\nu_{\min}$. Define the first effective update at the lower bound by

```math
k_\nu=\min\lbrace k\ge1:\nu_k=\nu_{\min}\rbrace.
```

Convergence-based termination requires all three conditions:

```math
\Delta_k<\varepsilon_{\mathrm{stop}},\qquad
\nu_k=\nu_{\min},\qquad k-k_\nu\ge50.
```

The first output at the lower bound is excluded from those $50$ additional updates. Independently, the solver terminates when $k=K_{\mathrm{max}}=20000$. Rejected trial evaluations are excluded from $k$ but their computation time is included.

## Run directly in Spyder

Python 3.10 or newer is required. Extract the entire archive before running it.
The outer repository folder contains this README and the four `run_*.py` entry
scripts. The implementation, configurations, and input data are in `SAP_ADMM/`.

Open one of the following files in Spyder and press **F5** (Run file):

| Entry script in the outer folder | Experiment |
| --- | --- |
| `run_signal.py` | Piecewise-constant signal denoising, both SAP-ADMM parameter settings |
| `run_general_a.py` | General matrix A, all five safeguard limits |
| `run_mnist.py` | MNIST image denoising, all ten digits |
| `run_all.py` | All three experiments |

The scripts locate the local source and inputs from their own file paths. A
package installation and a manual change of working directory are unnecessary.
Keep the entry scripts alongside the `SAP_ADMM/` folder. After replacing an
earlier checkout, restart the Spyder console once before running the updated scripts.

Each entry script has three editable settings:

```python
QUICK = False
MODE = "all"
OUTPUT_ROOT = None
```

- `QUICK = False` uses the paper trial counts: 50 per signal noise level, 20
  general-A instances, and 20 noisy observations per digit. Set it to `True`
  for one trial per setting. All model and solver parameters stay the same.
- `MODE = "all"` computes results and saves figures. Use `"compute"` for data
  only or `"plot"` to redraw existing results without recomputing them.
- `OUTPUT_ROOT = None` saves to `SAP_ADMM/results/`, or to
  `SAP_ADMM/quick_results/` in quick mode. A custom relative path is resolved
  inside `SAP_ADMM/`; an absolute path is used as supplied.

Each experiment has its own `signal/`, `general_a/`, or `mnist/` output folder.
It contains `metadata.json`, `results.npz`, `trial_metrics.csv`, `summary.csv`,
and a `figures/` folder with PNG/PDF plots when plotting is enabled. Figures
are saved to files; no interactive plot window is required. Existing outputs
at the same location are replaced when recomputing, so use a new output folder
when changing model or solver parameters.

The required third-party packages are NumPy, SciPy, Matplotlib, Pillow, and
threadpoolctl. If an import reports a missing dependency, open
`install_dependencies.py` in Spyder and run it once with F5. This uses the same
Python interpreter as the active Spyder console. Restart that console after
installation. Dependency installation needs internet access; the experiments
use the clean MNIST images already included in the repository.

The original scripts under `SAP_ADMM/experiments/` also support direct execution.
The three individual `run_*.py` scripts there compute data; use the outer entry
scripts for automatic plotting and editable quick-mode settings.

## Figure style and redrawing saved results

The figures follow the style of the supplied comparison-experiment plots:
Times-style serif fonts, blue solid lines for SAP-ADMM, gray dashed lines for
SAP-ADMM<sup>H</sup>, shared legends, and light dashed grids. Signal plots show the
mean F1 score with a one-sample-standard-deviation band; stacked bars show
the same four F1 categories with percentage labels. The general-A figure
shows mean F1, mean iteration count, and mean MSE from left to right.
MNIST images use a compact grid of square tiles (Original, Noisy, SAP-ADMM).
Only methods implemented in this repository are shown.

Plots are exported as 600-dpi PNG and vector PDF. Times New Roman is used
when available, with Nimbus Roman, Liberation Serif, STIXGeneral, and
DejaVu Serif as fallbacks; no font download is required.

To apply the new style to a completed experiment, open its outer
`run_*.py` entry script in Spyder, set `MODE = "plot"`, and press F5.
Keep `QUICK` and `OUTPUT_ROOT` consistent with the earlier run. To read
results stored in an older checkout, set `OUTPUT_ROOT` to the absolute
path of that checkout's `SAP_ADMM/results/` (or `quick_results/`) folder.
For signal plots, retain both `results.npz` and `recovery_example.npz`.
This redraws the figures without running the solvers or changing saved
measurements. PNG/PDF files with the same names in `figures/` are replaced.
The archived figures under `paper_reference/` are left unchanged.

## Command-line usage

From the outer repository folder, these commands install dependencies and run
all paper experiments without installing this project as a package:

```bash
python -m pip install -r SAP_ADMM/requirements.txt
python run_all.py
```

The package and module interface is also available. Enter the inner project
folder first:

```bash
cd SAP_ADMM
python -m pip install -e .
python -m experiments.run_all --quick --out quick_results
python -m experiments.run_all --out results
```

## Configuration and paper correspondence

Paths in this section and the tables below are relative to the inner `SAP_ADMM/` folder.

| Experiment | Configuration file | Solver entry point | Paper results |
| --- | --- | --- | --- |
| Signal, $(\alpha,t)=(15,1.5)$ | `experiments/config/signal.json` | `sap_admm` | Signal MSE, time, and support recovery |
| Signal, $(\alpha,t)=(2,1)$ with restart | Same configuration; `restart_iter=1000` | `sap_admm_halpern` | SAP-ADMM<sup>H</sup> signal entries |
| General $A$ | `experiments/config/general_a.json` | `sap_admm_generalA` | Sensitivity to $N_{\mathrm{max}}$ |
| MNIST | `experiments/config/mnist.json`, nested `admm` settings | `sap_admm_image` | SAP-ADMM image-quality entries |

For signal and MNIST, `lambda2_capped` is $\lambda_0$, `lambda1_factor` is the multiplier of $L_f$ in $\lambda_p$, `rho_scale` is the numerator in $\rho$, and `max_fail` is $N_{\mathrm{max}}$. Signal `alpha` and `t`, or image `acc_alpha` and `acc_t`, select the main SAP-ADMM parameters. `min_iterations_after_floor=50` specifies the additional effective updates required after reaching $\nu_{\min}$.

General-A settings use the paper parameter names directly. A null `lambda_p` resolves to $500/\sqrt{m}$, and a null `beta` resolves to $6\rho+10^{-8}$.

The supplied table CSVs are in `paper_reference/user_results/`. The experimental TeX and supplied figures are in `paper_reference/`. The runners compute the SAP-ADMM entries; comparison-method implementations are outside this package.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/sap_admm/` | Solvers, operators, metrics, and diagnostics |
| `experiments/` | Experiment and plotting modules |
| `experiments/config/` | Paper configurations |
| `experiments/inputs/` | Fixed MNIST clean images |
| `tests/` | Numerical and stopping-criterion tests |
| `paper_reference/` | Experimental section, figures, and supplied summary CSVs |

## Citation and license

Inside `SAP_ADMM/`, use `CITATION.cff` or `CITATION.bib` to cite the associated manuscript. The source code is distributed under the MIT License in `LICENSE`. Third-party software and image-input information are documented in `THIRD_PARTY_NOTICES.md`.
