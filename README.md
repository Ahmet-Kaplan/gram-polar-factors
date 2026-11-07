# Gram-based polar factors in finite and mixed precision

Code and archived results for the manuscript

> **Gram-based polar factors in finite and mixed precision: a sharp criterion, and
> an application to matrix-orthogonalizing optimizers**

submitted to *Numerical Linear Algebra with Applications*.

The archive contains the scripts that produce every number in the manuscript and
the JSON result files they write, together with the two tools that check the
manuscript against those files. It is intended to make the numerical claims
auditable rather than to serve as a library: nothing here is packaged for import.

## Contents

| path | contents |
| --- | --- |
| `experiments/` | the scripts behind the measurements, one per experiment |
| `experiments/baselines.py`, `model.py`, `data.py`, `robust_orth.py`, `csmuon.py` | modules the experiment scripts import: published orthogonalizers, the transformer, the buffer construction |
| `results/` | archived result files, the artefacts of record |
| `results/gpu_server/` | runs of the 10.7M-parameter training comparison of §6 |
| `results/metric_power/` | the twelve runs of the power analysis, one file per (method, seed) |
| `results/momentum_matrices.pt` | the momentum buffers measured over in §3.10: 24 matrices, 41 MB, read directly by six of the scripts |
| `figures/gen_fig4_ablations.py` | script behind one of the archived result files |
| `tools/verify_numbers.py` | resolves the manuscript's statements against the archives |
| `tools/check_tables.py` | compares every table cell against the archive |

## Checking the manuscript

Both tools take the manuscript, which is not part of this archive:

```bash
python tools/verify_numbers.py --paper=/path/to/main.tex --profile=nlaa
python tools/check_tables.py    --paper=/path/to/main.tex
```

`verify_numbers.py` resolves **300** statements of the manuscript against the
archived results and exits non-zero on any disagreement. Most of them compare a
value printed in the manuscript with the same value in a result file, which
detects drift between text and data; the remaining ones check claims that do not
reduce to a number, such as the attainment of both branches of the bound, the
presence of a result the text promises, and the absence of claims withdrawn in
revision. Pass `--profile=long` for the earlier, longer version of the manuscript.

`check_tables.py` parses the tables and compares them cell by cell, so that a
value entered in the wrong row is detected as well as one that disagrees with the
data: **249** cells across the thirteen tables.

Both tools read the archived files rather than recomputing the quantities, so the
checks are independent of the arithmetic of the machine that runs them. The
figure is looked for beside the manuscript, so the manuscript's own `figures/`
directory should accompany it if the figure check is to resolve.

## Regenerating a result

Each experiment script writes its result file to `results/` and prints a summary:

```bash
python experiments/exp_round5.py        # -> results/round5.json
python experiments/exp_aposteriori.py   # -> results/aposteriori.json
```

Several of these run from the archive alone, without a checkpoint or a corpus, and
rerunning `exp_orthogonality_f64.py` reproduces its archived result file unchanged.

## Environment

Python 3.11 with `numpy`, `scipy` and `torch`; the versions used are in
`requirements.txt`. Most scripts run in minutes on a CPU.

Two properties of the environment matter.

**Extended precision.** Four scripts (`exp_aposteriori.py`,
`exp_real_buffers.py`, `exp_mixed_precision.py`, `exp_round5.py`) measure a
rounding error against a reference precision higher than float64. On x86-64,
`numpy.longdouble` is 80-bit and sufficient. On arm64 macOS it is 64-bit with the
same unit roundoff as float64, so the reference would silently be the difference
between two float64 computations. These scripts detect the situation at startup and
refuse to run, rather than report a number of the right magnitude and no meaning.
The result files in `results/` were produced on x86-64.

**CUDA.** The training runs of §6 and the results in `results/gpu_server/` require
a CUDA device. The remaining scripts do not.

## Scope

The archived result files are the artefacts of record, and the checks resolve
against them rather than against a recomputation.

Not included: the manuscript sources, the training corpus and the model
checkpoints. Two consequences. The scripts that train a model — those marked
``GPU'' in Table 13 of the paper — need a CUDA device and the corpus. And
`exp_real_buffers.py` builds its sixteen matrices from a trained model, so
rerunning it needs that checkpoint; the matrices it analyses are nonetheless
archived in `results/momentum_matrices.pt`, which six other scripts read directly.
The large-scale runs of the companion study are outside the scope of this paper
and are not archived here.

## Licence

Code under the MIT licence, see `LICENSE`. The archived result files are released
under the same terms.

## Citing

```bibtex
@article{kaplan2026grampolar,
  author  = {Kaplan, Ahmet},
  title   = {Gram-based polar factors in finite and mixed precision:
             a sharp criterion, and an application to
             matrix-orthogonalizing optimizers},
  journal = {Numerical Linear Algebra with Applications},
  year    = {2026},
  note    = {Submitted}
}
```
