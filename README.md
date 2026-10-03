# mathlib-mcp

**Make GPU math libraries LLM-friendly, and measure whether it helps.**

LLMs write plausible but wrong code for GPU math libraries. They call functions that do not exist, use keyword arguments from an older SciPy, and miss behaviours that differ from NumPy, such as CuPy returning NaN for a singular matrix instead of raising. This project turns the APIs of **CuPy 14.2** and **nvmath-python 1.0** into an LLM-first representation, serves it to coding agents over **MCP**, and includes a **GPU benchmark** to measure the effect on generated code. Those two libraries are the Python front-ends to cuBLAS, cuFFT, cuSOLVER, cuSPARSE, cuDSS and cuTENSOR.

| Part | What it is |
|---|---|
| **API cards** | 2,741 public symbols and 275 API cards extracted from the library *sources* (no GPU or import needed). 42 cards are curated with the CUDA-X backend, pitfalls checked against the source and on a GPU, and an example verified on a GPU. |
| **MCP server** | `search_api`, `get_api_card`, `get_example`, `check_snippet`, plus an `llms.txt` resource. |
| **`check_snippet`** | Static checker that finds hallucinated APIs, wrong or positional-only keywords, too many positional arguments and missing arguments, and suggests the real name. |
| **Benchmark** | 26 GPU tasks across six CUDA-X libraries with NumPy/SciPy references; 17 have a "naive" solution that falls into a documented trap. |

> Status: extraction, curation, MCP server, benchmark and GPU validation are done. Next: the model evaluation (with/without the MCP server); results will appear here.

## Example

```text
$ check_snippet
import cupy as cp
import cupyx.scipy.sparse.linalg as spla
from cupy.linalg import lu_factor
import nvmath
x, info = spla.cg(A, b, tol=1e-8)
w = spla.eigsh(A, 3, 'SA')
e = nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS_GELU

FAIL: 4 problem(s) found
- line 3 [error: unknown-symbol] `cupy.linalg.lu_factor` does not exist (no `lu_factor` in module `cupy.linalg`). Did you mean: `cupyx.scipy.linalg.lu_factor`?
- line 5 [error: unknown-keyword] `cupyx.scipy.sparse.linalg.cg` has no parameter `tol`. Signature: `cg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)`. Did you mean: `rtol`, `atol`?
- line 6 [error: too-many-positional] `cupyx.scipy.sparse.linalg.eigsh` takes at most 2 positional argument(s) but 3 were given. Signature: `eigsh(a, k=6, *, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True)`.
- line 7 [error: unknown-attribute] `nvmath.linalg.advanced.MatmulEpilog` has no attribute `RELU_BIAS_GELU`. Did you mean: `nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS`, `nvmath.linalg.advanced.MatmulEpilog.GELU_BIAS`, `nvmath.linalg.advanced.MatmulEpilog.RELU_AUX_BIAS`?
Checked 6 API reference(s): ...
```

A curated card, as an agent sees it (abridged):

```text
### `cupyx.scipy.sparse.linalg.cg`
cg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
cupy 14.2.0 · function · backend: cuSPARSE (SpMV), cuBLAS · SciPy equivalent: scipy.sparse.linalg.cg

Pitfalls
- The tolerance keyword is `rtol` (keyword-only, matching SciPy >= 1.12). The old `tol=` raises `TypeError`.
- Returns `(x, info)`. `info == 0` means converged and `info > 0` is the iteration count reached without converging. Always check `info`.
...
```

## Use it with an agent

```bash
# Claude Code
claude mcp add mathlib -- uvx --from git+https://github.com/jarski/mathlib-mcp mathlib-mcp
```

Any MCP client works the same way: the server speaks stdio and has no runtime dependencies besides `mcp`. Semantic search is optional: `pip install mathlib-mcp[embeddings]` fuses a small embedding model into the BM25 ranking.

## How it works

```
library sources ──► static extractor ──► symbols.json  (existence + signatures, for check_snippet)
 (sdist / wheel,     ast + .pyi + .pyx     cards.json    (API cards)
  never imported)         │                llms.txt / llms-full.txt
                          ▼
              curation/cards/*.toml  (backend, pitfalls, examples; build fails if
                                      a curated API disappears from the source)
                          │
                          ▼
                 MCP server ◄──── coding agent
```

* **Static extraction.** CuPy and nvmath-python cannot be imported without CUDA, so the extractor reads their source with `ast`. It follows re-exports, `__all__` star imports, aliases (`MatmulEpilog = cublasLt.Epilogue`), lazy module `__getattr__`, `.pyi` stubs, Cython `.pyx` files, docstring templates (`{a}` placeholders filled by nvmath's `docstring_decorator`) and `functools.partial` wrappers. The output is reproducible from pinned versions: `scripts/fetch_sources.py`, then `mathlib-mcp-build`.
* **Curation is checked.** Every pitfall was checked against the 14.2.0 / 1.0.0 sources (for example, `cupy.linalg.solve` rejects a stacked `(B, M)` right-hand side, and cuBLASLt's bias epilog is per *row*). Behavioural claims are also executable [facts](bench/facts.py) that run on a GPU. A card's example is labelled "verified on <GPU, CUDA, versions>" only after it passes on a GPU in its current form.
* **Honest uncertainty.** Namespaces the index cannot list completely (Cython modules, excluded packages) give *unverified* notes instead of false errors. A test makes sure no curated example is ever flagged.

## Benchmark

| Family | Tasks | Examples of traps |
|---|---|---|
| cuBLAS / cuBLASLt | 6 | in-place `gemm(out=, beta=)`; per-row bias epilog; 24 GB broadcasting |
| cuFFT | 6 | `irfft` length for odd sizes; `nvmath.fft.fft` rejects real input and transforms all axes; nvmath inverse FFTs are unnormalized |
| cuSOLVER | 8 | batched `solve` needs `(B, M, 1)`; silent NaN unless `cupyx.errstate(linalg='raise')`; `svd` full matrices on a 60000×50 input |
| cuSPARSE | 4 | `cg(rtol=)` not `tol=`; `eigsh` has no `'SM'` and keyword-only `which` |
| cuDSS | 1 | column-major right-hand sides |
| cuTENSOR | 1 | `nvmath.tensor` contraction API |

Each task has a prompt, a NumPy/SciPy reference, a correctness check and a canonical GPU solution. Candidates run in a separate subprocess and are classified as `pass`, `wrong_answer`, `exception`, `not_on_gpu`, `expected_error_not_raised`, `timeout` or `crash`. Static scoring adds the API-error count from `check_snippet` and checks for the required API.

### GPU validation

Before any model is scored, everything is checked on a real GPU: the canonical solutions must pass, the naive trap solutions must fail, every curated example must run, and every behavioural pitfall must hold. Latest run ([report](bench/results/gpu_validation.json)):

| Tesla T4 (sm_75, 15 GB) · CUDA 12.9 · CuPy 14.2.0 · nvmath-python 1.0.0 | Result |
|---|---|
| Canonical task solutions pass | 26 / 26 |
| Naive trap solutions fail as designed | 17 / 17 |
| Curated card examples pass | 38 / 38 |
| Pitfall facts hold | 23 / 23 |

The traps fail for the intended reasons. Examples: `OutOfMemoryError: ... 28,800,000,000 bytes` (svd with full matrices), `ValueError: The M dimension of the bias vector (48) must match the M dimension of A` (per-row bias epilog), and `TypeError: ... must be a matrix or vector with col-major layout` (cuDSS). The first run also found something the docstrings do not say: **nvmath-python's inverse FFTs are unnormalized** (`ifft(fft(x)) == n * x`, unlike NumPy and CuPy). That finding is now a checked fact, a pitfall on four cards, and a benchmark task.

```bash
uv run python bench/kaggle/make_kernel.py validate --user <kaggle-user>   # self-contained script
uvx --from kaggle kaggle kernels push -p bench/kaggle/build/validate      # runs on a Kaggle T4
uvx --from kaggle kaggle kernels output <kaggle-user>/mathlib-mcp-gpu-validation -p out
uv run python -m bench.summarize_validation out/gpu_validation.json
uv run python -m bench.record_verification out/gpu_validation.json && uv run mathlib-mcp-build
```

## Development

```bash
uv sync
uv run python scripts/fetch_sources.py   # pinned CuPy / nvmath-python sources -> .sources/
uv run mathlib-mcp-build                 # regenerate src/mathlib_mcp/data/
uv run pytest                            # includes "packaged data is up to date"
uv run ruff check .
```

MIT licensed.
