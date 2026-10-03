# mathlib-mcp

**Make GPU math libraries LLM-friendly, and measure whether it helps.**

LLMs write plausible but wrong code for GPU math libraries. They call functions that do not exist, use keyword arguments from an older SciPy, and miss behaviours that differ from NumPy, such as CuPy returning NaN for a singular matrix instead of raising. This project turns the APIs of **CuPy 14.2** and **nvmath-python 1.0** into an LLM-first representation, serves it to coding agents over **MCP**, and includes a **GPU benchmark** to measure the effect on generated code. Those two libraries are the Python front-ends to cuBLAS, cuFFT, cuSOLVER, cuSPARSE, cuDSS and cuTENSOR.

| Part | What it is |
|---|---|
| **API cards** | 2,741 public symbols and 275 API cards extracted from the library *sources* (no GPU or import needed). 42 cards are curated with the CUDA-X backend, pitfalls checked against the source and on a GPU, and an example verified on a GPU. |
| **MCP server** | `search_api`, `get_api_card`, `get_example`, `check_snippet`, plus an `llms.txt` resource. |
| **`check_snippet`** | Static checker that finds hallucinated APIs, wrong or positional-only keywords, too many positional arguments and missing arguments, and suggests the real name. |
| **Benchmark** | 26 GPU tasks across six CUDA-X libraries with NumPy/SciPy references; 17 have a "naive" solution that falls into a documented trap. |

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results-dark.svg">
  <img alt="Solved rate on 26 GPU tasks. Claude Haiku 4.5: 51% without help, 87% with docs in the prompt, 92% with the MCP tools. Claude Sonnet 5.5: 94%, 99%, 99%." src="docs/results-light.svg" width="720">
</picture>

Two Claude models wrote solutions to the 26 benchmark tasks, 3 samples per task, under three conditions: no help; the full API reference (`llms-full.txt`) in the system prompt; or the mathlib-mcp tools. Every solution was run on a Tesla T4 and checked against NumPy/SciPy references ([full report](bench/results/eval-v1/report.md), [all attempts and transcripts](bench/results/eval-v1/)).

| | No help | Docs in prompt | MCP tools |
|---|---|---|---|
| **Claude Haiku 4.5**: tasks solved | 51% | 87% | **92%** |
| final code with a hallucinated API or wrong arguments | 36% | 1% | **0%** |
| input tokens per task | 0.4k | 101.6k | 15.3k |
| cost per task | $0.003 | $0.014 | $0.018 |
| **Claude Sonnet 5.5**: tasks solved | 94% | 99% | 99% |
| final code with a hallucinated API or wrong arguments | 4% | 1% | 1% |
| input tokens per task | 0.5k | 134.4k | **3.5k** |
| cost per task | $0.006 | $0.035 | **$0.008** |

What the numbers say:

- **Haiku 4.5 goes from coin-flip to reliable.** Tasks solved rise by **+41 points** (95% CI +26 to +58), and hallucinated APIs fall from 36% of attempts to 0%. The biggest gain is on nvmath-python (6% → 83% solved), a library newer than the model.
- **On-demand knowledge beats putting it all in the prompt.** Both conditions contain the same curated knowledge. MCP solved slightly more for Haiku (+5 points, CI +1 to +10). Sonnet matched the docs prompt with **38× fewer input tokens, at a quarter of the cost**.
- **Sonnet 5.5 already knows most of these APIs** (94% without help), so it calls the tools only when unsure (0.6 calls per task). Its one remaining MCP failure is a JAX keyword (`cupy.repeat(..., total_repeat_length=)`) in code it did not run through `check_snippet`.
- **The evaluation found a bug in the tool itself.** In all three failed Haiku + MCP attempts at `batched_solve_vectors`, Haiku called `cupy.cublas` after only `import cupy`. That submodule is not loaded by `import cupy`, but `check_snippet` accepted the code. This is a known open bug: the checker treats every submodule as an attribute of its package. The planned fix models Python's import closure.
- **Knowledge alone is not enough when the semantics are counter-intuitive.** Haiku read the pitfall that cuBLASLt's bias epilog is per *row*, and still added the bias per column (0/3 in every condition). Sonnet got it right every time.

Caveats:

- **Same author.** The tasks and the curated pitfalls were written by the same person. The docs and MCP conditions therefore measure how well *known* pitfalls are delivered, not whether unknown ones are discovered. An independently written held-out task set is the next step.
- **Wide intervals.** With 26 tasks × 3 samples, the 95% intervals are ±10–15 points. Sonnet is near the ceiling, so its differences are within noise.
- **Pipeline controls.** Every reference solution passed (26/26), every naive trap solution failed (17/17), and every null solution failed (26/26). There were 0 infrastructure errors. Generation cost $6.58 in total.

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

### Running the evaluation

```bash
uv run python -m bench.harness.generate --out bench/results/<run> --reps 3   # Claude API
uv run python bench/kaggle/make_kernel.py eval --user <kaggle-user> \
    --candidates bench/results/<run>/generations.jsonl
uvx --from kaggle kaggle kernels push -p bench/kaggle/build/eval             # Kaggle T4
uvx --from kaggle kaggle kernels output <kaggle-user>/mathlib-mcp-eval -p out
cp out/execution*.json* bench/results/<run>/
uv run python -m bench.harness.report bench/results/<run>
uv run python -m bench.harness.chart bench/results/<run> --out docs
```

The harness gets the tool definitions and handlers from the real MCP server, calls them in-process, and checks on every call that the responding model is the one requested. It keeps infrastructure errors out of the scores, and stores the full transcript of every attempt.

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
