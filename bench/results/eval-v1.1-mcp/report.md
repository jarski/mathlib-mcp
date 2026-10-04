# mathlib-mcp evaluation

Run `eval-v1.1-mcp`: 26 tasks × 3 samples per model and condition; code generated at commit `8eeaa65`, executed on Tesla T4 (CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0). Static checks (API errors, required API) use `check_snippet` at commit `c951554`.

Conditions: **none** = no help; **docs** = llms-full.txt in the system prompt; **mcp** = mathlib-mcp tools.

## Solved rate

Passes all GPU checks and uses the API the task asks for. 95% bootstrap CI over tasks.

| Model | none | docs | mcp |
|---|---|---|---|
| claude-haiku-4-5 | – | – | 96% [90%, 100%] |
| claude-sonnet-5-5 | – | – | 99% [96%, 100%] |

Paired differences (same tasks):


## API errors in the final code

Share of attempts whose final code has at least one `check_snippet` error (hallucinated function/module/member, wrong or missing argument).

| Model | none | docs | mcp |
|---|---|---|---|
| claude-haiku-4-5 | – | – | 0% [0%, 0%] |
| claude-sonnet-5-5 | – | – | 1% [0%, 4%] |

## Breakdown

| Model / condition | trap tasks | control tasks | nvmath tasks | CuPy tasks | functional pass |
|---|---|---|---|---|---|
| haiku / mcp | 95% | 100% | 100% | 95% | 96% |
| sonnet / mcp | 98% | 100% | 100% | 98% | 99% |

Per library family:

| Model / condition | cuBLAS | cuDSS | cuFFT | cuSOLVER | cuSPARSE | cuTENSOR |
|---|---|---|---|---|---|---|
| haiku / mcp | 100% | 100% | 94% | 92% | 100% | 100% |
| sonnet / mcp | 100% | 100% | 100% | 100% | 92% | 100% |

## Cost and effort per attempt

| Model / condition | cost | input tokens (incl. cached) | output tokens | model calls | tool calls | latency |
|---|---|---|---|---|---|---|
| haiku / mcp | $0.0186 | 16,213 | 1,209 | 4.4 | 4.8 | 11.0 s |
| sonnet / mcp | $0.0086 | 3,493 | 334 | 1.5 | 0.6 | 3.8 s |

Total generation cost: $2.12.

## Outcomes

- **haiku / mcp**: pass 75, expected_error_not_raised 2, exception:ValueError 1
- **sonnet / mcp**: pass 77, exception:TypeError 1

## Tool use (mcp condition)

| Model | search_api | get_api_card | get_example | check_snippet | hit turn limit |
|---|---|---|---|---|---|
| haiku | 97% | 100% | 23% | 100% | 0% |
| sonnet | 5% | 22% | 14% | 12% | 0% |

## Per task (solved / attempts)

| Task | haiku/mcp | sonnet/mcp |
|---|---|---|
| `batched_matmul` (cuBLAS) | 3/3 | 3/3 |
| `gemm_update_inplace` (cuBLAS) | 3/3 | 3/3 |
| `batched_quadratic_forms` (cuBLAS) | 3/3 | 3/3 |
| `pairwise_sq_dists` (cuBLAS) | 3/3 | 3/3 |
| `fused_linear_relu` (cuBLAS) | 3/3 | 3/3 |
| `matmul_plan_reuse` (cuBLAS) | 3/3 | 3/3 |
| `fft_convolve` (cuFFT) | 3/3 | 3/3 |
| `power_spectrum` (cuFFT) | 3/3 | 3/3 |
| `lowpass_images` (cuFFT) | 2/3 | 3/3 |
| `irfft_reconstruct` (cuFFT) | 3/3 | 3/3 |
| `nvmath_full_spectrum` (cuFFT) | 3/3 | 3/3 |
| `nvmath_spectral_gain` (cuFFT) | 3/3 | 3/3 |
| `batched_solve_vectors` (cuSOLVER) | 3/3 | 3/3 |
| `cholesky_solve_logdet` (cuSOLVER) | 3/3 | 3/3 |
| `lstsq_with_rank` (cuSOLVER) | 3/3 | 3/3 |
| `general_eigenvalues` (cuSOLVER) | 3/3 | 3/3 |
| `symmetric_top_k` (cuSOLVER) | 3/3 | 3/3 |
| `low_rank_approx` (cuSOLVER) | 3/3 | 3/3 |
| `solve_or_raise` (cuSOLVER) | 1/3 | 3/3 |
| `triangular_transposed_solve` (cuSOLVER) | 3/3 | 3/3 |
| `spmv_from_triplets` (cuSPARSE) | 3/3 | 3/3 |
| `cg_poisson_1d` (cuSPARSE) | 3/3 | 3/3 |
| `laplacian_2d_smallest_eigs` (cuSPARSE) | 3/3 | 3/3 |
| `sparse_row_normalize` (cuSPARSE) | 3/3 | 2/3 |
| `cudss_multiple_rhs` (cuDSS) | 3/3 | 3/3 |
| `cutensor_contraction` (cuTENSOR) | 3/3 | 3/3 |

## Compared with `eval-v1`

Paired by task (same tasks, same models).

| Comparison | baseline solved | difference |
|---|---|---|
| haiku: mcp (this run) - docs (baseline) | 87% [76%, 96%] | +9 points [+0, +21] |
| haiku: mcp (this run) - mcp (baseline) | 92% [81%, 100%] | +4 points [-6, +17] |
| haiku: mcp (this run) - none (baseline) | 51% [35%, 68%] | +45 points [+27, +63] |
| sonnet: mcp (this run) - docs (baseline) | 99% [96%, 100%] | +0 points [+0, +0] |
| sonnet: mcp (this run) - mcp (baseline) | 99% [96%, 100%] | +0 points [+0, +0] |
| sonnet: mcp (this run) - none (baseline) | 94% [87%, 99%] | +5 points [+0, +12] |

## Pipeline controls

- oracle solutions (pass expected): 26/26 as expected
- naive solutions (fail expected): 17/17 as expected
- null solutions (fail expected): 26/26 as expected
- infrastructure errors (excluded from scoring): 0
