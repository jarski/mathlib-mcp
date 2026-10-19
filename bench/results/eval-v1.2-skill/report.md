# mathlib-mcp evaluation

Run `eval-v1.2-skill`: 26 tasks × 3 samples per model and condition; code generated at commit `e09ba32`, executed on Tesla T4 (CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0). Static checks (API errors, required API) use `check_snippet` at commit `e09ba32`.

Conditions: **none** = no help; **docs** = llms-full.txt in the system prompt; **mcp** = mathlib-mcp tools; **skill** = the Agent Skill: SKILL.md in the system prompt, its files and checker as tools.

## Solved rate

Passes all GPU checks and uses the API the task asks for. 95% bootstrap CI over tasks.

| Model | none | docs | mcp | skill |
|---|---|---|---|---|
| claude-haiku-4-5 | – | – | – | 100% [100%, 100%] |
| claude-sonnet-5-5 | – | – | – | 99% [96%, 100%] |

Paired differences (same tasks):


## API errors in the final code

Share of attempts whose final code has at least one `check_snippet` error (hallucinated function/module/member, wrong or missing argument).

| Model | none | docs | mcp | skill |
|---|---|---|---|---|
| claude-haiku-4-5 | – | – | – | 0% [0%, 0%] |
| claude-sonnet-5-5 | – | – | – | 1% [0%, 4%] |

## Breakdown

| Model / condition | trap tasks | control tasks | nvmath tasks | CuPy tasks | functional pass |
|---|---|---|---|---|---|
| haiku / skill | 100% | 100% | 100% | 100% | 100% |
| sonnet / skill | 98% | 100% | 100% | 98% | 99% |

Per library family:

| Model / condition | cuBLAS | cuDSS | cuFFT | cuSOLVER | cuSPARSE | cuTENSOR |
|---|---|---|---|---|---|---|
| haiku / skill | 100% | 100% | 100% | 100% | 100% | 100% |
| sonnet / skill | 100% | 100% | 100% | 100% | 92% | 100% |

## Cost and effort per attempt

| Model / condition | cost | input tokens (incl. cached) | output tokens | model calls | tool calls | latency |
|---|---|---|---|---|---|---|
| haiku / skill | $0.0425 | 78,611 | 1,469 | 5.4 | 6.0 | 15.5 s |
| sonnet / skill | $0.0137 | 12,691 | 477 | 2.1 | 1.1 | 5.5 s |

Total generation cost: $4.38.

## Outcomes

- **haiku / skill**: pass 78
- **sonnet / skill**: pass 77, exception:TypeError 1

## Tool use (skill condition)

Share of attempts that called each tool at least once.

| Model | grep | read_file | check_gpu_code | hit turn limit |
|---|---|---|---|---|
| haiku | 100% | 92% | 100% | 0% |
| sonnet | 59% | 0% | 38% | 0% |

## Per task (solved / attempts)

| Task | haiku/skill | sonnet/skill |
|---|---|---|
| `batched_matmul` (cuBLAS) | 3/3 | 3/3 |
| `gemm_update_inplace` (cuBLAS) | 3/3 | 3/3 |
| `batched_quadratic_forms` (cuBLAS) | 3/3 | 3/3 |
| `pairwise_sq_dists` (cuBLAS) | 3/3 | 3/3 |
| `fused_linear_relu` (cuBLAS) | 3/3 | 3/3 |
| `matmul_plan_reuse` (cuBLAS) | 3/3 | 3/3 |
| `fft_convolve` (cuFFT) | 3/3 | 3/3 |
| `power_spectrum` (cuFFT) | 3/3 | 3/3 |
| `lowpass_images` (cuFFT) | 3/3 | 3/3 |
| `irfft_reconstruct` (cuFFT) | 3/3 | 3/3 |
| `nvmath_full_spectrum` (cuFFT) | 3/3 | 3/3 |
| `nvmath_spectral_gain` (cuFFT) | 3/3 | 3/3 |
| `batched_solve_vectors` (cuSOLVER) | 3/3 | 3/3 |
| `cholesky_solve_logdet` (cuSOLVER) | 3/3 | 3/3 |
| `lstsq_with_rank` (cuSOLVER) | 3/3 | 3/3 |
| `general_eigenvalues` (cuSOLVER) | 3/3 | 3/3 |
| `symmetric_top_k` (cuSOLVER) | 3/3 | 3/3 |
| `low_rank_approx` (cuSOLVER) | 3/3 | 3/3 |
| `solve_or_raise` (cuSOLVER) | 3/3 | 3/3 |
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
| haiku: skill (this run) - docs (baseline) | 87% [76%, 96%] | +13 points [+4, +24] |
| haiku: skill (this run) - mcp (baseline) | 92% [81%, 100%] | +8 points [+0, +19] |
| haiku: skill (this run) - none (baseline) | 51% [35%, 68%] | +49 points [+32, +65] |
| sonnet: skill (this run) - docs (baseline) | 99% [96%, 100%] | +0 points [+0, +0] |
| sonnet: skill (this run) - mcp (baseline) | 99% [96%, 100%] | +0 points [+0, +0] |
| sonnet: skill (this run) - none (baseline) | 94% [87%, 99%] | +5 points [+0, +12] |

## Pipeline controls

- oracle solutions (pass expected): 26/26 as expected
- naive solutions (fail expected): 17/17 as expected
- null solutions (fail expected): 26/26 as expected
- infrastructure errors (excluded from scoring): 0
