# mathlib-mcp evaluation

Run `eval-v1`: 26 tasks × 3 samples per model and condition; code generated at commit `3184562`, executed on Tesla T4 (CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0).

Conditions: **none** = no help; **docs** = llms-full.txt in the system prompt; **mcp** = mathlib-mcp tools.

## Solved rate

Passes all GPU checks and uses the API the task asks for. 95% bootstrap CI over tasks.

| Model | none | docs | mcp |
|---|---|---|---|
| claude-haiku-4-5 | 51% [35%, 67%] | 87% [76%, 96%] | 92% [81%, 100%] |
| claude-sonnet-5-5 | 94% [87%, 99%] | 99% [96%, 100%] | 99% [96%, 100%] |

Paired differences (same tasks):

- haiku: mcp - none: +41 points [+26, +58]
- haiku: docs - none: +36 points [+18, +54]
- haiku: mcp - docs: +5 points [+1, +10]
- sonnet: mcp - none: +5 points [+0, +12]
- sonnet: docs - none: +5 points [+0, +12]
- sonnet: mcp - docs: +0 points [+0, +0]

## API errors in the final code

Share of attempts whose final code has at least one `check_snippet` error (hallucinated function/module/member, wrong or missing argument).

| Model | none | docs | mcp |
|---|---|---|---|
| claude-haiku-4-5 | 36% [22%, 51%] | 1% [0%, 4%] | 0% [0%, 0%] |
| claude-sonnet-5-5 | 4% [0%, 10%] | 1% [0%, 4%] | 1% [0%, 4%] |

## Breakdown

| Model / condition | trap tasks | control tasks | nvmath tasks | CuPy tasks | functional pass |
|---|---|---|---|---|---|
| haiku / none | 42% | 100% | 6% | 65% | 51% |
| haiku / docs | 85% | 100% | 83% | 88% | 87% |
| haiku / mcp | 91% | 100% | 83% | 95% | 92% |
| sonnet / none | 92% | 100% | 78% | 98% | 94% |
| sonnet / docs | 98% | 100% | 100% | 98% | 99% |
| sonnet / mcp | 98% | 100% | 100% | 98% | 99% |

Per library family:

| Model / condition | cuBLAS | cuDSS | cuFFT | cuSOLVER | cuSPARSE | cuTENSOR |
|---|---|---|---|---|---|---|
| haiku / none | 50% | 0% | 72% | 46% | 58% | 0% |
| haiku / docs | 83% | 100% | 89% | 83% | 92% | 100% |
| haiku / mcp | 83% | 100% | 100% | 88% | 100% | 100% |
| sonnet / none | 100% | 67% | 89% | 100% | 92% | 67% |
| sonnet / docs | 100% | 100% | 100% | 100% | 92% | 100% |
| sonnet / mcp | 100% | 100% | 100% | 100% | 92% | 100% |

## Cost and effort per attempt

| Model / condition | cost | input tokens (incl. cached) | output tokens | model calls | tool calls | latency |
|---|---|---|---|---|---|---|
| haiku / none | $0.0025 | 351 | 424 | 1.0 | 0.0 | 3.5 s |
| haiku / docs | $0.0142 | 101,596 | 486 | 1.0 | 0.0 | 4.5 s |
| haiku / mcp | $0.0183 | 15,346 | 1,193 | 4.4 | 4.6 | 11.5 s |
| sonnet / none | $0.0062 | 477 | 526 | 1.0 | 0.0 | 4.7 s |
| sonnet / docs | $0.0348 | 134,399 | 368 | 1.0 | 0.0 | 3.7 s |
| sonnet / mcp | $0.0084 | 3,525 | 344 | 1.5 | 0.6 | 4.0 s |

Total generation cost: $6.58.

## Outcomes

- **haiku / none**: pass 40, exception:AttributeError 15, exception:TypeError 9, exception:ValueError 5, exception:ImportError 5, exception:ModuleNotFoundError 2, exception:RuntimeError 1, wrong_answer 1
- **haiku / docs**: pass 68, exception:ValueError 6, exception:TypeError 1, exception:NameError 1, expected_error_not_raised 1, exception:NotImplementedError 1
- **haiku / mcp**: pass 72, exception:ValueError 3, exception:AttributeError 3
- **sonnet / none**: pass 73, exception:TypeError 4, wrong_answer 1
- **sonnet / docs**: pass 77, exception:TypeError 1
- **sonnet / mcp**: pass 77, exception:TypeError 1

## Tool use (mcp condition)

| Model | search_api | get_api_card | get_example | check_snippet | hit turn limit |
|---|---|---|---|---|---|
| haiku | 95% | 100% | 27% | 99% | 0% |
| sonnet | 6% | 23% | 15% | 12% | 0% |

## Per task (solved / attempts)

| Task | haiku/none | haiku/docs | haiku/mcp | sonnet/none | sonnet/docs | sonnet/mcp |
|---|---|---|---|---|---|---|
| `batched_matmul` (cuBLAS) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `gemm_update_inplace` (cuBLAS) | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `batched_quadratic_forms` (cuBLAS) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `pairwise_sq_dists` (cuBLAS) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `fused_linear_relu` (cuBLAS) | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| `matmul_plan_reuse` (cuBLAS) | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `fft_convolve` (cuFFT) | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `power_spectrum` (cuFFT) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `lowpass_images` (cuFFT) | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `irfft_reconstruct` (cuFFT) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `nvmath_full_spectrum` (cuFFT) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `nvmath_spectral_gain` (cuFFT) | 0/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 |
| `batched_solve_vectors` (cuSOLVER) | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| `cholesky_solve_logdet` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `lstsq_with_rank` (cuSOLVER) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `general_eigenvalues` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `symmetric_top_k` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `low_rank_approx` (cuSOLVER) | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `solve_or_raise` (cuSOLVER) | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `triangular_transposed_solve` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `spmv_from_triplets` (cuSPARSE) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `cg_poisson_1d` (cuSPARSE) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `laplacian_2d_smallest_eigs` (cuSPARSE) | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `sparse_row_normalize` (cuSPARSE) | 3/3 | 2/3 | 3/3 | 2/3 | 2/3 | 2/3 |
| `cudss_multiple_rhs` (cuDSS) | 0/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| `cutensor_contraction` (cuTENSOR) | 0/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |

## Pipeline controls

- oracle solutions (pass expected): 26/26 as expected
- naive solutions (fail expected): 17/17 as expected
- null solutions (fail expected): 26/26 as expected
- infrastructure errors (excluded from scoring): 0
