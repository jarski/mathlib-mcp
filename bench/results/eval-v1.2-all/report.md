# mathlib-mcp evaluation

Run `eval-v1.2-all`: 26 tasks × 3 samples per model and condition; code generated at commit `merged`, executed on Tesla T4 (CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0). Static checks (API errors, required API) use `check_snippet` at commit `e09ba32`.

Conditions: **none** = no help; **docs** = llms-full.txt in the system prompt; **mcp** = mathlib-mcp tools; **skill** = the Agent Skill: SKILL.md in the system prompt, its files and checker as tools.

## Solved rate

Passes all GPU checks and uses the API the task asks for. 95% bootstrap CI over tasks.

| Model | none | docs | mcp | skill |
|---|---|---|---|---|
| claude-haiku-4-5 | 51% [35%, 67%] | 87% [76%, 96%] | 96% [90%, 100%] | 100% [100%, 100%] |
| claude-sonnet-5-5 | 94% [87%, 99%] | 99% [96%, 100%] | 99% [96%, 100%] | 99% [96%, 100%] |

Paired differences (same tasks):

- haiku: mcp - none: +45 points [+27, +63]
- haiku: docs - none: +36 points [+18, +54]
- haiku: mcp - docs: +9 points [+0, +21]
- haiku: skill - none: +49 points [+32, +65]
- haiku: skill - docs: +13 points [+4, +24]
- haiku: skill - mcp: +4 points [+0, +10]
- sonnet: mcp - none: +5 points [+0, +12]
- sonnet: docs - none: +5 points [+0, +12]
- sonnet: mcp - docs: +0 points [+0, +0]
- sonnet: skill - none: +5 points [+0, +12]
- sonnet: skill - docs: +0 points [+0, +0]
- sonnet: skill - mcp: +0 points [+0, +0]

## API errors in the final code

Share of attempts whose final code has at least one `check_snippet` error (hallucinated function/module/member, wrong or missing argument).

| Model | none | docs | mcp | skill |
|---|---|---|---|---|
| claude-haiku-4-5 | 36% [22%, 51%] | 1% [0%, 4%] | 0% [0%, 0%] | 0% [0%, 0%] |
| claude-sonnet-5-5 | 3% [0%, 6%] | 1% [0%, 4%] | 1% [0%, 4%] | 1% [0%, 4%] |

## Breakdown

| Model / condition | trap tasks | control tasks | nvmath tasks | CuPy tasks | functional pass |
|---|---|---|---|---|---|
| haiku / none | 42% | 100% | 6% | 65% | 51% |
| haiku / docs | 85% | 100% | 83% | 88% | 87% |
| haiku / mcp | 95% | 100% | 100% | 95% | 96% |
| haiku / skill | 100% | 100% | 100% | 100% | 100% |
| sonnet / none | 92% | 100% | 78% | 98% | 94% |
| sonnet / docs | 98% | 100% | 100% | 98% | 99% |
| sonnet / mcp | 98% | 100% | 100% | 98% | 99% |
| sonnet / skill | 98% | 100% | 100% | 98% | 99% |

Per library family:

| Model / condition | cuBLAS | cuDSS | cuFFT | cuSOLVER | cuSPARSE | cuTENSOR |
|---|---|---|---|---|---|---|
| haiku / none | 50% | 0% | 72% | 46% | 58% | 0% |
| haiku / docs | 83% | 100% | 89% | 83% | 92% | 100% |
| haiku / mcp | 100% | 100% | 94% | 92% | 100% | 100% |
| haiku / skill | 100% | 100% | 100% | 100% | 100% | 100% |
| sonnet / none | 100% | 67% | 89% | 100% | 92% | 67% |
| sonnet / docs | 100% | 100% | 100% | 100% | 92% | 100% |
| sonnet / mcp | 100% | 100% | 100% | 100% | 92% | 100% |
| sonnet / skill | 100% | 100% | 100% | 100% | 92% | 100% |

## Cost and effort per attempt

| Model / condition | cost | input tokens (incl. cached) | output tokens | model calls | tool calls | latency |
|---|---|---|---|---|---|---|
| haiku / none | $0.0025 | 351 | 424 | 1.0 | 0.0 | 3.5 s |
| haiku / docs | $0.0142 | 101,596 | 486 | 1.0 | 0.0 | 4.5 s |
| haiku / mcp | $0.0186 | 16,213 | 1,209 | 4.4 | 4.8 | 11.0 s |
| haiku / skill | $0.0425 | 78,611 | 1,469 | 5.4 | 6.0 | 15.5 s |
| sonnet / none | $0.0062 | 477 | 526 | 1.0 | 0.0 | 4.7 s |
| sonnet / docs | $0.0348 | 134,399 | 368 | 1.0 | 0.0 | 3.7 s |
| sonnet / mcp | $0.0086 | 3,493 | 334 | 1.5 | 0.6 | 3.8 s |
| sonnet / skill | $0.0137 | 12,691 | 477 | 2.1 | 1.1 | 5.5 s |

Total generation cost: $11.00.

## Outcomes

- **haiku / none**: pass 40, exception:AttributeError 15, exception:TypeError 9, exception:ValueError 5, exception:ImportError 5, exception:ModuleNotFoundError 2, exception:RuntimeError 1, wrong_answer 1
- **haiku / docs**: pass 68, exception:ValueError 6, exception:TypeError 1, exception:NameError 1, expected_error_not_raised 1, exception:NotImplementedError 1
- **haiku / mcp**: pass 75, expected_error_not_raised 2, exception:ValueError 1
- **haiku / skill**: pass 78
- **sonnet / none**: pass 73, exception:TypeError 4, wrong_answer 1
- **sonnet / docs**: pass 77, exception:TypeError 1
- **sonnet / mcp**: pass 77, exception:TypeError 1
- **sonnet / skill**: pass 77, exception:TypeError 1

## Tool use (mcp condition)

Share of attempts that called each tool at least once.

| Model | search_api | get_api_card | get_example | check_snippet | hit turn limit |
|---|---|---|---|---|---|
| haiku | 97% | 100% | 23% | 100% | 0% |
| sonnet | 5% | 22% | 14% | 12% | 0% |

## Tool use (skill condition)

Share of attempts that called each tool at least once.

| Model | grep | read_file | check_gpu_code | hit turn limit |
|---|---|---|---|---|
| haiku | 100% | 92% | 100% | 0% |
| sonnet | 59% | 0% | 38% | 0% |

## Per task (solved / attempts)

| Task | haiku/none | haiku/docs | haiku/mcp | haiku/skill | sonnet/none | sonnet/docs | sonnet/mcp | sonnet/skill |
|---|---|---|---|---|---|---|---|---|
| `batched_matmul` (cuBLAS) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `gemm_update_inplace` (cuBLAS) | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `batched_quadratic_forms` (cuBLAS) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `pairwise_sq_dists` (cuBLAS) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `fused_linear_relu` (cuBLAS) | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `matmul_plan_reuse` (cuBLAS) | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `fft_convolve` (cuFFT) | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `power_spectrum` (cuFFT) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `lowpass_images` (cuFFT) | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `irfft_reconstruct` (cuFFT) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `nvmath_full_spectrum` (cuFFT) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `nvmath_spectral_gain` (cuFFT) | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| `batched_solve_vectors` (cuSOLVER) | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `cholesky_solve_logdet` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `lstsq_with_rank` (cuSOLVER) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `general_eigenvalues` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `symmetric_top_k` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `low_rank_approx` (cuSOLVER) | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `solve_or_raise` (cuSOLVER) | 2/3 | 2/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `triangular_transposed_solve` (cuSOLVER) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `spmv_from_triplets` (cuSPARSE) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `cg_poisson_1d` (cuSPARSE) | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `laplacian_2d_smallest_eigs` (cuSPARSE) | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `sparse_row_normalize` (cuSPARSE) | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| `cudss_multiple_rhs` (cuDSS) | 0/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 |
| `cutensor_contraction` (cuTENSOR) | 0/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 |

## Pipeline controls

- oracle solutions (pass expected): 26/26 as expected
- naive solutions (fail expected): 17/17 as expected
- null solutions (fail expected): 26/26 as expected
- infrastructure errors (excluded from scoring): 0
