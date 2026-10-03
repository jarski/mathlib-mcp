"""FFTs: cuFFT through CuPy and through nvmath-python."""

import numpy as np

from bench.task import Task, assert_close


def _dtype_close(dtype, rtol, atol):
    def check(actual, expected, inputs):
        if np.asarray(actual).dtype != dtype:
            raise AssertionError(f"expected {np.dtype(dtype)} output, "
                                 f"got {np.asarray(actual).dtype}")
        assert_close(actual, expected, rtol, atol)
    return check


def _lowpass_reference(images, radius):
    _, H, W = images.shape
    F = np.fft.fft2(images.astype(np.float64), axes=(-2, -1))
    ky = np.abs(np.fft.fftfreq(H) * H)
    kx = np.abs(np.fft.fftfreq(W) * W)
    mask = (ky[:, None] <= radius) & (kx[None, :] <= radius)
    return np.fft.ifft2(F * mask, axes=(-2, -1)).real.astype(np.float32)


def _rfft_inputs(rng, case):
    n = 1001 if case == 0 else 512
    signals = rng.standard_normal((6, n))
    return np.fft.rfft(signals, axis=-1), n


TASKS = [
    Task(
        id="fft_convolve",
        family="cuFFT",
        difficulty="medium",
        prompt="""\
Implement `fft_convolve(x, h)`.

`x` and `h` are real 1-D float64 CuPy arrays of lengths n and m. Return the full
linear convolution (length n + m - 1, float64), i.e. the same result as
`numpy.convolve(x, h, mode='full')`, computed with FFTs on the GPU.""",
        entry_point="fft_convolve",
        solution="""\
import cupy as cp


def fft_convolve(x, h):
    n = x.size + h.size - 1
    X = cp.fft.rfft(x, n=n)
    H = cp.fft.rfft(h, n=n)
    return cp.fft.irfft(X * H, n=n)
""",
        make_inputs=lambda rng, case: (
            # n + m - 1 is odd in both cases, so irfft needs the explicit length
            (rng.standard_normal(1000), rng.standard_normal(38)) if case == 0
            else (rng.standard_normal(513), rng.standard_normal(129))),
        reference=lambda x, h: np.convolve(x, h, mode="full"),
        rtol=1e-6,
        atol=1e-9,
        trap="Zero-padding to n + m - 1 and passing that length to irfft (odd lengths).",
        naive="""\
import cupy as cp


def fft_convolve(x, h):
    n = x.size + h.size - 1
    return cp.fft.irfft(cp.fft.rfft(x, n=n) * cp.fft.rfft(h, n=n))
""",
    ),
    Task(
        id="power_spectrum",
        family="cuFFT",
        difficulty="easy",
        prompt="""\
Implement `power_spectrum(signals)`.

`signals` is a float32 CuPy array of shape (B, N) holding B real signals. Return
`|FFT|**2` of each row for the non-negative frequencies only, as a float32 array of
shape (B, N//2 + 1).""",
        entry_point="power_spectrum",
        solution="""\
import cupy as cp


def power_spectrum(signals):
    X = cp.fft.rfft(signals, axis=-1)
    return (X.real ** 2 + X.imag ** 2).astype(cp.float32)
""",
        make_inputs=lambda rng, case: (rng.standard_normal((16, 256)).astype(np.float32),),
        reference=lambda s: (np.abs(np.fft.rfft(s.astype(np.float64), axis=-1)) ** 2
                             ).astype(np.float32),
        check=_dtype_close(np.float32, 1e-3, 1e-2),
        trap="Control task (rfft).",
    ),
    Task(
        id="lowpass_images",
        family="cuFFT",
        difficulty="medium",
        prompt="""\
Implement `lowpass(images, radius)`.

`images` is a float32 CuPy array of shape (B, H, W) (B real images) and `radius` is
an int. For each image independently: take its 2-D FFT, keep only the frequencies
whose integer frequency indices satisfy |ky| <= radius and |kx| <= radius, set all
other coefficients to zero, and return the real part of the inverse 2-D FFT. The
integer frequency index of position p along an axis of length L is
`(numpy.fft.fftfreq(L) * L)[p]` (0, 1, 2, ..., then negative values). Return float32
(B, H, W).""",
        entry_point="lowpass",
        solution="""\
import cupy as cp


def lowpass(images, radius):
    _, H, W = images.shape
    F = cp.fft.fft2(images, axes=(-2, -1))
    ky = cp.abs(cp.fft.fftfreq(H) * H)
    kx = cp.abs(cp.fft.fftfreq(W) * W)
    mask = (ky[:, None] <= radius) & (kx[None, :] <= radius)
    return cp.fft.ifft2(F * mask, axes=(-2, -1)).real.astype(cp.float32)
""",
        make_inputs=lambda rng, case: (rng.random((10, 64, 32), dtype=np.float32),
                                       5 if case == 0 else 9),
        reference=_lowpass_reference,
        check=_dtype_close(np.float32, 1e-4, 1e-4),
        trap="Transforming only the image axes (not the batch axis) and the frequency "
             "index convention.",
    ),
    Task(
        id="irfft_reconstruct",
        family="cuFFT",
        difficulty="easy",
        prompt="""\
Implement `reconstruct(coeffs, n)`.

`coeffs` is a complex128 CuPy array of shape (B, n//2 + 1): the `rfft` coefficients
(along the last axis) of B real signals of length `n`, where `n` (a Python int) may be
odd. Return the real signals as a float64 array of shape (B, n).""",
        entry_point="reconstruct",
        solution="""\
import cupy as cp


def reconstruct(coeffs, n):
    return cp.fft.irfft(coeffs, n=n, axis=-1)
""",
        make_inputs=_rfft_inputs,
        reference=lambda coeffs, n: np.fft.irfft(coeffs, n=n, axis=-1),
        rtol=1e-10,
        atol=1e-12,
        trap="Without `n=`, irfft returns 2*(m-1) samples, which is wrong for odd n.",
        naive="""\
import cupy as cp


def reconstruct(coeffs, n):
    return cp.fft.irfft(coeffs, axis=-1)
""",
    ),
    Task(
        id="nvmath_full_spectrum",
        family="cuFFT",
        difficulty="hard",
        prompt="""\
Implement `full_spectrum(x)` with nvmath-python.

`x` is a real float32 CuPy array of shape (B, N). Return the complete complex FFT of
each row (all N frequencies, like `numpy.fft.fft(x, axis=-1)`) as a complex64 CuPy
array of shape (B, N). Use nvmath-python's FFT (`nvmath.fft`), not `cupy.fft`.""",
        entry_point="full_spectrum",
        solution="""\
import cupy as cp
import nvmath


def full_spectrum(x):
    # nvmath.fft.fft is complex-to-complex only and transforms all axes by default
    return nvmath.fft.fft(x.astype(cp.complex64), axes=[-1])
""",
        make_inputs=lambda rng, case: (rng.standard_normal((12, 512)).astype(np.float32),),
        reference=lambda x: np.fft.fft(x.astype(np.float64), axis=-1).astype(np.complex64),
        check=_dtype_close(np.complex64, 1e-3, 1e-3),
        required_api=("nvmath.fft",),
        requires=("nvmath",),
        trap="nvmath.fft.fft rejects real input, and without `axes` it does an N-D FFT.",
        naive="""\
import nvmath


def full_spectrum(x):
    return nvmath.fft.fft(x)
""",
    ),
]
