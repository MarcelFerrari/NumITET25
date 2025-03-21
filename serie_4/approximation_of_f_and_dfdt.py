import numpy as np
from collections.abc import Callable, Awaitable

def evaliptrig(y: np.ndarray, N: int) -> np.ndarray:
    """
    Evaluate a trigonometric interpolation polynomial on some data

    Args:
        y (np.ndarray): input data, len must be even
        N (int): number of evaluation points of the interpolated polynomial

    Returns:
        np.ndarray: evaluated values of the interpolation
    """
    n = len(y)
    if n % 2 != 0:
        raise ValueError("y must be of even length")
    
    max_freq = n // 2  # Nyquist frequency (since n is even)
    
    # Compute Fourier coefficients (normalized)
    # c = 1/n F_N * y
    c =  1/n * np.fft.fft(y)  

    # Prepare zero-padded frequency array
    # a = [c0, c1, ..., c N/2-1, 0, 0, 0, 0, 0, c-N/2 ... c-1]
    a = np.zeros(N, dtype=complex)
    a[:max_freq] = c[:max_freq]
    a[N - max_freq :] = c[max_freq:]
    
    # Perform inverse FFT on the zero-padded spectrum
    # ifft(y) = 1/N F_N^H y
    # F_N^H y = N * ifft(y)
    v = np.fft.ifft(a) * N  # Scale output to compensate for padding

    return v
    

def evaliDtrig(y: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the trigonometric interpolation derivative of y.

    Args:
        y (np.ndarray): input function values (even length)
        N (int): number of evaluation points

    Returns:
        np.ndarray: interpolated derivative
    """
    n = len(y)

    if n % 2 != 0:
        raise ValueError("y must be of even length")

    max_freq = n // 2  # Nyquist frequency (since n is even)

    # Compute Fourier coefficients (normalized)
    c = np.fft.fft(y) / n

    # Multiply by 2π i k
    # [0, 1] -> d = 1
    # k/N for j = -N//2 ... N//2 - 1 but flipped order!
    freqs = np.fft.fftfreq(n, d=1) * n
    #equivalently
    #freqs = np.array(list(range(n//2)) + list(range(-n//2, 0)))
    
    # Compute derivative in Fourier space
    c = (2 * np.pi * 1j * freqs) * c  

    # Prepare zero-padded frequency array
    a = np.zeros(N, dtype=complex)
    a[:max_freq] = c[:max_freq]
    a[N - max_freq :] = c[max_freq:]

    # Perform inverse FFT on the zero-padded spectrum
    v = np.fft.ifft(a) * N  # Scale output to compensate for padding

    return v

    
def compute_convergence(
    a: float,
    b: float,
    fn: Callable[[np.array], np.array],
    dfn: Callable[[np.array], np.array] | None,
    n_eval: int,
    n_convergence: int,
) -> np.array:
    """
    Computes the convergence of a trigonometric interpolation polynomial to the true value of the function or the derivate

    Args:
        a (float): left interval limit
        b (float): right interval limit
        fn (Callable[[np.array], np.array]): function of which to compute the derivate
        dfn (Callable[[np.array], np.array]) | None: exact derivate. If None computes the convergence to the function, else to this function
        n_eval (int): number of evaluations points, must be >= 2**n_convergence
        n_convergence (int): convergence will be evaluate at 2**np.arange(1,n_convergence+1) approx points

    Returns:
        np.array: l2 norm of the error of the evaluation at np.linspace(a, b, n_eval, endpoint=False)
    """
    if(n_eval < 2**n_convergence):
      raise ValueError("Eval must be large enough")

    # calculate exact solution for the finest grid
    xx = np.linspace(a, b, n_eval, endpoint=False)
    if dfn is None:
        fv = fn(xx)
    else:
        fv = dfn(xx)

    # iterate over grid densities:
    # calculate exact solutions on less dense grids,
    # interpolate to finest grid and find l2 norm of the errors
    ns = 2 ** np.arange(1, n_convergence + 1)
    norms = np.zeros(n_convergence)
    for i, n in enumerate(ns):
        t = np.linspace(a, b, n, endpoint=False)
        y = fn(t)
        if dfn is None:
            v = np.real(evaliptrig(y, n_eval))
        else:
            v = np.real(evaliDtrig(y, n_eval))
        norms[i] = np.linalg.norm(v - fv) / np.sqrt(n_eval)
    return norms



