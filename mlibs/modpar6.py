# ==========================================================================================
#  modpar6.py - Modular Parameter Utilities for MODFLOW 6 Model Generation and Visualization
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project 
#
#  DESCRIPTION:
#  ------------
#  As part of the ConfinedLab project, this module provides flexible utilities 
#  for generating and visualizing spatially variable parameter fields for MODFLOW 6 groundwater models.
#  It includes functions to create 2D and 3D correlated fields of hydraulic parameters by specifying a desired 
#  spatial structure (e.g., isotropic or anisotropic) and statistical properties (e.g., mean, variance).
#  correlation structure. 
#
#  MAIN FEATURES:
#  --------------
#  - Generate 2D and 3D random fields of hydraulic parameters (K, Sy, Ss) with specified variogram models.
#  - Helper functions to get variogram parameters according to common summary statistics of prior data/knowledge

import numpy as np
from scipy.stats import norm
from scipy.fft import fftn, ifftn, fftfreq

def moments_from_arithmetic_mean_variance(arith_mean, arith_var):
    """
    Helper function to convert between log-normal distribution parameters and 
    common summary stats (or knowledge) of hydraulic properties.

    Let K be a random variable following a log normal distribution with moments mu and sigma^2.

    Given arithmetic mean (m=E[K]) and variance (v=var[K]) of of the variable K,
    return the parameters of the log-normal distribution of K (mu and sigma^2).

    The aritmetic mean generally corresponds to the expected value of K in a heterogeneous aquifer.
    The variance could be expresed as percent variation relative to the mean, for example with a 
    coefficient of variation CV = sqrt(v)/m, then v = (CV*m)^2. 
    """
    m = arith_mean
    v = arith_var
    sigma2 = np.log(1.0 + v / m**2) #Can be used as an approximation of the sill for a variogram of Z=ln(K)
    mu = np.log(m) - 0.5 * sigma2
    geom_mean = np.exp(mu)
    sigma = np.sqrt(sigma2)

    return geom_mean, mu, sigma2, sigma

def moments_from_percentiles(k1, p1, k2, p2):
    """
    Helper function to convert between log-normal distribution parameters and 
    common summary stats (or knowledge) of hydraulic properties.

    Let K be a random variable following a log normal distribution with moments mu and sigma^2.

    Given two percentiles of the lognormal variable K, k1 at p1, k2 at p2, 
    with p1 and p2 within (0,1) (e.g. 0.05 and 0.95) 
    return the parameters of the log-normal distribution of K (mu and sigma^2).
    
    Very often a given hydraulic property of an aquifer is known as a broad range.
    One could assume that this range represents the 90% confidence interval (5th and 95th percentiles)
    of a log-normal distribution, and use this function to estimate the distribution parameters.

    Raises:
        ValueError: if p1 or p2 are not strictly within (0,1), if p1 == p2,
                    or if k1/k2 are not both positive.
    """
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError(f"Percentiles must be strictly within (0,1), got p1={p1}, p2={p2}")
    if p1 == p2:
        raise ValueError("Percentiles must be distinct")
    if k1 <= 0 or k2 <= 0:
        raise ValueError(f"k1 and k2 must be positive (lognormal support), got k1={k1}, k2={k2}")

    z1 = norm.ppf(p1)
    z2 = norm.ppf(p2)

    # Guard against inconsistent ordering (e.g. p1 > p2 but k1 < k2, or vice versa),
    if (k2 - k1) * (z2 - z1) < 0:
        raise ValueError(
            "Inconsistent inputs: (k1, p1) and (k2, p2) must be ordered consistently "
            "(k2 > k1 iff p2 > p1)."
        )

    sigma = (np.log(k2) - np.log(k1)) / (z2 - z1)  # Can be used as an approximation of the sill for a variogram of Z=ln(K)
    mu = np.log(k1) - z1 * sigma
    geom_mean = np.exp(mu)
    sigma2 = sigma**2
    return geom_mean, mu, sigma2, sigma

def moments_from_log_mean_variance(log_mean, log_var, log_base=10):
    """
    Helper function to convert between log-normal distribution parameters and 
    common summary stats (or knowledge) of hydraulic properties.

    Let K be a random variable following a log normal distribution with moments mu and sigma^2.
    Let Y = log_b(K) be the logarithm in base b of K, which follows a normal distribution with mean mu_b and variance sigma^2_b.
    
    Given mu_b and sigma^2_b of Y = log_b(K), return the parameters of the log-normal distribution of K (mu and sigma^2).
    """
    mu_b = log_mean
    sigma2_b = log_var
    
    mu = mu_b * np.log(log_base)
    sigma2 = sigma2_b * ((np.log(log_base))**2) #Can be used as an approximation of the sill for a variogram of Z=ln(K)

    geom_mean = np.exp(mu)
    sigma = np.sqrt(sigma2)
    return geom_mean, mu, sigma2, sigma

def generate_random_field(shape, variogram_type="exponential",
                          geom_mean=1e-4, sill=1.0, nugget=0.0, range_param=10.0,
                          drow=1.0, dcol=1.0, param_type="K", seed=None, log_base=None):
    """
    Generate a 2D random field of a variable following a log-normal distribution with a desired spatial correlation structure
    using a spectral (FFT-based) simulation method.

    This function creates a spatially correlated random field by filtering Gaussian white noise in the frequency domain
    according to a specified variogram model spectrum (exponential, gaussian, or spherical_approx). The resulting field is then
    transformed to a log-normal distribution, commonly used for simulating heterogeneous properties such as hydraulic conductivity.

    Args:
        shape (tuple): Shape of the output field grid (nx, ny).
        variogram_type (str): Type of variogram/covariance model ("exponential", "gaussian", or "spherical_approx").
            NOTE: "spherical_approx" is NOT the true spherical variogram's spectral density (the spherical
            covariance's Fourier transform has no simple closed form in 2D). It's a smooth stand-in with a
            faster-decaying spectrum than the exponential model, giving qualitatively similar range/sill
            behavior. If you need an exact spherical variogram, simulate the spectrum numerically from the
            truncated spherical covariance (via FFT) or use turning-bands / sequential Gaussian simulation instead.
        geom_mean (float): Geometric mean of the log-normal field. Normally the most meaningful statistic for log-normal variables in hydrogeology.
        sill (float): Sill of the variogram (total variance parameter of the log-normal distribution, including nugget). Should represent total variability.
        nugget (float): Nugget effect (variance at zero distance). Represents unstructured variability. Must be <= sill.
        range_param (float): Correlation length (practical range) in model units.
        drow (float): Grid spacing in the row direction (model units).
        dcol (float): Grid spacing in the column direction (model units).
        seed (int, optional): Random seed for reproducibility.
        log_base (float, optional): Base of logarithm for the transformation of the randomly generated field. If None, sill and nugget 
                                    are interpreted as variance of the natural logarithm. If log_base is specified (e.g., 10), sill and nugget
                                    are interpreted as variance of the logarithm in that base, and converted accordingly.

    Returns:
        np.ndarray: 2D array of shape (nx, ny) representing the log-normal random field with spatial correlation.

    Raises:
        ValueError: If variogram_type is unsupported, or if nugget > sill.
    """

    if nugget > sill:
        raise ValueError(f"nugget ({nugget}) cannot exceed sill ({sill})")

    rng = np.random.default_rng(seed)
    nx, ny = shape

    # Frequency grid (in 1/meters)
    kx = fftfreq(nx, d=drow).reshape(-1, 1)
    ky = fftfreq(ny, d=dcol).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)

    # Power spectral density ~ FT of covariance
    if variogram_type == "exponential":
        spectrum = 1.0 / (1.0 + (2*np.pi*k*range_param)**2)**1.5
    elif variogram_type == "gaussian":
        spectrum = np.exp(-(np.pi*k*range_param)**2)
    elif variogram_type == "spherical_approx":
        # Approximation only: faster-decaying spectrum than the exponential model,
        # chosen for qualitatively similar range/sill behavior. This is NOT derived
        # from the true spherical covariance's Fourier transform (see docstring).
        spectrum = 1.0 / (1.0 + (2*np.pi*k*range_param)**2)**2
    else:
        raise ValueError("Unsupported variogram type")

    # Generate white noise in space, FFT, filter
    w = rng.normal(size=(nx, ny))
    W = fftn(w)

    # Inverse FFT to get correlated field, back to real space
    Z = np.real(ifftn(W * np.sqrt(spectrum)))

    # Standardize: Gaussian field, mean 0, std 1
    Z = (Z - np.mean(Z)) / np.std(Z)

    # Convert geom_mean and sill if log_base is specified
    if log_base is not None:
        logb = np.log(log_base)
        mu = np.log(geom_mean)
        sill = sill * (logb**2)
        nugget = nugget * (logb**2)
    else:
        mu = np.log(geom_mean)
        sill = sill
        nugget = nugget

    # Rescale to desired variance
    # Structured variability
    sigma = np.sqrt(sill - nugget)
    Z = (sigma * Z)
    # Unstructured variability (nugget) — use the seeded rng for reproducibility
    Z = Z + (np.sqrt(nugget) * rng.normal(size=(nx, ny)))

    # Shift to the desired mean
    Z = Z + mu  # Now Z is a gaussian field with mean mu and total variance sill
                # (structured variance sill-nugget + nugget variance nugget = sill)

    # Log-normal transformation
    field = np.exp(Z)  # Now field follows a log-normal distribution with parameters mu and sigma2=sill

    # Apply parameter-specific constraints
    if param_type.lower() == "sy":
        # Specific yield is bounded between [0, 1], usually << 1
        field = np.clip(field, 0.001, 0.5)

    return field

def stack_fields_to_3D(field_list, nlay, nrow, ncol):
    """
    Stack a list of 2D fields into a 3D array of shape (nlay, nrow, ncol).

    Args:
        field_list (list): List of 2D arrays, each of shape (nrow, ncol).
        nlay (int): Number of layers (should match len(field_list)).
        nrow (int): Number of rows in each field.
        ncol (int): Number of columns in each field.

    Returns:
        np.ndarray: 3D array of shape (nlay, nrow, ncol).

    Raises:
        ValueError: If input dimensions do not match.
    """

    if not isinstance(field_list, (list, tuple)):
        raise ValueError("field_list must be a list or tuple of 2D arrays.")
    if len(field_list) != nlay:
        raise ValueError(f"field_list must have length nlay ({nlay}).")
    for i, arr in enumerate(field_list):
        arr = np.asarray(arr)
        if arr.shape != (nrow, ncol):
            raise ValueError(f"Field at index {i} has shape {arr.shape}, expected ({nrow}, {ncol}).")
    arr3d = np.stack([np.asarray(f) for f in field_list], axis=0)
    return arr3d

def par_df_to_1Darray(df, prefix):
    # Helper function to extract parameter arrays from setup file dataframe
    
    subset = df[df.index.str.startswith(prefix)]
    subset = subset.sort_index()  # Ensure correct order
    return subset["value"].to_numpy()