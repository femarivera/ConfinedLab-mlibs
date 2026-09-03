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
import pandas as pd
from scipy.stats import norm
from scipy.fft import fftn, ifftn, fftfreq
import os
import pyemu
from mlibs import modgeom6 # type: ignore
from pyemu.pst.pst_utils import SFMT, FFMT
import re
from io import StringIO


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

def read_parfile_safe(parfile):
    """
    Drop-in replacement for pyemu.pst_utils.read_parfile. PEST++ sometimes writes
    parval1 with a very long decimal expansion that runs directly into the
    following scale field with no separating whitespace. Since scale/offset are
    always the fixed literal "1.0000000000E+00"/"0.0000000000E+00" we write
    ourselves, insert the missing space right before that literal wherever it's
    glued to a preceding digit, then parse normally.
    """
    with open(parfile) as f:
        header = f.readline()
        body = f.read()
    body = re.sub(r'(\d)(1\.0000000000E\+00)', r'\1 \2', body)
    par_df = pd.read_csv(StringIO(body), header=None,
                          names=["parnme", "parval1", "scale", "offset"], sep=r"\s+")
    par_df.index = par_df.parnme
    return par_df


def parameterize_V1(par_df, name, nglay=None, nrow=None, ncol=None,
                  irch=None, pest=False, pest_dir='pest', expand=True):
    """
    Resolve parameter `name` purely from its par_df['type'] entry:
      'single'   -> one bare scalar value, no glay indexing
      '2darray'  -> one scalar per geological layer; expanded via `irch` into
                    (nrow, ncol) when expand=True, or returned as the raw
                    1D (nglay,) array when expand=False (so it can be subdivided
                    and re-expanded later with a different irch)
      '3darray'  -> one field per geological layer -> (nglay, nrow, ncol).
                    When pest=True, each glay row's own par_df['pp'] flag decides
                    how that layer is resolved: True -> pilot-point kriged via
                    fac2real, False -> a single calibrated scalar from par.dat,
                    broadcast uniformly. When pest=False, always a plain per-layer
                    scalar from par_df['value'] (par_df_to_1Darray/compute_3Darray),
                    regardless of 'pp'.
    Source is the setup file (par_df) when pest=False, PEST's current
    par.dat / pilot-point files when pest=True.
    """
    ptype = par_df.loc[name, 'type'] if name in par_df.index else par_df.loc[f'{name}_01', 'type']

    if ptype == 'single':
        if not pest:
            return par_df.loc[name, 'value']
        pest_par_df = read_parfile_safe(os.path.join(pest_dir, 'par.dat')) #pyemu.pst_utils.read_parfile gets me some bug I have not managed
        return pest_par_df.loc[name.lower(), 'parval1']

    if ptype == '2darray':
        if not pest:
            values_1d = par_df_to_1Darray(par_df, name)
        else:
            pest_par_df = read_parfile_safe(os.path.join(pest_dir, 'par.dat'))
            values_1d = np.array([
                pest_par_df.loc[f'{name}_{i+1:02d}'.lower(), 'parval1'] for i in range(nglay)
            ])
        return modgeom6.compute_recharge(irch, values_1d) if expand else values_1d

    if ptype == '3darray':
        if not pest:
            values_1d = par_df_to_1Darray(par_df, name)
            return modgeom6.compute_3Darray(values_1d, nglay, nrow, ncol)

        pest_par_df = read_parfile_safe(os.path.join(pest_dir, 'par.dat'))
        field_list = []
        for glay in range(nglay):
            parname = f'{name}_{glay+1:02d}'
            if par_df.loc[parname, 'pp']:
                pp_file  = os.path.join(pest_dir, f'{parname}_pp.dat')
                fac_file = os.path.join(pest_dir, f'{parname}.fac')
                field = pyemu.utils.fac2real(pp_file=pp_file, factors_file=fac_file, out_file=None)
                field_list.append(np.asarray(field).reshape(nrow, ncol))
            else:
                value = pest_par_df.loc[parname.lower(), 'parval1']
                field_list.append(np.full((nrow, ncol), value))
        return stack_fields_to_3D(field_list, nglay, nrow, ncol)

    raise ValueError(f"Unknown par_df type '{ptype}' for parameter '{name}'")

def _parnme_order_from_tpl(tpl_file):
    """Parameter names in file order, read straight from a .tpl file's ~...~ markers."""
    names = []
    with open(tpl_file) as f:
        next(f)  # skip "ptf ~"
        for line in f:
            m = re.search(r'~\s*(\S+)\s*~', line)
            if m:
                names.append(m.group(1))
    return names


def parameterize(par_df, name, nglay=None, nrow=None, ncol=None,
                  irch=None, pest=False, pest_dir='pest', expand=True,
                  ensemble=False, ensemble_file=None, ensemble_stat='mean', ensemble_real=None):
    """
    Resolve parameter `name` purely from its par_df['type'] entry:
      'single'   -> one bare scalar value, no glay indexing
      '2darray'  -> one scalar per geological layer; expanded via `irch` into
                    (nrow, ncol) when expand=True, or returned as the raw
                    1D (nglay,) array when expand=False (so it can be subdivided
                    and re-expanded later with a different irch)
      '3darray'  -> one field per geological layer -> (nglay, nrow, ncol).
                    When pest=True, each glay row's own par_df['pp'] flag decides
                    how that layer is resolved: True -> pilot-point kriged via
                    fac2real, False -> a single calibrated scalar from par.dat,
                    broadcast uniformly. When pest=False, always a plain per-layer
                    scalar from par_df['value'] (par_df_to_1Darray/compute_3Darray),
                    regardless of 'pp'.
    Source is the setup file (par_df) when pest=False, PEST's current
    par.dat / pilot-point files when pest=True.

    ensemble (bool): if True (requires pest=True), resolve the parameter's value(s) from a
        PEST++ IES parameter ensemble CSV (e.g. 'cal_ss.10.par.csv') instead of from
        par.dat/pilot-point files -- for running the model with a summary of, or a draw from,
        the calibrated posterior.
    ensemble_file (str): path to the ensemble CSV (pestpp-ies format: columns are parameter
        names, index is realization name).
    ensemble_stat (str): 'mean', 'median', 'p5', 'p95', or 'random'. The first four are
        computed independently per parameter/pilot point across all realizations. 'random'
        instead pulls a single realization's values, given by ensemble_real.
    ensemble_real (str): required when ensemble_stat='random' -- the realization name to use.
        Pick this ONCE per forward run (e.g. np.random.choice(ens_df.index)) and pass the SAME
        value into every parameterize() call for that run, so all parameters come from the
        same coherent posterior sample rather than independently mismatched realizations.
    """
    ptype = par_df.loc[name, 'type'] if name in par_df.index else par_df.loc[f'{name}_01', 'type']

    if ensemble:
        if not pest:
            raise ValueError("ensemble=True requires pest=True")
        ens_df = pd.read_csv(ensemble_file, index_col='real_name')

        def _stat(colnames):
            sub = ens_df[colnames]
            if ensemble_stat == 'mean':
                return sub.mean().values
            elif ensemble_stat == 'median':
                return sub.median().values
            elif ensemble_stat == 'p5':
                return sub.quantile(0.05).values
            elif ensemble_stat == 'p95':
                return sub.quantile(0.95).values
            elif ensemble_stat == 'random':
                if ensemble_real is None:
                    raise ValueError("ensemble_stat='random' requires ensemble_real")
                return sub.loc[ensemble_real].values
            else:
                raise ValueError(f"Unknown ensemble_stat '{ensemble_stat}'")

        if ptype == 'single':
            return _stat([name])[0]

        if ptype == '2darray':
            colnames = [f'{name}_{i+1:02d}' for i in range(nglay)]
            values_1d = _stat(colnames)
            return modgeom6.compute_recharge(irch, values_1d) if expand else values_1d

        if ptype == '3darray':
            field_list = []
            for glay in range(nglay):
                parname = f'{name}_{glay+1:02d}'
                if par_df.loc[parname, 'pp']:
                    pp_file  = os.path.join(pest_dir, f'{parname}_pp.dat')
                    fac_file = os.path.join(pest_dir, f'{parname}.fac')
                    tpl_file = os.path.join(pest_dir, f'{parname}_pp.dat.tpl')
                    pp_df = pyemu.pp_utils.pp_file_to_dataframe(pp_file)
                    parnmes = _parnme_order_from_tpl(tpl_file)
                    assert len(parnmes) == len(pp_df), f"{parname}: tpl/dat pilot point count mismatch"
                    pp_df['parval1'] = _stat(parnmes)
                    field = pyemu.utils.fac2real(pp_file=pp_df, factors_file=fac_file, out_file=None)
                    field_list.append(np.asarray(field).reshape(nrow, ncol))
                else:
                    field_list.append(np.full((nrow, ncol), _stat([parname])[0]))
            return stack_fields_to_3D(field_list, nglay, nrow, ncol)

        raise ValueError(f"Unknown par_df type '{ptype}' for parameter '{name}'")

    if ptype == 'single':
        if not pest:
            return par_df.loc[name, 'value']
        pest_par_df = read_parfile_safe(os.path.join(pest_dir, 'par.dat')) #pyemu.pst_utils.read_parfile gets me some bug I have not managed
        return pest_par_df.loc[name.lower(), 'parval1']

    if ptype == '2darray':
        if not pest:
            values_1d = par_df_to_1Darray(par_df, name)
        else:
            pest_par_df = read_parfile_safe(os.path.join(pest_dir, 'par.dat'))
            values_1d = np.array([
                pest_par_df.loc[f'{name}_{i+1:02d}'.lower(), 'parval1'] for i in range(nglay)
            ])
        return modgeom6.compute_recharge(irch, values_1d) if expand else values_1d

    if ptype == '3darray':
        if not pest:
            values_1d = par_df_to_1Darray(par_df, name)
            return modgeom6.compute_3Darray(values_1d, nglay, nrow, ncol)

        pest_par_df = read_parfile_safe(os.path.join(pest_dir, 'par.dat'))
        field_list = []
        for glay in range(nglay):
            parname = f'{name}_{glay+1:02d}'
            if par_df.loc[parname, 'pp']:
                pp_file  = os.path.join(pest_dir, f'{parname}_pp.dat')
                fac_file = os.path.join(pest_dir, f'{parname}.fac')
                field = pyemu.utils.fac2real(pp_file=pp_file, factors_file=fac_file, out_file=None)
                field_list.append(np.asarray(field).reshape(nrow, ncol))
            else:
                value = pest_par_df.loc[parname.lower(), 'parval1']
                field_list.append(np.full((nrow, ncol), value))
        return stack_fields_to_3D(field_list, nglay, nrow, ncol)

    raise ValueError(f"Unknown par_df type '{ptype}' for parameter '{name}'")

def write_par_tpl(par_df, tpl_file, par_file):
    """
    Write a PEST template file and matching initial par.dat for a set of scalar
    parameters, in the 4-column format expected by pyemu.pst_utils.read_parfile
    (header line + parnme/parval1/scale/offset). Only parval1 is a PEST marker;
    scale/offset are fixed text so the file PEST writes back stays parseable.

    par_df: DataFrame indexed by parameter name, with a 'value' column.
    """
    with open(tpl_file, "w") as f:
        f.write("ptf ~\n")
        f.write("single point\n")
        for name in par_df.index:
            f.write(SFMT(name) + "~{0:^30s}~".format(name) + FFMT(1.0) + FFMT(0.0) + "\n")

    with open(par_file, "w") as f:
        f.write("single point\n")
        for name, value in par_df['value'].items():
            f.write(SFMT(name) + FFMT(value) + FFMT(1.0) + FFMT(0.0) + "\n")

def setup_ins(obs_names, ins_filename):
    """
    Write a PEST instruction file assuming one observation value per line, in the
    same order as obs_names -- matches the long-format output written by the
    model_pest_*.py post-processing step (SIMULATED EQUIVALENT is the first
    whitespace-delimited token on each data line).
    """
    with open(ins_filename, "w") as f:
        f.write("pif ~\n")
        for i, name in enumerate(obs_names):
            f.write(f"{'l2' if i == 0 else 'l1'} !{name}!\n")
