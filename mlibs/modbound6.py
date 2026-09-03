# ==========================================================================================
#  modbound6.py - Modular Utilities for Boundary Condition Generation in MODFLOW 6 Models
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project 
#
#  DESCRIPTION:
#  ------------
#  As part of the ConfinedLab project, this module provides flexible utilities 
#  for generating and manipulating boundary condition stress period data (SPD) for steady state MODFLOW 6 groundwater models.
#  The approach supports the creation of river (RIV), general head boundary (GHB), and drain (DRN) package inputs,
#  as well as utilities for extracting active cell indices from model arrays.
#
#  MAIN FEATURES:
#  --------------
#  - Generate SPD for river, general head, and drain boundaries with flexible elevation and conductance options.
#  - Support for both proportion-based and absolute elevation specification for boundary stages.
#  - Input validation for clarity and reliability in all boundary condition functions.
#  - Utilities for extracting active cell indices from irch/idomain arrays, with options for subsetting and sampling.

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def create_riv_spd(
    cells,
    ztop,
    thickness_array,
    k_array,
    river_length,
    river_width=1,
    riverbed_thickness=1,
    stage_type="proportion",
    a=0.1,
    b=1.0,
    conc=None
):
    """
    Create river boundary condition parameters for multiple specified cells.

    Parameters:
        cells (list of tuples): List of (k, i, j) tuples specifying layer, row, and column indices.
        ztop (3D array): Top elevation of each cell (shape: nlay x nrow x ncol).
        thickness_array (3D array): Thickness of each cell (shape: nlay x nrow x ncol).
        k_array (float, 1D array): Hydraulic conductivity of riverbed for each layer (shape: nlay) for conductance calculation.
        river_length (float): Length of the river over the cells (horizontal discretization) for conductance calculation.
        river_width (float, optional): Width of the river (default: 1) for conductance calculation.
        riverbed_thickness (float, optional): Thickness of the riverbed (default: 1) for conductance calculation.
        stage_type (str, optional): Way off setting the river stage relative to the cell top elevation.
                                    'proportion' or 'absolute'. If 'proportion', a is fraction of thickness. If 'absolute', a is absolute offset.
        a (float, optional): Offset for the location of river stage relative to the top elevation of the cell.
                            If 'proportion', must be 0 < a < 1. If 'absolute', must be a > 0.
        b (float, optional): Desired separation between river stage and river bottom (in model units, default: 1.0).
        conc (float, 2D array, or None, optional): River concentration (if applicable). Default is None.

    Returns:
        riv_spd (dict): Stress period data for the river boundary condition.
            Format: {0: [(k, i, j, stage, cond, bottom[, conc]), ...]}
    """
    # Input checks
    if not isinstance(cells, (list, tuple)) or not all(isinstance(cell, (tuple, list)) and len(cell) == 3 for cell in cells):
        raise ValueError("cells must be a list of (k, i, j) tuples.")
    if not (hasattr(ztop, "shape") and len(ztop.shape) == 3):
        raise ValueError("ztop must be a 3D array (nlay, nrow, ncol).")
    if not (hasattr(thickness_array, "shape") and len(thickness_array.shape) == 3):
        raise ValueError("thickness_array must be a 3D array (nlay, nrow, ncol).")
    if not (isinstance(river_length, (float, int)) and river_length > 0):
        raise ValueError("river_length must be a positive number.")
    if not (isinstance(riverbed_thickness, (float, int)) and riverbed_thickness > 0):
        raise ValueError("riverbed_thickness must be a positive number.")
    if not (isinstance(river_width, (float, int)) and river_width > 0):
        raise ValueError("river_width must be a positive number.")
    if stage_type == "proportion":
        if not (isinstance(a, (float, int)) and 0 < a < 1):
            raise ValueError("For 'proportion' stage_type, 'a' must be a float strictly between 0 and 1 (0 < a < 1).")
    elif stage_type == "absolute":
        if not (isinstance(a, (float, int)) and a >= 0):
            raise ValueError("For 'absolute' stage_type, 'a' must be a positive float or int (a >= 0).")
    else:
        raise ValueError("stage_type must be either 'proportion' or 'absolute'.")
    if not (isinstance(b, (float, int)) and b >= 0):
        raise ValueError("b must be a non-negative number (desired separation in model units).")
    if conc is not None and not (isinstance(conc, (float, int)) or hasattr(conc, "shape")):
        raise ValueError("conc must be a float, int, 2D array, or None.")
    nlay, nrow, ncol = ztop.shape
    if thickness_array.shape != (nlay, nrow, ncol):
        raise ValueError("thickness_array must have the same shape as ztop.")
    if isinstance(k_array, (float, int)):
        k_array = np.full(nlay, k_array, dtype=float)
    elif hasattr(k_array, "shape") and len(k_array.shape) == 1:
        if k_array.shape[0] != nlay:
            raise ValueError(
            f"k_array must have length nlay ({nlay}), "
            f"but has length {k_array.shape[0]}.")
    else:
        raise ValueError(
        "k_array must be either a scalar or a 1D array of length nlay.")
    
    # Initialize the stress period data dictionary
    riv_spd = {}
    riv_entries = []

    for k, i, j in cells:
        cell_bottom = ztop[k, i, j] - thickness_array[k, i, j]
        if stage_type == "proportion":
            riv_stage = ztop[k, i, j] - (a * thickness_array[k, i , j])
        elif stage_type == "absolute":
            riv_stage = ztop[k, i, j] - a
            if riv_stage < cell_bottom:
                riv_stage = ztop[k, i, j] - (0.1 * thickness_array[k, i , j])

        riv_bottom = riv_stage - b
        if riv_bottom < cell_bottom:
            # If below cell bottom, set it a proportion below river stage
            riv_bottom = riv_stage - (0.1 * (riv_stage-cell_bottom))
            # # Ensure still below stage
            # riv_bottom = min(riv_bottom, riv_stage - 0.01)        

        assert ztop[k, i, j] >= riv_stage >= riv_bottom >= cell_bottom, (
            f"Inconsistent elevations for cell (k={k}, i={i}, j={j}): "
            f"ztop={ztop[k, i, j]}, riv_stage={riv_stage}, riv_bottom={riv_bottom}, cell_bottom={cell_bottom}"
        )

        riv_cond = (k_array[k] * river_length * river_width) / (riverbed_thickness)

        if conc is not None:
            cell_conc = conc[i, j] if hasattr(conc, "shape") else conc
            riv_entries.append((k, i, j, riv_stage, riv_cond, riv_bottom, cell_conc))
        else:
            riv_entries.append((k, i, j, riv_stage, riv_cond, riv_bottom))

    riv_spd[0] = riv_entries
    return riv_spd

def create_ghb_spd(
    cells,
    ztop,
    thickness_array,
    k_array,
    ghb_length,
    ghb_distance=1,
    gh_type="proportion",
    a=0.1,
    conc=None):
    """
    Create ghb boundary condition parameters for multiple specified cells.

    Parameters:
    - cells (list of tuples): A list of (k, i, j) tuples specifying the layer, row, and column indices of the cells.
    - ztop (3D array): Top elevation of each cell (shape = nlay x nrow x ncol). 
    - thickness_array (3D array): Thickness of each cell (shape = nlay x nrow x ncol).
    - k_array (float, 1D array): Hydraulic conductivity for each layer (shape = nlay).   
    - ghb_length (float): Length of the ghb for conductance calculation (for lateral flow it corresponds to the cell width
                        perpendicular to the flow direction).
    - ghb_width (float): Width of the ghb for conductance calculation (for lateral flow it corresponds to the saturated thickness
                         perpendicular to the flow direction: used built-in).
    - ghb_distance (float): Distance of the gh from the cell boundary (default 1 model units) for conductance calculation.
      For more on conductance calculations see https://www.xmswiki.com/wiki/GMS:GHB_Package and the MODFLOW 6 documentation. 
    - gh_type (str, optional): Way of setting the gh elevation relative to the cell top elevation.
                                 'proportion' or 'absolute'. If 'proportion', a is fraction of thickness. 
                                 If 'absolute', a is absolute offset.
    - a (float, optional): Offset for the location of ghb stage relative to the top elevation of the cell.
                            If 'proportion', must be 0 < a < 1. If 'absolute', must be a > 0.    
    - conc (float, 2D array, or None, optional): ghb concentration (if applicable). Default is None.

    Returns:
    - ghb_spd (dict): Stress period data for the ghb boundary condition.
    """

    # Input checks
    if not isinstance(cells, (list, tuple)) or not all(isinstance(cell, (tuple, list)) and len(cell) == 3 for cell in cells):
        raise ValueError("cells must be a list of (k, i, j) tuples.")
    if not (hasattr(ztop, "shape") and len(ztop.shape) == 3):
        raise ValueError("ztop must be a 3D array (nlay, nrow, ncol).")
    if not (hasattr(thickness_array, "shape") and len(thickness_array.shape) == 3):
        raise ValueError("thickness_array must be a 3D array (nlay, nrow, ncol).")
    if not (isinstance(ghb_length, (float, int)) and ghb_length > 0):
        raise ValueError("ghb_length must be a positive number.")
    if not (isinstance(ghb_distance, (float, int)) and ghb_distance > 0):
        raise ValueError("ghb_distance must be a positive number.")
    if gh_type == "proportion":
        if not (isinstance(a, (float, int)) and 0 < a < 1):
            raise ValueError("For 'proportion' gh_type, 'a' must be a float strictly between 0 and 1 (0 < a < 1).")
    elif gh_type == "absolute":
        if not (isinstance(a, (float, int)) and a > 0):
            raise ValueError("For 'absolute' gh_type, 'a' must be a positive float or int (a > 0).")
    else:
        raise ValueError("gh_type must be either 'proportion' or 'absolute'.")
    if conc is not None and not (isinstance(conc, (float, int)) or hasattr(conc, "shape")):
        raise ValueError("conc must be a float, int, 2D array, or None.")

    nlay, nrow, ncol = ztop.shape
    if thickness_array.shape != (nlay, nrow, ncol):
        raise ValueError("thickness_array must have the same shape as ztop.")
    if isinstance(k_array, (float, int)):
        k_array = np.full(nlay, k_array, dtype=float)
    elif hasattr(k_array, "shape") and len(k_array.shape) == 1:
        if k_array.shape[0] != nlay:
            raise ValueError(
            f"k_array must have length nlay ({nlay}), "
            f"but has length {k_array.shape[0]}.")
    else:
        raise ValueError(
        "k_array must be either a scalar or a 1D array of length nlay.")
    
    # Initialize the stress period data dictionary
    ghb_spd = {}
    ghb_entries = []

    for k, i, j in cells:
        cell_bottom = ztop[k, i, j] - thickness_array[k, i, j]
        if gh_type == "proportion":
            ghb_elev = ztop[k, i, j] - (a * thickness_array[k, i , j])
        elif gh_type == "absolute":
            ghb_elev = ztop[k, i, j] - a
            if ghb_elev < cell_bottom:
                ghb_elev = ztop[k, i, j] - (0.1 * thickness_array[k, i , j])

        assert ztop[k, i, j] >= ghb_elev >= cell_bottom, (
            f"Inconsistent elevations for cell (k={k}, i={i}, j={j}): "
            f"ztop={ztop[k, i, j]}, ghb_elev={ghb_elev}, cell_bottom={cell_bottom}"
        )

        ghb_cond = (k_array[k] * ghb_length * thickness_array[k, i, j]) / (ghb_distance)

        if conc is not None:
            cell_conc = conc[i, j] if hasattr(conc, "shape") else conc
            ghb_entries.append((k, i, j, ghb_elev, ghb_cond, cell_conc))
        else:
            ghb_entries.append((k, i, j, ghb_elev, ghb_cond))

    ghb_spd[0] = ghb_entries
    return ghb_spd

def create_drn_spd(
    cells,
    ztop,
    thickness_array,
    k_array,
    drain_length,
    drain_width=1,
    drainbed_thickness=1,
    elev_type="proportion",
    a=0.1,
    conc=None):
    """
    Create drain boundary condition parameters for multiple specified cells.

    Parameters:
    - cells (list of tuples): A list of (k, i, j) tuples specifying the layer, row, and column indices of the cells.
    - ztop (3D array): Top elevation of each cell (shape = nlay x nrow x ncol). 
    - thickness_array (3D array): Thickness of each cell (shape = nlay x nrow x ncol).
    - k_array (float, 1D array): Hydraulic conductivity for each layer (shape = nlay).   
    - drain_length (float): Length of the drain for conductance calculation.
    - drain_width (float): Width of the drain for conductance calculation.
    - drainbed_thickness (float): Thickness of the sediments at the drain for conductance calculation (default 1 model units).
    - elev_type (str, optional): Way of setting the drain elevation relative to the cell top elevation.
                                 'proportion' or 'absolute'. If 'proportion', a is fraction of thickness. 
                                 If 'absolute', a is absolute offset.
    - a (float, optional): Offset for the location of drain stage relative to the top elevation of the cell.
                            If 'proportion', must be 0 < a < 1. If 'absolute', must be a > 0.    
    - conc (float, 2D array, or None, optional): Drain concentration (if applicable). Default is None.

    Returns:
    - drn_spd (dict): Stress period data for the drain boundary condition.
    """

    # Input checks
    if not isinstance(cells, (list, tuple)) or not all(isinstance(cell, (tuple, list)) and len(cell) == 3 for cell in cells):
        raise ValueError("cells must be a list of (k, i, j) tuples.")
    if not (hasattr(ztop, "shape") and len(ztop.shape) == 3):
        raise ValueError("ztop must be a 3D array (nlay, nrow, ncol).")
    if not (hasattr(thickness_array, "shape") and len(thickness_array.shape) == 3):
        raise ValueError("thickness_array must be a 3D array (nlay, nrow, ncol).")
    if not (isinstance(drain_length, (float, int)) and drain_length > 0):
        raise ValueError("drain_length must be a positive number.")
    if not (isinstance(drainbed_thickness, (float, int)) and drainbed_thickness > 0):
        raise ValueError("drainbed_thickness must be a positive number.")
    if not (isinstance(drain_width, (float, int)) and drain_width > 0):
        raise ValueError("drain_width must be a positive number.")
    if elev_type == "proportion":
        if not (isinstance(a, (float, int)) and 0 < a < 1):
            raise ValueError("For 'proportion' elev_type, 'a' must be a float strictly between 0 and 1 (0 < a < 1).")
    elif elev_type == "absolute":
        if not (isinstance(a, (float, int)) and a >= 0):
            raise ValueError("For 'absolute' elev_type, 'a' must be a positive float or int (a >= 0).")
    else:
        raise ValueError("elev_type must be either 'proportion' or 'absolute'.")
    if conc is not None and not (isinstance(conc, (float, int)) or hasattr(conc, "shape")):
        raise ValueError("conc must be a float, int, 2D array, or None.")

    nlay, nrow, ncol = ztop.shape
    if thickness_array.shape != (nlay, nrow, ncol):
        raise ValueError("thickness_array must have the same shape as ztop.")
    if isinstance(k_array, (float, int)):
        k_array = np.full(nlay, k_array, dtype=float)
    elif hasattr(k_array, "shape") and len(k_array.shape) == 1:
        if k_array.shape[0] != nlay:
            raise ValueError(
            f"k_array must have length nlay ({nlay}), "
            f"but has length {k_array.shape[0]}.")
    else:
        raise ValueError(
        "k_array must be either a scalar or a 1D array of length nlay.")
    
    # Initialize the stress period data dictionary
    drn_spd = {}
    drn_entries = []

    for k, i, j in cells:
        cell_bottom = ztop[k, i, j] - thickness_array[k, i, j]
        if elev_type == "proportion":
            drn_elev = ztop[k, i, j] - (a * thickness_array[k, i , j])
        elif elev_type == "absolute":
            drn_elev = ztop[k, i, j] - a
            if drn_elev < cell_bottom:
                drn_elev = ztop[k, i, j] - (0.1 * thickness_array[k, i , j])

        assert ztop[k, i, j] >= drn_elev >= cell_bottom, (
            f"Inconsistent elevations for cell (k={k}, i={i}, j={j}): "
            f"ztop={ztop[k, i, j]}, drn_elev={drn_elev}, cell_bottom={cell_bottom}")

        drn_cond = (k_array[k] * drain_length * drain_width) / (drainbed_thickness)

        if conc is not None:
            cell_conc = conc[i, j] if hasattr(conc, "shape") else conc
            drn_entries.append((k, i, j, drn_elev, drn_cond, cell_conc))
        else:
            drn_entries.append((k, i, j, drn_elev, drn_cond))

    drn_spd[0] = drn_entries
    return drn_spd

def extract_active_cells(irch, idomain):
    """
    Extract active cell indices (k, i, j) from irch and idomain arrays.

    Parameters:
        irch: 2D array-like of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain: 3D array-like of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).

    Returns:
        List[Tuple[int, int, int]]: List of (k, i, j) indices for active cells.
    """

    # Input checks
    nrow, ncol = irch.shape
    nlay, idom_nrow, idom_ncol = idomain.shape
    if (idom_nrow, idom_ncol) != (nrow, ncol):
        raise ValueError(f"idomain shape (nlay, nrow, ncol) must match irch shape (nrow, ncol) in last two dimensions. Got {idomain.shape} and {irch.shape}.")
    
    active_cells = []
    for i in range(nrow):
        for j in range(ncol):
            k = int(irch[i, j])
            if 0 <= k < nlay and idomain[k, i, j] == 1:
                active_cells.append((k, i, j))
    return active_cells

def extract_active_cells_n(irch, idomain, n):
    """
    Extract active cell indices (k, i, j) from irch and idomain arrays,
    checking every n-th column.

    Parameters:
        irch: 2D array-like of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain: 3D array-like of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).
        n (int): Step size for column indexing. The function will check the 1st, n-th, 2n-th, etc., columns.

    Returns:
        list of (k, i, j) indices for active cells checked every n-th column.
    """

    # Input checks
    nrow, ncol = irch.shape
    nlay, idom_nrow, idom_ncol = idomain.shape
    if (idom_nrow, idom_ncol) != (nrow, ncol):
        raise ValueError(f"idomain shape (nlay, nrow, ncol) must match irch shape (nrow, ncol) in last two dimensions. Got {idomain.shape} and {irch.shape}.")
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer.")
    nrow, ncol = irch.shape
    
    active_cells = []
    for i in range(nrow):
        j_vals = np.arange(0, ncol - 1, n)  # select every n-th column, excluding the last one
        k_vals = irch[i, j_vals]            # extract layer numbers for selected columns
        active_mask = idomain[k_vals, i, j_vals] == 1
        for k, j, active in zip(k_vals, j_vals, active_mask):
            if active:
                active_cells.append((int(k), i, int(j)))

    return active_cells

def extract_active_cells_range(irch, idomain, row_start, row_end, col_start, col_end):
    """
    Extract top most active cell indices (k, i, j) from irch and idomain arrays,
    within a specified submatrix defined by row and column ranges.

    Parameters:
        irch: 2D array-like of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain: 3D array-like of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).
        row_start (int): Starting row index (inclusive).
        row_end (int): Ending row index (inclusive).
        col_start (int): Starting column index (inclusive).
        col_end (int): Ending column index (inclusive).

    Returns:
        list of (k, i, j) indices for top most active cells within the specified submatrix.
    """
    # Input checks
    nrow, ncol = irch.shape
    nlay, idom_nrow, idom_ncol = idomain.shape
    if (idom_nrow, idom_ncol) != (nrow, ncol):
        raise ValueError(f"idomain shape (nlay, nrow, ncol) must match irch shape (nrow, ncol) in last two dimensions. Got {idomain.shape} and {irch.shape}.")
    if not (0 <= row_start <= row_end < nrow):
        raise ValueError(f"Row range {row_start}-{row_end} out of bounds (0–{nrow-1})")
    if not (0 <= col_start <= col_end < ncol):
        raise ValueError(f"Column range {col_start}-{col_end} out of bounds (0–{ncol-1})")
    active_cells = []

    for i in range(row_start, row_end + 1):
        j_vals = np.arange(col_start, col_end + 1)
        k_vals = irch[i, j_vals]
        active_mask = idomain[k_vals, i, j_vals] == 1

        for k, j, active in zip(k_vals, j_vals, active_mask):
            if active:
                active_cells.append((int(k), i, int(j)))

    return active_cells

def extract_active_cells_n_range(irch, idomain, n, col_start, col_end):
    """
    Extract active cell indices (k, i, j) from irch and idomain arrays,
    checking every n-th column within the column range [col_start, col_end).

    Parameters:
        irch: 2D array-like of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain: 3D array-like of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).
        n (int): Step size for column indexing.
        col_start (int): Starting column index (inclusive).
        col_end (int): Ending column index (exclusive).

    Returns:
        list of (k, i, j) indices for active cells checked every n-th column within the range.
    """

    nrow, ncol = irch.shape
    nlay, idom_nrow, idom_ncol = idomain.shape

    if (idom_nrow, idom_ncol) != (nrow, ncol):
        raise ValueError(f"idomain shape {idomain.shape} does not match irch shape {irch.shape} in last two dimensions.")
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer.")
    if not (0 <= col_start < ncol) or not (0 < col_end <= ncol) or col_start >= col_end:
        raise ValueError("Invalid column range specified.")

    active_cells = []
    for i in range(nrow):
        # select columns within the given range, stepping by n
        j_vals = np.arange(col_start, col_end, n)
        k_vals = irch[i, j_vals]
        for k, j in zip(k_vals, j_vals):
            if 0 <= k < nlay and idomain[k, i, j] == 1:
                active_cells.append((int(k), i, int(j)))

    return active_cells

def extract_active_cells_zone(irch, idomain, zone_array, row_start, row_end, col_start, col_end, zones):
    """
    Extract active cell indices (k, i, j) from irch, idomain, and zone_array,
    within a specified submatrix and filtered by specific zone numbers.

    Parameters:
        irch: 2D array-like of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain: 3D array-like of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).
        zone_array: 3D array-like of shape (nlay, nrow, ncol) with integer zone numbers.
        row_start (int): Starting row index (inclusive).
        row_end (int): Ending row index (inclusive).
        col_start (int): Starting column index (inclusive).
        col_end (int): Ending column index (inclusive).
        zones (list or set): List of zone numbers to include.

    Returns:
        list of (k, i, j) indices for active cells within the specified submatrix
        that belong to the specified zones.
    """

    nrow, ncol = irch.shape
    nlay, idom_nrow, idom_ncol = idomain.shape
    if (idom_nrow, idom_ncol) != (nrow, ncol):
        raise ValueError(f"idomain shape {idomain.shape} incompatible with irch shape {irch.shape}")
    if zone_array.shape != idomain.shape:
        raise ValueError(f"zone_array shape {zone_array.shape} must match idomain shape {idomain.shape}")
    if not (0 <= row_start <= row_end < nrow):
        raise ValueError(f"Row range {row_start}-{row_end} out of bounds (0–{nrow-1})")
    if not (0 <= col_start <= col_end < ncol):
        raise ValueError(f"Column range {col_start}-{col_end} out of bounds (0–{ncol-1})")

    active_cells = []

    for i in range(row_start, row_end + 1):
        j_vals = np.arange(col_start, col_end + 1)
        k_vals = irch[i, j_vals]
        # mask of active cells in idomain
        active_mask = idomain[k_vals, i, j_vals] == 1
        # mask of cells in specified zones
        zone_mask = np.isin(zone_array[k_vals, i, j_vals], zones)
        # combined mask
        mask = active_mask & zone_mask

        for k, j, include in zip(k_vals, j_vals, mask):
            if include:
                active_cells.append((int(k), i, int(j)))

    return active_cells

def create_icelltype(cutoff, nlay, nrow, ncol, side="left"):
    """
    Create a 3D array (nlay, nrow, ncol) with 1's and 0's based on cutoff column.
    1: Convertible cell
    0: Confined cell

    Parameters
    ----------
    cutoff : int
        Column index where the unconfined/confined transition happens.
    side : str
        "left": columns before cutoff are set to 0. Confined to the left
        "right": columns after cutoff are set to 0. Confined to the right
    nlay, nrow, ncol : int
        Dimensions of the output array.

    Returns
    -------
    icelltype : np.ndarray
        Array of shape (nlay, nrow, ncol).
    """

    icelltype = np.zeros((nlay, nrow, ncol), dtype=int)

    if side == "left":
        icelltype[..., cutoff:] = 1
    elif side == "right":
        icelltype[..., :cutoff] = 1
    else:
        raise ValueError("side must be either 'left' or 'right'.")

    return icelltype

def linear_gradient_array(h1, h2, nlay, nrow, ncol):
    """
    Create a 3D array with values linearly varying from h1 (first column)
    to h2 (last column), constant across layers and rows. Useful for stablishing
    synthetic intial conditions for a 2 cross-sectional model.

    Parameters
    ----------
    h1 : float
        Value at the first column.
    h2 : float
        Value at the last column.
    nlay, nrow, ncol : int
        Dimensions of the output array.

    Returns
    -------
    arr : np.ndarray
        Array of shape (nlay, nrow, ncol).
    """
    
    # Create linear values along the column axis
    col_values = np.linspace(h1, h2, ncol)

    # Broadcast to full 3D shape
    arr = np.tile(col_values, (nlay, nrow, 1))

    return arr

def export_grid_topview(nrow, ncol, drow, dcol, irch, out_shp="grid_topview.shp", crs="EPSG:4326"):
    """
    Export a top-view of a structured grid as a polygon shapefile with specified CRS.
    """
    # ensure arrays
    drow = np.atleast_1d(drow)
    dcol = np.atleast_1d(dcol)

    if drow.size == 1:
        drow = np.repeat(drow, nrow)
    if dcol.size == 1:
        dcol = np.repeat(dcol, ncol)

    # build coordinate edges
    x_edges = np.concatenate([[0], np.cumsum(dcol)])
    y_edges = np.concatenate([[0], np.cumsum(drow)])

    cells = []
    for i in range(nrow):
        for j in range(ncol):
            k = irch[i, j]
            if k < 0:  # skip inactive cells
                continue
            x0, x1 = x_edges[j], x_edges[j+1]
            y0, y1 = y_edges[nrow - i - 1], y_edges[nrow - i] 
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            cells.append({
                "geometry": poly,
                "row": i,
                "col": j,
                "irch": int(k)
            })

    # assign EPSG:4326 CRS
    gdf = gpd.GeoDataFrame(cells, crs=crs)
    gdf.to_file(out_shp)
    print(f"Exported top-view grid to {out_shp} with CRS {crs}")

def active_cells_from_line(grid_shp, river_shp):
    
    """
    Extracts a list of tupples with the active cells being intercepted by a line feature.

    grid_shp: Path to the grid shapefile created with modbound6.export_grid_topview
    river_shp: path to the river shapefile drawn over the grid. Must be created with the same crs.

    cell_ids: list of tupples with intercepted cell ids. 
    """
    # Load both
    grid = gpd.read_file(grid_shp)
    river = gpd.read_file(river_shp)

    # Ensure CRS match
    if grid.crs != river.crs:
        river = river.to_crs(grid.crs)

    # Spatial join: find polygons (cells) that intersect the river line
    inter = gpd.sjoin(grid, river, how="inner", predicate="intersects")

    # Extract cell ids (k,i,j)
    cell_ids = [(int(row.irch), int(row.row), int(row.col)) for idx, row in inter.iterrows()]
    return cell_ids