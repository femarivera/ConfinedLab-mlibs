# ==========================================================================================
#  modgeom6.py - Modular Utilities for Synthetic Geometry Generation of Multilayer Systems
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project 
#
#  DESCRIPTION:
#  ------------
#  As part of the ConfinedLab project, this module provides robust, flexible, and well-documented utilities 
#  for generating and manipulating synthetic multilayer groundwater model geometries using structured grids.
#  The approach generates a 3D rectangular grid with defined number of layers, rows, and columns. The geometry
#  to be generated will always correspond to a multiayer system with a dip direction along the column axis. 
#
#  MAIN FEATURES:
#  --------------
#  - Create idomain arrays for left- or right-dipping systems with customizable outcrop and cofined areas.
#  - Compute top elevation arrays with options for smooth linear transitions and sloping topography.
#  - Compute thickness arrays with or without transition zones.
#  - Calculate bottom elevations, irch arrays, recharge arrays, and more.
#  - All functions include input validation for clarity and robustness.
#  
#  DEPENDENCIES:
#  -------------
#  - numpy
#  Default parameters on the main functions are set to generate geometries for systems with defined outcropping 
#  and confined areas. This usually generates a simple yet realistic geometry of a multilayer system representing
#  the typical stratigraphic configuration of a sedimentary basin without faulting or folding. 

import numpy as np

def compute_idomain(nlay, nrow, ncol, outcrop_cells, direction = "right"):
    """
    Create an idomain array for a synthetic multilayer system dipping to the left or right.

    Parameters:
        nlay (int): Number of layers.
        nrow (int): Number of rows.
        ncol (int): Number of columns.
        outcrop_cells (1D array): 1D array of length (nlay), with the column indices (int) representing
            the threshold for each layer. For 'left', each layer is active for columns <= outcrop_cells[i].
            For 'right', each layer is active for columns >= outcrop_cells[i].
        direction (str): "right" (default) for right-dipping (confined to the right side), 
                         "left" for left-dipping (confined to the left side).

    Returns:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 for active and 0 for inactive cells.
    """
    #input checks
    outcrop_cells = np.asarray(outcrop_cells)
    if len(outcrop_cells) != nlay:
        raise ValueError("outcrop_cells must have length equal to nlay.")
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'.")
    if np.any(outcrop_cells < 0) or np.any(outcrop_cells > ncol):
        raise ValueError("All outcrop_cells values must be in the range [0, ncol].")

    if direction == "left":
        if not np.all(np.diff(outcrop_cells) >= 0):
            raise ValueError("For 'left', outcrop_cells must be ascending or constant.")
    else:  # direction == "right"
        if not np.all(np.diff(outcrop_cells) <= 0):
            raise ValueError("For 'right', outcrop_cells must be descending or constant.")
    # Initialize idomain array with ones (active cells)
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    # Apply the condition for each layer using the corresponding outcrop length
    for layer in range(nlay - 1):
        L = int(outcrop_cells[layer])
        if direction == "left":
            idomain[layer, :, L:] = 0  
        else:  # direction == "right"
            idomain[layer, :, :L] = 0 

    # Last layer remains fully active
    return idomain

def compute_top(
    idomain,
    outcrop_z,
    transition=True,
    slope=True,
    direction="right",
    transition_cells=None,
    transition_type="contain",
    outcrop_zmin=None,
    outcrop_zmax=None
):
    """
    Generalized function to compute top elevations for multilayer systems with options for:
    - left/right dipping systems,
    - smooth transitions between outcrops,
    - contained or extended transitions,
    - sloping or flat outcrop elevations.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z (1D array-like): Array of top elevations for each layer (length nlay).
        transition (bool): If True, add transition zones between outcrops. If False, use simple top assignment.
        slope (bool): If True (default), use sloping topography (requires outcrop_zmin and outcrop_zmax). If False, use flat outcrop_z.
        direction (str): "left" or "right" (default "right"). Direction of system dip/outcrop.
        transition_cells (int or None): Number of columns for the transition zone between layers. Required if transition=True.
        transition_type (str): "contain" (default) or "extend". "contain" keeps transitions within idomain, "extend" allows transitions to extend beyond.
        outcrop_zmin (1D array-like or None): Minimum elevation for each layer (required if slope=True).
        outcrop_zmax (1D array-like or None): Maximum elevation for each layer (required if slope=True).

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations.
    """

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z = np.asarray(outcrop_z)
    if outcrop_z.shape[0] != nlay:
        raise ValueError("outcrop_z must have length equal to nlay.")
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'.")

    if not transition:
        # No transition zone: ignore slope and related parameters
        top = np.zeros((nrow, ncol), dtype=float)
        active_layers = np.sum(idomain, axis=0)
        irch = nlay - active_layers
        for layer_id in range(nlay):
            top[irch == layer_id] = outcrop_z[layer_id]
        return top

    # If transition is True, check transition parameters
    if transition_cells is None or not isinstance(transition_cells, int) or transition_cells < 0:
        raise ValueError("transition_cells must be a positive integer when transition=True.")
    if transition_type not in ("contain", "extend"):
        raise ValueError("transition_type must be 'contain' or 'extend'.")

    # If slope is True, check slope parameters
    if slope:
        if outcrop_zmin is None or outcrop_zmax is None:
            raise ValueError("outcrop_zmin and outcrop_zmax must be provided when slope=True.")
        outcrop_zmin = np.asarray(outcrop_zmin)
        outcrop_zmax = np.asarray(outcrop_zmax)
        if outcrop_zmin.shape[0] != nlay or outcrop_zmax.shape[0] != nlay:
            raise ValueError("outcrop_zmin and outcrop_zmax must have length equal to nlay.")

    # Compute topmost active layer index (irch)
    active_layers = np.sum(idomain, axis=0)  # (nrow, ncol)
    irch = nlay - active_layers  # Topmost active layer index per cell
    top = np.zeros((nrow, ncol), dtype=float)

    # Step 1: Assign base top elevations
    if slope:
        for layer_id in range(nlay):
            for row in range(nrow):
                mask = (irch[row, :] == layer_id)
                n_cells = np.sum(mask)
                if n_cells > 0:
                    if direction == "right":
                        slope_vals = np.linspace(outcrop_zmax[layer_id], outcrop_zmin[layer_id], n_cells)
                    else:
                        slope_vals = np.linspace(outcrop_zmin[layer_id], outcrop_zmax[layer_id], n_cells)
                    top[row, mask] = slope_vals
    else:
        for layer_id in range(nlay):
            top[irch == layer_id] = outcrop_z[layer_id]

    # Step 2: Add transitions
    for layer_id in range(nlay):
        if direction == "right":
            # For right-dipping, transitions are to the left (lower column indices)
            transition_mask = (irch == layer_id) & (np.roll(irch, 1, axis=-1) == layer_id + 1)
        else:
            # For left-dipping, transitions are to the right (higher column indices)
            transition_mask = (irch == layer_id) & (np.roll(irch, -1, axis=-1) == layer_id + 1)

        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                if direction == "right":
                    if transition_type == "extend":
                        start = max(0, idx - transition_cells + 1)
                        end = idx + 1
                    else:  # "contain"
                        start = idx
                        end = min(ncol-1, idx + transition_cells)
                    n = end - start
                    if n > 1:
                        if slope:
                            top[row, start:end] = np.linspace(
                                outcrop_zmin[layer_id + 1], top[row, end], n
                            )
                        else:
                            top[row, start:end] = np.linspace(
                                outcrop_z[layer_id + 1], outcrop_z[layer_id], n
                            )
                else:
                    # direction == "left"
                    if transition_type == "extend":
                        start = idx
                        end = min(ncol-1, idx + transition_cells)
                    else:  # "contain"
                        start = max(0, idx - transition_cells + 1)
                        end = idx + 1
                    n = end - start
                    if n > 1:
                        if slope:
                            top[row, start:end] = np.linspace(
                                outcrop_zmax[layer_id], top[row, end], n
                            )
                        else:
                            top[row, start:end] = np.linspace(
                                outcrop_z[layer_id], outcrop_z[layer_id + 1], n
                            )

    return top

def compute_thickness(
    idomain,
    base_thicknesses,
    transition=True,
    transition_cells=None,
    transition_type="contain"
):
    """
    Generalized function to compute thickness arrays for multilayer systems with options for:
    - simple thickness assignment (no transition),
    - smooth transitions between active/inactive zones,
    - contained or extended transitions.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        base_thicknesses (1D array-like): Array of length nlay, with the base thickness for each model layer.
        transition (bool): If True, add transition zones between active/inactive areas. If False, use simple assignment.
        transition_cells (int or None): Number of columns for the transition zone. Required if transition=True.
        transition_type (str): "contain" (default) or "extend". "contain" keeps transitions within idomain, "extend" allows transitions to extend beyond.

    Returns:
        thickness_array (numpy.ndarray): 3D array (nlay, nrow, ncol) with thicknesses.
    """

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    base_thicknesses = np.asarray(base_thicknesses)
    if base_thicknesses.shape[0] != nlay:
        raise ValueError("base_thicknesses must have length equal to nlay.")

    if not transition:
        # Simple assignment
        thickness_array = np.zeros_like(idomain, dtype=float)
        for layer in range(nlay):
            thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thicknesses[layer], 0)

        return thickness_array

    # If transition is True, check transition parameters
    if transition_cells is None or not isinstance(transition_cells, int) or transition_cells < 0:
        raise ValueError("transition_cells must be a positive integer when transition=True.")
    if transition_type not in ("contain", "extend"):
        raise ValueError("transition_type must be 'contain' or 'extend'.")

    if transition_type == "contain":
        # Contained transition (within idomain)
        # Initialize the thickness array
        nlay, nrow, ncol = idomain.shape
        thickness_array = np.zeros_like(idomain, dtype=float)

        # Loop through layers and apply base thickness and smooth transition
        for layer in range(nlay):
            # Get base thickness for the current layer
            base_thickness = base_thicknesses[layer]
            
            # Set thickness for active cells (idomain == 1) to the base thickness
            thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thickness, 0)
            
            # Now we smooth the transition from base thickness to zero within the same layer
            for row in range(nrow):
                for col in range(ncol):
                    if idomain[layer, row, col] == 0:  # If the cell is inactive
                        # Check if there are adjacent active cells to create a smooth transition
                        if col > 0 and idomain[layer, row, col - 1] == 1:  # Transition from left
                            transition_range = np.linspace(0, base_thickness, transition_cells)
                            # Apply transition over the next `transition_cells` columns
                            for t in range(min(transition_cells, ncol - col)):
                                thickness_array[layer, row, col - t] = transition_range[t]
                        elif col < ncol - 1 and idomain[layer, row, col + 1] == 1:  # Transition from right
                            transition_range = np.linspace(0, base_thickness, transition_cells)
                            # Apply transition over the previous `transition_cells` columns
                            for t in range(min(transition_cells, col + 1)):
                                thickness_array[layer, row, col + t] = transition_range[t]
        
        return thickness_array
    else:
        # Extended transition (beyond idomain)
        # Initialize the thickness array
        nlay, nrow, ncol = idomain.shape
        thickness_array = np.zeros_like(idomain, dtype=float)

        # Loop through layers and apply base thickness and smooth transition
        for layer in range(nlay):
            # Get base thickness for the current layer
            base_thickness = base_thicknesses[layer]
            
            # Set thickness for active cells (idomain == 1) to the base thickness
            thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thickness, 0)
            
            # Now we smooth the transition from base thickness to zero within the same layer
            for row in range(nrow):
                for col in range(ncol):
                    if idomain[layer, row, col] == 0:  # If the cell is inactive
                        # Check if there are adjacent active cells to create a smooth transition
                        if col > 0 and idomain[layer, row, col - 1] == 1:  # Transition from left
                            transition_range = np.linspace(base_thickness, 0, transition_cells)
                            # Apply transition over the next `transition_cells` columns
                            for t in range(min(transition_cells, ncol - col)):
                                thickness_array[layer, row, col + t] = transition_range[t]
                        elif col < ncol - 1 and idomain[layer, row, col + 1] == 1:  # Transition from right
                            transition_range = np.linspace(base_thickness, 0, transition_cells)
                            # Apply transition over the previous `transition_cells` columns
                            for t in range(min(transition_cells, col + 1)):
                                thickness_array[layer, row, col - t] = transition_range[t]
        
        return thickness_array

def compute_bottom(ztop, thickness_array):
    """
    Compute the bottom elevations for each layer based on ztop and thickness_array.

    Parameters:
        ztop (ndarray): 2D array of shape (nrow, ncol), representing the top elevation of the model.
        thickness_array (ndarray): 3D array of shape (nlay, nrow, ncol), with the thickness for each layer.

    Returns:
        bottom (ndarray): 3D array of shape (nlay, nrow, ncol) representing the bottom elevations for each layer.
    """

    # Input checks
    if thickness_array.ndim != 3:
        raise ValueError("thickness_array must be a 3D array (nlay, nrow, ncol).")
    if ztop.ndim != 2:
        raise ValueError("ztop must be a 2D array (nrow, ncol).")
    nlay, nrow, ncol = thickness_array.shape
    if ztop.shape != (nrow, ncol):
        raise ValueError("ztop shape must match (nrow, ncol) of thickness_array.")

    # Initialize the bottom array
    bottom = np.zeros_like(thickness_array, dtype=float)

    # Compute the bottom elevations for each layer
    for layer in range(nlay):
        if layer == 0:
            # For the first layer, bottom elevation is simply ztop - thickness
            bottom[layer, :, :] = ztop - thickness_array[layer, :, :]
        else:
            # For subsequent layers, subtract the thickness of the current layer from the previous layer's bottom
            bottom[layer, :, :] = bottom[layer - 1, :, :] - thickness_array[layer, :, :]
    
    return bottom

def idomain_from_thickness(thickness_array, epsilon):
    """
    Create an idomain array from the thickness array.

    Parameters:
        thickness_array (numpy.ndarray): 3D array (nlay, nrow, ncol) containing thickness values for each layer.
        epsilon (float): Thickness threshold under which cells are deactivated.

    Returns:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 (active) where thickness > epsilon and 0 (inactive) otherwise.
    """

    # Input checks
    if thickness_array.ndim != 3:
        raise ValueError("thickness_array must be a 3D array (nlay, nrow, ncol).")
    if not isinstance(epsilon, (float, int)) or epsilon < 0:
        raise ValueError("epsilon must be a non-negative number.")

    # Set idomain to 1 (active) where thickness > epsilon, else 0 (inactive)
    idomain = np.where(thickness_array > epsilon, 1, 0)
    
    return idomain

def idomain_from_zone(zone_array, zones):
    """
    Create an idomain array from the zone array.

    Parameters:
        zone_array (numpy.ndarray): 3D array (nlay, nrow, ncol) containing zone index for each layer.
        zones (list or array-like): Zone indexes of the layers to be deactivated.

    Returns:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 0 (inactive) for specified zones,
                                 1 (active) otherwise.
    """

    # Input checks
    if zone_array.ndim != 3:
        raise ValueError("zone_array must be a 3D array (nlay, nrow, ncol).")

    # Start with all active
    idomain = np.ones_like(zone_array, dtype=int)

    # Deactivate where zone_array matches any of the zones
    mask = np.isin(zone_array, zones)
    idomain[mask] = 0

    return idomain

def compute_irch(idomain):
    """
    Calculate the topmost active layer index (irch) for each cell.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 for active and 0 for inactive cells.

    Returns:
        irch (numpy.ndarray): 2D array (nrow, ncol) where each value is the index of the topmost active layer.
    """

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    # # Calculate the number of layers
    # nlay = idomain.shape[0]
    # # Sum idomain across the layers
    # active_layers = np.sum(idomain, axis=0)
    # # Calculate irch
    # irch = nlay - active_layers
    active = idomain == 1
    irch = np.argmax(active, axis=0)          # first active layer
    irch[~np.any(active, axis=0)] = -1       # mark fully inactive columns
    return irch

def compute_recharge(irch, R):
    """
    Compute the recharge (surface recharge) array based on irch and layer-specific recharge rates.

    Parameters:
        irch (numpy.ndarray): 2D array of shape (nrow, ncol), containing layer indices (0 to nlay-1).
        R (numpy.ndarray or list): 1D array of shape (nlay), containing recharge rates for each layer.

    Returns:
        numpy.ndarray: 2D array of shape (nrow, ncol) with recharge values assigned based on irch.
    """ 

    # Input checks
    if irch.ndim != 2:
        raise ValueError("irch must be a 2D array (nrow, ncol).")
    R = np.asarray(R)
    nlay = np.max(irch) + 1
    #if R.shape[0] != nlay:
    #    raise ValueError("R must have length equal to the number of layers in irch (max(irch)+1).") 
    # Deactivate: This limits the generation of certain geometries where some layers might appear completely confined
    if np.any((irch < 0) | (irch >= nlay)):
        raise ValueError("All values in irch must be valid layer indices (0 to nlay-1).")

    # Initialize a recharge array with the same shape as irch
    rch = np.zeros_like(irch, dtype=float)

    # Assign recharge rates based on the layer index in irch
    for layer_idx, recharge_rate in enumerate(R):
        rch[irch == layer_idx] = recharge_rate

    return rch

def compute_ztop_array(ztop, zbot):
    """
    Create a 3D array of top elevations for each layer, where the first layer uses ztop and
    subsequent layers use the bottom elevation of the layer above. Usually used for starting conditions
    in certain groundwater models.

    Parameters:
        ztop (numpy.ndarray): 2D array (nrow, ncol), top elevation of the model.
        zbot (numpy.ndarray): 3D array (nlay, nrow, ncol), bottom elevations for each layer.

    Returns:
        ztop_array (numpy.ndarray): 3D array (nlay, nrow, ncol) of top elevations for each layer.
    """

    # Input checks
    if ztop.ndim != 2:
        raise ValueError("ztop must be a 2D array (nrow, ncol).")
    if zbot.ndim != 3:
        raise ValueError("zbot must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = zbot.shape
    if ztop.shape != (nrow, ncol):
        raise ValueError("ztop shape must match (nrow, ncol) of zbot.")

    # Initialize the ztop_array
    ztop_array = np.zeros((nlay, nrow, ncol))  # Initialize the start array
    
    # Assign ztop to the first layer
    ztop_array[0, :, :] = ztop
    
    # Assign each subsequent layer from zbot
    for i in range(1, nlay):
        ztop_array[i, :, :] = zbot[i - 1, :, :]
    
    return ztop_array

def compute_3Darray(values_1d, nlay, nrow, ncol, dtype=float):
    """
    Expands a 1D array of layer values to a layered 3D array, assigning each value to active cells in the corresponding layer.

    Args:
        values_1d (np.ndarray): 1D array of length nlay, with values for each layer.
        nlay (int): Number of layers.
        nrow (int): Number of rows.
        ncol (int): Number of columns.
        dtype (data-type, optional): Desired data type for the output array. Default is float.

    Returns:
        np.ndarray: 3D array of shape (nlay, nrow, ncol), with each active cell in layer i assigned values_1d[i], and np.nan elsewhere.

    Raises:
        ValueError: If input shapes are inconsistent or invalid.
    """

    # Input checks
    if not isinstance(values_1d, np.ndarray):
        raise ValueError("values_1d must be a numpy array.")
    if values_1d.ndim != 1:
        raise ValueError("values_1d must be a 1D array.")
    if values_1d.shape[0] != nlay:
        raise ValueError("Length of values_1d must match number of layers (nlay).")

    arr3d = np.empty((nlay, nrow, ncol), dtype=dtype)
    for ilay in range(nlay):
        arr3d[ilay, :, :] = values_1d[ilay]
    return arr3d

def subdivide_layers(idomain, ztop, zbot, nsub_layers):
    """
    Subdivide structured grid layers into thinner layers according to a list of nsub per layer.

    Parameters
    ----------
    idomain : ndarray, shape (nlay, nrow, ncol)
        Active/inactive array.
    ztop : ndarray, shape (nlay, nrow, ncol)
        Top elevation of each cell.
    zbot : ndarray, shape (nlay, nrow, ncol)
        Bottom elevation of each cell.
    nsub_layers : list or array of int, length = nlay
        Number of subdivisions for each parent layer.

    Returns
    -------
    idomain_new, ztop_new, zbot_new, thickness_new : ndarray
        New arrays after subdivision.
        size (sum(nsub_layers), nrow, ncol)
    """
    nlay, nrow, ncol = idomain.shape
    nlay_new = sum(nsub_layers)

    idomain_new = np.empty((nlay_new, nrow, ncol), dtype=idomain.dtype)
    ztop_new = np.empty((nlay_new, nrow, ncol))
    zbot_new = np.empty((nlay_new, nrow, ncol))

    idx_new = 0
    for ilay in range(nlay):
        nsub = nsub_layers[ilay]
        for isub in range(nsub):
            frac_top = isub / nsub
            frac_bot = (isub + 1) / nsub
            idomain_new[idx_new] = idomain[ilay]
            ztop_new[idx_new] = ztop[ilay] - (ztop[ilay] - zbot[ilay]) * frac_top
            zbot_new[idx_new] = ztop[ilay] - (ztop[ilay] - zbot[ilay]) * frac_bot
            idx_new += 1
    
    thickness_new = ztop_new - zbot_new

    return nlay_new, idomain_new, ztop_new, zbot_new, thickness_new

def subdivide_array(arr, nsub_layers):
    """
    Repeat an array along the first axis according to custom subdivisions.

    Parameters
    ----------
    arr : ndarray, shape (nlay, ...)
        Original array (1D, 2D, or 3D) where axis=0 is layers.
    nsub_layers : list or array of int, length = nlay
        Number of subdivisions for each parent layer.

    Returns
    -------
    arr_new : ndarray
        New array with shape (sum(nsub_layers), ...)
    """
    return np.concatenate(
        [np.repeat(arr[[i]], nsub_layers[i], axis=0) for i in range(len(nsub_layers))],
        axis=0)

def storage_coefficient(sy_cells, idomain, ss, sy, thickness):
    """
    sy_cells: list of tuples
        List of (k, i, j) indices where specific yield (sy) should be used instead of specific storage (ss).
    idomain: 3D ndarray
        IDOMAIN array (0=inactive, 1=active).
    ss: 3D array
        Specific storage values for each cell (nlay, nrow, ncol).
    sy: 3D array
        Specific yield values for each cell (nlay, nrow, ncol).
    thickness: 3D ndarray
        Thickness of each cell (nlay, nrow, ncol).
    """
    nlay, nrow, ncol = idomain.shape
    
    # Initialize storage coefficient array
    storage_coeff = np.zeros((nlay, nrow, ncol), dtype=float)

    # Start by assigning ss * thickness everywhere
    storage_coeff = ss * thickness # element-wise multiply (3D × 3D)

    # Now overwrite cells in sy_cells with sy for their layer
    for (k, i, j) in sy_cells:
        if idomain[k, i, j] == 1:      # ensure cell is active
            storage_coeff[k, i, j] = sy[k, i, j]

    # inactive cells are nan
    storage_coeff[idomain == 0] = np.nan

    return storage_coeff

def storage_cell_type(sy_cells, idomain):
    nlay, nrow, ncol = idomain.shape
    
    # Initialize output with convertible storage everywhere
    sto_cell_type = np.ones((nlay, nrow, ncol), dtype=float)

    # Now overwrite cells in sy_cells with 0 for confined only
    for (k, i, j) in sy_cells:
        if idomain[k, i, j] == 1:      # ensure cell is active
            sto_cell_type[k, i, j] = 0

    # inactive cells are nan
    sto_cell_type[idomain == 0] = np.nan

    return sto_cell_type

def insert_soil_layer(ztop, zbot, idomain, soil_thickness=5.0):
    """
    Insert a new soil layer over the current model domain.

    The model top (ztop) remains unchanged, and all existing layers
    are shifted downward by `soil_thickness`.

    Parameters
    ----------
    ztop : (nlay, nrow, ncol) array
        Current model layer elevations.
    zbot : (nlay, nrow, ncol) array
        Bottom elevations of each layer (top to bottom order).
    idomain : (nlay, nrow, ncol) array
        IDOMAIN array (0=inactive, 1=active, <0=constant-head).
    soil_thickness : float
        Thickness of the new top soil layer (positive number).

    Returns
    -------
    new_ztop : (nlay, nrow, ncol) array
        Topography stays the same, the top of all layers are shifted.
    new_zbot : (nlay+1, nrow, ncol) array
        Updated bottom elevations with the new top layer inserted.
    new_idomain : (nlay+1, nrow, ncol) array
        Updated IDOMAIN array (soil layer active by default).
    """
    nlay, nrow, ncol = zbot.shape
    model_ztop = ztop[0, :, :]  # shape (nrow, ncol)

    # Define new soil layer
    soil_ztop = model_ztop
    soil_zbot = soil_ztop  - soil_thickness  # bottom of soil layer

    # Shift all existing layers downward by soil_thickness
    shifted_ztop = ztop - soil_thickness
    shifted_zbot = zbot - soil_thickness

    # Stack the new soil layer on top
    new_ztop = np.vstack((soil_ztop[None, :, :], shifted_ztop))
    new_zbot = np.vstack((soil_zbot[None, :, :], shifted_zbot))
    # Add corresponding idomain layer (default = 1)
    new_idomain = np.vstack((np.ones((1, nrow, ncol), dtype=idomain.dtype), idomain))

    new_nlay = nlay + 1

    new_thickness = new_ztop - new_zbot
    return new_ztop, new_zbot, new_idomain, new_nlay, new_thickness

def add_top_value(arr, top_value=None):
    """Insert top_value (or first value to a 1D layer wise array) after adding a soil model layer."""
    if top_value is None:
        top_value = arr[0]
    return np.insert(arr, 0, top_value, axis=0)

def add_top_layer(arr3d, value=None):
    """Insert a top layer to a 3D array (nlay, nrow, ncol) after adding a soil model layer."""
    if value is None:
        value = arr3d[0, :, :]
    return np.vstack((value[None, :, :], arr3d))



