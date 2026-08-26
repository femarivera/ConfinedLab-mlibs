##########################################################################################
#  modtransient6.py - Modular Transient Utilities for MODFLOW 6 Model Time-Series Analysis
##########################################################################################
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  As part of the ConfinedLab project, this module provides utilities for analyzing,
#  processing, and visualizing transient (time-dependent) results from MODFLOW 6 groundwater models.
#
#  MAIN FEATURES:
#  --------------
#  - Extract and process time-series data (heads, flows, budgets) from MODFLOW 6 outputs.
#  - Visualize temporal evolution of model heads and budget components.
#  - Computes the proportions of water flow to wells from storage release and capture rates.
#  - If zones are defined, plots and analyses water budgets for each zone.

import pandas as pd
import numpy as np
import os
import sys
import flopy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score
from scipy.interpolate import griddata

# Import local modules
sys.path.append('..')
from mlibs import modplot6 # type: ignore


def simplify_name(name):
    """
    Simplifies a column or component name for plotting and display.
    If the input string contains parentheses, extracts and returns the text inside them.
    Otherwise, returns the stripped original string.

    Args:
        name (str): The input string to simplify (e.g., a column name).

    Returns:
        str: Simplified name for display or legend.
    """
    # Extract content inside parentheses, if present
    if '(' in name and ')' in name:
        simplified = name.split('(')[1].split(')')[0].strip()  # Extract text inside parentheses
    else:
        simplified = name.strip()  # Fallback if no parentheses are found
    return simplified
   
def plot_head_time_series(head_file_path, 
                          gwf, 
                          output_path, 
                          show=False, 
                          save=False,
                          figsize=(14, 12), 
                          fontsize=14, 
                          tau=None, 
                          time_units='days'):
    """
    Plot MODFLOW 6 simulated groundwater head time series for one or more observation points.

    This function reads head observation data from a CSV file (as exported by Flopy or MODFLOW 6),
    and generates a time series plot for each observation location. It supports optional display
    and saving of the plot, automatic axis scaling, and marking equilibrium times based on a 
    provided time constant (tau).

    Args:
        head_file_path (str): Path to the head observation CSV file.
        gwf (flopy.modflow.ModflowGwf): Flopy groundwater flow model object.
        output_path (str): Path to save the plot if save is True.
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        save (bool, optional): Whether to save the plot to disk. Defaults to False.
        figsize (tuple, optional): Figure size in inches. Defaults to (14, 12).
        fontsize (int, optional): Font size for plot labels and titles. Defaults to 14.
        tau (float or None, optional): Time constant for equilibrium analysis. If provided, 
            vertical lines are drawn at 3*tau (95% equilibrium) and 5*tau (99% equilibrium).
        time_units (str, optional): "days" or "years". Units for time axis label. Defaults to 'days'.
                                    Assumes model inputs in days by default.

    Returns:
        None. Displays and/or saves a plot showing groundwater head values over time for each observation.
    """

    # Retrieve head observation data using Flopy
    #csv = gwf.head_obs.output.obs(f=head_file_path).get_data()
    csv = pd.read_csv(head_file_path)

    fig = plt.figure(figsize=figsize)

    # Plot head values over time
    if time_units == 'days':
        time_axis = csv["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_axis = csv["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    for name in csv.columns[1:]:  # Skip the first column (time)
        plt.plot(time_axis, csv[name], label=name)

    # Automatically adapt the y-axis limits based on the data range
    plt.xlabel(time_axis_label, fontsize=fontsize/1.2)
    plt.ylabel('Head [m]', fontsize=fontsize/1.2)
    plt.title('HEAD TIME SERIES', fontsize=fontsize)
    plt.legend(fontsize=fontsize/1.2)
    plt.grid(True)

    # Plot equilibrium lines if tau is provided
    if tau is not None:
        eq_95 = 3 * tau
        eq_99 = 5 * tau
        plt.axvline(eq_95, color='red', linestyle='--', label=f'95% Equilibrium (3τ) at {eq_95} {time_axis_label}')
        plt.axvline(eq_99, color='blue', linestyle='--', label=f'99% Equilibrium (5τ) at {eq_99} {time_axis_label}')
        plt.legend(fontsize=fontsize/1.2, loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  

    if show:
        plt.tight_layout()
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def time_step_length(stress_period_data):
    """
    Compute the time step lengths for each stress period in a MODFLOW 6 simulation.

    This function calculates the sequence of time step durations for each stress period,
    accounting for both uniform and variable (TSMULT) time stepping. It supports MODFLOW's
    time step multiplier logic, returning a list of lists where each inner list contains
    the time step lengths for one stress period.

    Parameters:
        stress_period_data (list of tuples): Each tuple is (PERLEN, NSTP, TSMULT), where
            PERLEN (float): Length of the stress period.
            NSTP (int): Number of time steps in the stress period.
            TSMULT (float): Time step multiplier (1 for uniform, >1 for variable).

    Returns:
        time_steps (list of lists): Each inner list contains the time step lengths for one stress period.
    """
    time_steps = []
    
    for period_data in stress_period_data:
        perlen, nstp, tsmult = period_data
        period_time_steps = []
        
        if tsmult == 1:
            # If TSMULT is 1, each time step is equal (PERLEN / NSTP)
            delta_t = perlen / nstp
            period_time_steps = [delta_t] * nstp
        else:
            # Calculate the first time step using the formula
            delta_t1 = (perlen * (tsmult - 1)) / ((tsmult ** nstp) - 1)
            period_time_steps = [delta_t1]
            
            # Generate successive time steps by multiplying previous time step by TSMULT
            for i in range(1, nstp):
                delta_t_next = period_time_steps[-1] * tsmult
                period_time_steps.append(delta_t_next)
        
        # Add the period time steps to the main time_steps list
        time_steps.append(period_time_steps)
    
    return time_steps

def generate_cumulative_time(stress_period_data):
    """
    Generate the cumulative simulation time at each time step for a MODFLOW 6 transient model.

    This function computes the cummulative elapsed time for all time steps across all stress periods,
    using the MODFLOW 6 stress period definitions and time step multipliers. The result is a list of
    cumulative times, useful for plotting or indexing time-dependent results.

    Parameters:
        stress_period_data (list of tuples): Each tuple is (PERLEN, NSTP, TSMULT), where
            PERLEN (float): Length of the stress period.
            NSTP (int): Number of time steps in the stress period.
            TSMULT (float): Time step multiplier (1 for uniform, >1 for variable).

    Returns:
        cumulative_time (list of float): Cumulative simulation time at each time step across all stress periods.
    """
    time_steps = time_step_length(stress_period_data)  # Generate the time steps first
    
    cumulative_time = []
    total_time = 0  # Initialize total cumulative time
    
    for period_time_steps in time_steps:
        for t in period_time_steps:
            total_time += t
            cumulative_time.append(total_time)
    
    return cumulative_time

def elapsed_time(stress_period_data, sp_num, ts_num):
    """
    Calculate the total elapsed simulation time up to a specific stress period and time step in MODFLOW 6.

    Parameters:
        stress_period_data (list of tuples): Each tuple is (PERLEN, NSTP, TSMULT), where
            PERLEN (float): Length of the stress period.
            NSTP (int): Number of time steps in the stress period.
            TSMULT (float): Time step multiplier (1 for uniform, >1 for variable).
        sp_num (int): Stress period index (0-based).
        ts_num (int): Time step index within the stress period (0-based).

    Returns:
        elapsed_time (float): Total elapsed simulation time up to the specified stress period and time step.
    """
    # Generate time steps and cumulative time for all stress periods
    time_steps = time_step_length(stress_period_data)
    cumulative_time = generate_cumulative_time(stress_period_data)
    
    elapsed_time = 0
    # Sum the elapsed time for all previous stress periods
    for i in range(sp_num):
        elapsed_time += sum(time_steps[i])
    
    # Add the time steps for the current stress period up to the requested time step
    elapsed_time += sum(time_steps[sp_num][:ts_num+1])
    
    return elapsed_time

def total_sim_time(stress_period_data):
    """
    Returns the total simulation time (total cumulative time at the end of the last stress period).

    Parameters:
    - stress_period_data (list): List of tuples, where each tuple is (PERLEN, NSTP, TSMULT)

    Returns:
    - total_time (float): Total simulation time
    """
    cumulative_time = generate_cumulative_time(stress_period_data)
    return cumulative_time[-1]  # Return the total elapsed time after the last stress period

def process_csv_budget(csv_path):
    """
    Processes a MODFLOW 6 budget CSV to compute water balance components and percentages. Recommended use before
    plotting time series.

    Args:
        csv_path (str): Path to the budget CSV file.

    Outputs:
        Updates the CSV with new columns for induced recharge, captured discharge, storage release, capture rates, 
        percentages, and net flows for each inflow/outflow pair.
    """
    # Load the CSV file
    data = pd.read_csv(csv_path)
    # Prepare data for time series
    time_data = data["time"]

    # Extract the reference inflow and outflow from the first time step (first row)
    reference_inflow = data["TOTAL_IN"].iloc[0]
    reference_outflow = data["TOTAL_OUT"].iloc[0]

    # Identify columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_well = [col for col in data.columns if "WEL" in col and "OUT" in col]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "WEL" not in col and "STO" not in col and col != "TOTAL_OUT"]

    # Storage
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Storage componenets
    columns_storage_ss_in = [col for col in data.columns if "STO" in col and "IN" in col and "SS" in col]
    columns_storage_ss_out = [col for col in data.columns if "STO" in col and "OUT" in col and "SS" in col]
    columns_storage_sy_in = [col for col in data.columns if "STO" in col and "IN" in col and "SY" in col]
    columns_storage_sy_out = [col for col in data.columns if "STO" in col and "OUT" in col and "SY" in col]

    # Compute components
    induced_recharge = data[columns_in].sum(axis=1) - reference_inflow
    discharge = data[columns_out].sum(axis=1)
    captured_discharge = reference_outflow - discharge
    total_pumped = data[columns_well].sum(axis=1)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    from_storage = storage_in - storage_out
    storage_change_rate = storage_out - storage_in
    capture = induced_recharge + captured_discharge
    storage_change_integrals = np.array([np.trapz(storage_change_rate[:i+1], time_data[:i+1]) - 
                                         np.trapz(storage_change_rate[:i], time_data[:i]) 
                                         if i > 0 else 0 for i in range(len(storage_change_rate))])
    storage_change = np.cumsum(storage_change_integrals)

    # Compute percentages (handle division by zero)
    induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
    captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
    from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
    capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

    # Compute storage change rates per drainance, compressibility, and total
    sto_ss = data[columns_storage_ss_out].sum(axis=1) - data[columns_storage_ss_in].sum(axis=1)
    sto_sy = data[columns_storage_sy_out].sum(axis=1) - data[columns_storage_sy_in].sum(axis=1)
    sto_total = sto_ss + sto_sy

    # Add computed components and percentages to the DataFrame
    data["Induced_Recharge"] = induced_recharge
    data["Captured_Discharge"] = captured_discharge
    data["Storage_Release"] = from_storage
    data["Capture"] = capture
    data["Storage_Change_rate"] = storage_change_rate
    data["Storage_Change"] = storage_change
    data["Induced_Recharge_Pct"] = induced_recharge_pct
    data["Captured_Discharge_Pct"] = captured_discharge_pct
    data["Storage_Release_Pct"] = from_storage_pct
    data["Capture_Pct"] = capture_pct
    data["STO-SS"] = sto_ss
    data["STO-SY"] = sto_sy
    data["STO-TOTAL"] = sto_total

    # Compute net flow for each inflow/outflow pair
    net_flow_columns = []
    for col_in in columns_in:
        base_name = col_in[:-3]  # Remove the "_IN" suffix
        matching_out_col = base_name + "_OUT"
        if matching_out_col in columns_out:
            net_flow_col_name = base_name + "_Net_Flow"
            data[net_flow_col_name] = data[col_in] - data[matching_out_col]
            net_flow_columns.append(net_flow_col_name)

    # Overwrite the original file with the updated DataFrame
    data.to_csv(csv_path, index=False)

def process_csv_zonebudget(csv_path):
    """
    Processes a MODFLOW 6 zonebudget CSV to compute water balance components and percentages.Recommended use before
    plotting time series.

    Args:
        csv_path (str): Path to the zone budget CSV file.

    Outputs:
        Updates the CSV with new columns for induced recharge, captured discharge, storage release, capture rates, 
        percentages, and net flows for each inflow/outflow pair, for each zone.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Filter inflow, outflow, and storage columns
    inflow_columns = [
        col for col in df.columns if 
        ("IN" in col or "FROM" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    outflow_columns = [
        col for col in df.columns if 
        ("OUT" in col or "TO" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col and "WEL" not in col
    ]
    storage_out_columns = [col for col in df.columns if "STO" in col and "OUT" in col]
    storage_in_columns = [col for col in df.columns if "STO" in col and "IN" in col]
    pumped_columns = [col for col in df.columns if "WEL" in col and "OUT" in col]

    # Storage componenets
    columns_storage_ss_in = [col for col in df.columns if "STO" in col and "IN" in col and "SS" in col]
    columns_storage_ss_out = [col for col in df.columns if "STO" in col and "OUT" in col and "SS" in col]
    columns_storage_sy_in = [col for col in df.columns if "STO" in col and "IN" in col and "SY" in col]
    columns_storage_sy_out = [col for col in df.columns if "STO" in col and "OUT" in col and "SY" in col]

    # Calculate reference inflow and outflow at time zero (reference state)
    reference_inflow = df.loc[df['totim'] == df['totim'].min(), inflow_columns].sum(axis=1).values[0]
    reference_outflow = df.loc[df['totim'] == df['totim'].min(), outflow_columns].sum(axis=1).values[0]

    # Compute components using vectorized operations
    induced_recharge = df[inflow_columns].sum(axis=1) - reference_inflow
    captured_discharge = reference_outflow - df[outflow_columns].sum(axis=1)
    storage_in = df[storage_in_columns].sum(axis=1)
    storage_out = df[storage_out_columns].sum(axis=1)
    from_storage = storage_in - storage_out
    total_pumped = df[pumped_columns].sum(axis=1)
    capture = induced_recharge + captured_discharge

    # Compute percentages (handle division by zero)
    induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
    captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
    from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
    capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

    # Compute storage change rates per drainance, compressibility, and total
    sto_ss = df[columns_storage_ss_out].sum(axis=1) - df[columns_storage_ss_in].sum(axis=1)
    sto_sy = df[columns_storage_sy_out].sum(axis=1) - df[columns_storage_sy_in].sum(axis=1)
    sto_total = sto_ss + sto_sy

    # Add computed components and percentages to the DataFrame
    df["Induced_Recharge"] = induced_recharge
    df["Captured_Discharge"] = captured_discharge
    df["From_Storage"] = from_storage
    df["Capture"] = capture
    df["Induced_Recharge_Pct"] = induced_recharge_pct
    df["Captured_Discharge_Pct"] = captured_discharge_pct
    df["From_Storage_Pct"] = from_storage_pct
    df["Capture_Pct"] = capture_pct
    df["STO-SS"] = sto_ss
    df["STO-SY"] = sto_sy
    df["STO-TOTAL"] = sto_total

    # Overwrite the CSV file
    df.to_csv(csv_path, index=False)

def plot_bud_time_series(file_path, 
                         output_path, 
                         show=False, 
                         save=False,
                         figsize=(14, 12), 
                         fontsize=14,
                         time_units='days'):
    """
    Creates time series plots for inflow, outflow, storage components, and change in storage over time
    based on a budget summary CSV output of a MODFLOW 6 Transient simulation.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns ending in _IN, _OUT, containing TOTAL_IN, TOTAL_OUT, and columns with STO.
        output_path (str): Path to save the plot if save is True.
        show (bool): Display the plot interactively.
        save (bool): Save the plot to disk.
        figsize (tuple): Figure size in inches.
        fontsize (int): Font size for plot labels.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'. 
                        Assumes model inputs in days by default.

    Outputs:
        A figure with four subplots showing:
        1. Inflow components over time.
        2. Outflow components over time.
        3. Total Inflows and Total Outflows over time.
        4. Cumulative change in Storage over time.
    """

    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Identify columns for inflow, outflow, and total
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "STO" not in col and col != "TOTAL_OUT"]
    columns_total = ["TOTAL_IN", "TOTAL_OUT"]
    data_total_in = data[columns_in].sum(axis=1)
    data_total_out = data[columns_out].sum(axis=1)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Compute storage change in m3
    time_data = data["time"]
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in
    storage_change_integrals = np.array([np.trapz(storage_change_rate[:i+1], time_data[:i+1]) - 
                                         np.trapz(storage_change_rate[:i], time_data[:i]) 
                                         if i > 0 else 0 for i in range(len(storage_change_rate))])
    storage_change = np.cumsum(storage_change_integrals) 
        
    # Find the global max across all relevant columns
    ymax = max(
    data[columns_in].to_numpy().max(),
    data[columns_out].to_numpy().max(),
    data_total_in.max(),
    data_total_out.max())

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Prepare data for time series plotting
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    # Plot inflow components (excluding TOTAL_IN and STO columns)
    for col in columns_in:
        axs[0, 0].plot(time_data, data[col], label=simplify_name(col))
    axs[0, 0].set_title("INFLOW COMPONENTS", fontsize=fontsize)
    axs[0, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 0].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[0, 0].legend(fontsize=fontsize/1.2)
    axs[0, 0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
    axs[0, 0].grid()

    # Plot outflow components (excluding TOTAL_OUT and STO columns)
    for col in columns_out:
        axs[0, 1].plot(time_data, data[col], label=simplify_name(col))
    axs[0, 1].set_title("OUTFLOW COMPONENTS", fontsize=fontsize)
    axs[0, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 1].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[0, 1].legend(fontsize=fontsize/1.2)
    axs[0, 1].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
    axs[0, 1].grid()

    # TOTAL IN and TOTAL OUT
    axs[1, 0].plot(time_data, data_total_in, label="TOTAL INFLOW", color="blue")
    axs[1, 0].plot(time_data, data_total_out, label="TOTAL OUTFLOW", color="red")
    axs[1, 0].set_title("TOTAL FLOWS", fontsize=fontsize)
    axs[1, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 0].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[1, 0].legend(fontsize=fontsize/1.2)
    axs[1, 0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
    axs[1, 0].grid()

    # Plot CHANGE IN STORAGE (STORAGE OUT - STORAGE IN)
    axs[1, 1].plot(time_data, storage_change, label="STORAGE CHANGE", color="green")
    axs[1, 1].set_title("CHANGE IN STORAGE", fontsize=fontsize)
    axs[1, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 1].set_ylabel("m³", fontsize=fontsize/1.2)
    axs[1, 1].legend(fontsize=fontsize/1.2)
    axs[1, 1].grid()

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    
    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_net_flow_time_series(file_path, 
                              output_path, 
                              show=False, 
                              save=False, 
                              figsize=(14, 12), 
                              fontsize=16, 
                              tau=None, 
                              time_units='days'):
    """
    Creates time series plots for the difference between inflow and outflow components.
    Positive values represent net inflows, and negative values represent net outflows.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns ending in _IN, _OUT.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        tau (float or None): Time constant. If provided, vertical lines will be drawn at 3*tau and 5*tau. Defaults to None.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                         Assumes model inputs in days by default.

    Outputs:
        A figure with time series plots showing the difference between inflow and outflow
        for each component.
    """

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify matching inflow and outflow columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_out = [col.replace("_IN", "_OUT") for col in columns_in if col.replace("_IN", "_OUT") in data.columns]

    # Prepare time data
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot net flux for each component
    legend_labels = []
    for col_in, col_out in zip(columns_in, columns_out):
        net_flux = data[col_in] - data[col_out]
        label = simplify_name(col_in)
        ax.plot(time_data, net_flux, label=label)
        legend_labels.append(label)

    # Sort legend labels alphabetically
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])
    handles, labels = zip(*sorted_handles_labels)
    ax.legend(handles, labels, fontsize=fontsize / 1.2)

    # Horizontal line at zero for the X-axis
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax.set_title("Net Flow (Inflow - Outflow) Components", fontsize=fontsize)
    ax.set_xlabel(time_axis_label, fontsize=fontsize / 1.2)
    ax.set_ylabel("Net Flow [m³/day]", fontsize=fontsize / 1.2)
    ax.grid()

    # Plot equilibrium lines if tau is provided
    if tau is not None:
        eq_95 = 3 * tau
        eq_99 = 5 * tau
        ax.axvline(eq_95, color='red', linestyle='--', label=f'95% Equilibrium (3τ) at {eq_95} {time_axis_label}')
        ax.axvline(eq_99, color='blue', linestyle='--', label=f'99% Equilibrium (5τ) at {eq_99} {time_axis_label}')
        ax.legend(fontsize=fontsize / 1.2, loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout()

    if show:
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_water_to_wells(file_path, 
                        output_path, 
                        show = False,
                        save = False,
                        figsize = (14, 12),
                        fontsize = 14, 
                        time_units='days'):
    """
    Plots various water budget components related to well abstraction:
    
    The first time step in the transient simulation should be a steady state stress period
    to evaluate the effects of pumping starting from natural/baseline/reference conditions.

    - Induced Recharge: Inflow components (excluding recharge) 
      (corresponds to the induced inflows from flow boundaries like RIV, GHB, etc.).
    - Decreased Discharge: Outflow components (excluding well abstractions)
      (corresponds to the captured/intercepted discharge).
    - From Storage: Storage release sourcing well abstraction.
    - Capture: Sum of Induced Recharge and Decreased Discharge.
    - Total Pumped: Sum of all well abstractions.
    
    Args:
        file_path (str): Path to the budget CSV file containing water budget data.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.  
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles. Defaults to 14.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                         Assumes model inputs in days by default.
    
    Outputs:
        A figure with four subplots:
        1. Induced Recharge, Decreased Discharge, and From Storage over time.
        2. Capture and From Storage over time.
        3. Percentages of Induced Recharge, Decreased Discharge, and From Storage with respect to Total Pumped.
        4. Percentages of Capture and From Storage with respect to Total Pumped.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Extract the reference inflow and outflow from the first time step (first row)
    reference_inflow = data["TOTAL_IN"].iloc[0]  # value from the first row in "TOTAL_INFLOW"
    reference_outflow = data["TOTAL_OUT"].iloc[0]  # value from the first row in "TOTAL_OUTFLOW"

    # Identify columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_well = [col for col in data.columns if "WEL" in col and "OUT" in col]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "WEL" not in col and "STO" not in col and col != "TOTAL_OUT"]
    
        
    # Storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare time data
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    # Compute components
    induced_recharge = data[columns_in].sum(axis=1) - reference_inflow
    decreased_discharge = data[columns_out].sum(axis=1)
    captured_discharge = reference_outflow - decreased_discharge
    total_pumped = data[columns_well].sum(axis=1)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    from_storage = storage_in - storage_out
    capture = induced_recharge + captured_discharge

    # Compute percentages (handle division by zero)
    induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
    captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
    from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
    capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Induced Recharge, Decreased Discharge, and From Storage
    axs[0, 0].plot(time_data, induced_recharge, label="Induced inflows", color = "blue")
    axs[0, 0].plot(time_data, captured_discharge, label="Captured outflows", color = "red")
    axs[0, 0].plot(time_data, from_storage, label="Storage release", color = "green")
    axs[0, 0].set_title("WATER TO WELLS", fontsize=fontsize)
    axs[0, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 0].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[0, 0].legend(fontsize=fontsize/1.2)
    axs[0, 0].grid()

    # Plot 2: Capture and From Storage
    axs[0, 1].plot(time_data, capture, label="Capture", color="purple")
    axs[0, 1].plot(time_data, from_storage, label="Storage release", color="green")
    axs[0, 1].set_title("CAPTURE AND STORAGE", fontsize=fontsize)
    axs[0, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 1].set_ylabel("m³/day",fontsize=fontsize/1.2)
    axs[0, 1].legend(fontsize=fontsize/1.2)
    axs[0, 1].grid()

    # Plot 3: Percentages of Induced Recharge, Decreased Discharge, and From Storage
    axs[1, 0].plot(time_data, induced_recharge_pct, label="Induced inflows %", color="blue")
    axs[1, 0].plot(time_data, captured_discharge_pct, label="Captured outflows %", color="red")
    axs[1, 0].plot(time_data, from_storage_pct, label="Storage release %", color="green")
    axs[1, 0].set_title("WATER TO WELLS PERCENTAGE", fontsize=fontsize)
    axs[1, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 0].set_ylabel("Percentage (%)",fontsize=fontsize/1.2)
    axs[1, 0].legend(fontsize=fontsize/1.2)
    axs[1, 0].grid()

    # Plot 4: Percentages of Capture and From Storage
    axs[1, 1].plot(time_data, capture_pct, label="Capture %", color="purple")
    axs[1, 1].plot(time_data, from_storage_pct, label="Storage release %", color="green")
    axs[1, 1].set_title("CAPTURE AND STORAGE PERCENTAGE", fontsize=fontsize)
    axs[1, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 1].set_ylabel("Percentage (%)",fontsize=fontsize/1.2)
    axs[1, 1].legend(fontsize=fontsize/1.2)
    axs[1, 1].grid()

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Adjust layout and show plot
    plt.ioff()
    if show:
        plt.tight_layout()
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig) 

def plot_bud_sum_transient(file_path, 
                           time, 
                           output_path, 
                           show = False, 
                           save = False):
    """
    Creates bar plots for inflow, outflow, and total flows from a MODFLOW 6 budget summary CSV at a specified time.

    Args:
        file_path (str): Path to the budget CSV file (one row per time step).
        time (float): Simulation time to plot. Corresponds to the elapsed time in model units.
        output_path (str): Path to save the figure if save is True.
        show (bool): Display the plot interactively.
        save (bool): Save the plot to disk.

    Outputs:
        A figure with subplots for inflow, outflow, total flows, and change in storage.
    """

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "STO" not in col and col != "TOTAL_OUT"]
    columns_total = ["TOTAL_IN", "TOTAL_OUT"]
    columns_storage_in = [col for col in data.columns if col.endswith("_IN") and "STO" in col and col != "TOTAL_IN"]
    columns_storage_out = [col for col in data.columns if col.endswith("_OUT") and "STO" in col and col != "TOTAL_OUT"]

    # Filter the data for the specified time
    data_time = data[data['time'] == time]

    # Check if data for the specified time exists
    if data_time.empty:
        print(f"No data found for time: {time}")
        return

    # Prepare data for plots (we assume the time column has only one row for each time step)
    data_in = data_time[columns_in].iloc[0]
    data_out = data_time[columns_out].iloc[0]
    data_total = data_time[columns_total].iloc[0]
    data_total_in = data_time[columns_in].sum(axis=1).iloc[0]
    data_total_out = data_time[columns_out].sum(axis=1).iloc[0]
    sum_storage_in = data_time[columns_storage_in].sum(axis=1).iloc[0]  # Sum along the rows
    sum_storage_out = data_time[columns_storage_out].sum(axis=1).iloc[0]  # Sum along the rows
    data_storage = sum_storage_out - sum_storage_in

    # Simplify column names for plotting
    columns_in_simplified = [simplify_name(col) for col in columns_in]
    columns_out_simplified = [simplify_name(col) for col in columns_out]

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 4, figsize=(19, 5))

    # Determine the common y-axis range based on the "Total Inflow and Outflow" plot
    common_ylim_max = max(max(data_in.values), max(data_out.values), max(data_total.values), data_storage) * 1.1 # Add 10% padding

    # Plot inflow components
    axs[0].bar(columns_in_simplified, data_in.values, color="blue")
    axs[0].set_title("Inflow Components")
    axs[0].set_xlabel("Component")
    axs[0].set_ylabel("m³/day")
    axs[0].set_ylim(0,common_ylim_max) 
    for i, val in enumerate(data_in.values):
        axs[0].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot outflow components
    axs[1].bar(columns_out_simplified, data_out.values, color="red")
    axs[1].set_title("Outflow Components")
    axs[1].set_xlabel("Component")
    axs[1].set_ylabel("m³/day")
    axs[1].set_ylim(0,common_ylim_max) 
    for i, val in enumerate(data_out.values):
        axs[1].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot total inflow and outflow
    axs[2].bar(["Total Inflows", "Total Outflows"], [data_total_in, data_total_out], color="green")
    axs[2].set_title("Total Inflow and Outflow")
    axs[2].set_xlabel("Component")
    axs[2].set_ylabel("m³/day")
    axs[2].set_ylim(0,common_ylim_max) 
    for i, val in enumerate([data_total_in, data_total_out]):
        axs[2].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot change in storage
    axs[3].bar(["Change in storage"], [data_storage], color="purple")  # Wrap label and value in lists for single bar
    axs[3].set_title("Change in storage")
    axs[3].set_xlabel("Component")
    axs[3].set_ylabel("m³/day")
    axs[3].set_ylim(-common_ylim_max, common_ylim_max )
    # Add text label for the single bar
    axs[3].text(0, data_storage, f"{data_storage:.2f}", ha="center", va="bottom")


    # Adjust layout and show plot
    plt.ioff()
    if show:
        plt.tight_layout()
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig) 

def plot_zone_budget(csv_path, 
                     csv_output_dir,
                     fig_output_dir, 
                     show = False, 
                     save = False, 
                     figsize = (14, 12),
                     fontsize = 14,
                     zone_descriptions = None,
                     time_units = 'days'):
    """
    Plots time series of inflows, outflows, total flows, and storage change for each zone from a budget CSV.

    Args:
        csv_path (str): Path to the zone budget CSV file.
        csv_output_dir (str): Directory to save vertical leakage data.
        fig_output_dir (str): Directory to save the figures if save is True.
        show (bool): Display plots interactively.
        save (bool): Save plots to disk.
        figsize (tuple): Figure size in inches.
        fontsize (int): Font size for plot labels.
        zone_descriptions (dict or None): Optional mapping of zone numbers to descriptions.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                            Assumes model inputs in days by default.

    Outputs:
        Figures for each zone showing inflow, outflow, total flows, storage change, and inter-zone transfers.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Identify unique zones
    zones = df['zone'].unique()
    zone_descriptions = zone_descriptions

    # Filter columns for inflows and outflows
    inflow_columns = [
        col for col in df.columns if 
        ("IN" in col or "FROM" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    outflow_columns = [
        col for col in df.columns if 
        ("OUT" in col or "TO" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    storage_out_columns = [
        col for col in df.columns if "STO" in col and "OUT" in col
    ]
    storage_in_columns = [
        col for col in df.columns if "STO" in col and "IN" in col
    ]
    
    # Create plots for each zone
    for zone in zones:
        # Exclude "FROM/TO" columns containing the zone's own number
        zone_specific_exclude = f"ZONE {int(zone)}"
        zone_inflow_columns = [col for col in inflow_columns if zone_specific_exclude not in col]
        zone_outflow_columns = [col for col in outflow_columns if zone_specific_exclude not in col]
        
        zone_data = df[df['zone'] == zone]
        time_data = zone_data["totim"]

        storage_in = zone_data[storage_in_columns].sum(axis=1)
        storage_out = zone_data[storage_out_columns].sum(axis=1)
        storage_change_rate = storage_out - storage_in
        #Make sure integrals are done with time in original units (days)
        storage_change_integrals = np.array([np.trapz(storage_change_rate[:i+1], time_data[:i+1]) - 
                                         np.trapz(storage_change_rate[:i], time_data[:i]) 
                                         if i > 0 else 0 for i in range(len(storage_change_rate))])
        storage_change = np.cumsum(storage_change_integrals)
        data_in = zone_data[zone_inflow_columns].sum(axis=1)
        data_out = zone_data[zone_outflow_columns].sum(axis=1)

        ymax = max(
        zone_data[zone_inflow_columns].to_numpy().max(),
        zone_data[zone_outflow_columns].to_numpy().max(),
        data_in.max(),
        data_out.max()
)

        # Create a subplot for inflows and outflows
        fig, ax = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        
        # Prepare data for time series plotting
        if time_units == 'days':
            time_data = zone_data["totim"]  # Assuming input time is in days
            time_axis_label = 'Time [days]'
        elif time_units == 'years':
            time_data = zone_data["totim"] / 360  # Convert days to years
            time_axis_label = 'Time [years]'
        else:
            raise ValueError("time_units must be 'days' or 'years'")

        # Plot inflows
        for col in zone_inflow_columns:
            ax[0,0].plot(time_data, zone_data[col], label=simplify_name(col))
        ax[0,0].set_title(f'ZONE {zone} INFLOW COMPONENTS', fontsize=fontsize)
        ax[0,0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[0,0].set_ylabel('Flow [m³/day]', fontsize=fontsize/1.2)
        ax[0,0].legend(fontsize=fontsize/1.2)
        ax[0,0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
        ax[0,0].grid()
        
        # Plot outflows
        for col in zone_outflow_columns:
            ax[0,1].plot(time_data, zone_data[col], label=simplify_name(col))
        ax[0,1].set_title(f'ZONE {zone} OUTFLOWS', fontsize=fontsize)
        ax[0,1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[0,1].set_ylabel('Flow [m³/day]', fontsize=fontsize/1.2)
        ax[0,1].legend(fontsize=fontsize/1.2)
        ax[0,1].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
        ax[0,1].grid()

        # Plot TOTAL IN TOTAL OUT
        ax[1,0].plot(time_data, data_in, label="TOTAL INFLOWS")
        ax[1,0].plot(time_data, data_out, label="TOTAL OUTFLOWS")
        ax[1,0].set_title(f'ZONE {zone} TOTAL FLOWS', fontsize=fontsize)
        ax[1,0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[1,0].set_ylabel('Flow [m³/day]', fontsize=fontsize/1.2)
        ax[1,0].legend(fontsize=fontsize/1.2)
        ax[1,0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
        ax[1,0].grid()
        
        # Plot change in storage
        ax[1,1].plot(time_data, storage_change, label="CHANGE IN STORAGE")
        ax[1,1].set_title(f'ZONE {zone} CHANGE IN STORAGE', fontsize=fontsize)
        ax[1,1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[1,1].set_ylabel('Flow [m³]', fontsize=fontsize/1.2)
        ax[1,1].legend(fontsize=fontsize/1.2)
        ax[1,1].grid()
        
        # Adjust layout and show plot
        plt.ioff()
        if show:
            plt.tight_layout()
            plt.show()

         #Save plot
        if save:
            image_path = os.path.join(fig_output_dir, f'Zone {zone} budget.png')
            fig.savefig(image_path, dpi=300)
            plt.close(fig) 

    # Plot the difference "FROM ZONE x - TO ZONE x" for all other zones
    # Prepare an empty dict to collect vertical leakage data
    vertical_leakage_dict = {}

    # Use unique timesteps
    timesteps = sorted(df['totim'].unique())
    vertical_leakage_dict['totim'] = timesteps

    # Compute FROM-TO difference for each zone
    for zone in zones:
        other_zones = [f"ZONE {int(z)}" for z in zones if z != zone]
        from_columns = [f"FROM {oz}" for oz in other_zones]
        to_columns = [f"TO {oz}" for oz in other_zones]
        
        # Sum over all rows for this zone at each timestep
        differences = []
        for t in timesteps:
            mask = (df['zone'] == zone) & (df['totim'] == t)
            diff = df.loc[mask, from_columns].sum(axis=1).values - df.loc[mask, to_columns].sum(axis=1).values
            # If empty (no row for this timestep), set to 0
            differences.append(diff[0] if len(diff) > 0 else 0)
        
        vertical_leakage_dict[f'ZONE_{zone}'] = differences

    # Create DataFrame
    vertical_leakage_df = pd.DataFrame(vertical_leakage_dict)

    # Save to CSV
    csv_path = os.path.join(csv_output_dir, "vertical_leakage.csv")
    vertical_leakage_df.to_csv(csv_path, index=False)
    print(f"Vertical leakage data saved to {csv_path}")

    # Optional: Plotting
    fig2 = plt.figure(figsize=figsize)
    if time_units == 'days':
        time_data = vertical_leakage_df['totim']
        time_axis_label = 'Time [days]'
    if time_units == 'years':
        time_data = vertical_leakage_df['totim'] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")
    
    for zone in zones:
        description = zone_descriptions.get(zone, f"ZONE {zone}")
        plt.plot(time_data, vertical_leakage_df[f'ZONE_{zone}'], label=f'ZONE {zone} - {description}')

    plt.title('WATER TRANSFERS VIA VERTICAL LEAKAGE', fontsize=fontsize)
    plt.xlabel(time_axis_label, fontsize=fontsize)
    plt.ylabel('Net leakage (Inflows - Outflows) [m³/day]', fontsize=fontsize)
    plt.legend(fontsize=fontsize/1.2)
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        image_path = os.path.join(fig_output_dir, "zonebudget_summary_t.png")
        fig2.savefig(image_path, dpi=300)
        plt.close(fig2)

def plot_water_to_wells_zonebud(csv_path, 
                                output_dir, 
                                show = False, 
                                save = False,
                                fontsize = 14, 
                                time_units = 'days'):
    """
    Plots water budget components and sources of water to wells for each zone from a zone budget CSV.

    Args:
        csv_path (str): Path to the zone budget CSV file.
        output_dir (str): Directory to save figures if save is True.
        show (bool): Display plots interactively.
        save (bool): Save plots to disk.
        fontsize (int): Font size for plot labels.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                         Assumes model inputs in days by default.

    Outputs:
        Figures for each zone showing storage release, induced recharge, captured discharge, capture, and their percentages.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Identify unique zones
    zones = df['zone'].unique()

    # Filter inflow, outflow, and storage columns
    inflow_columns = [
        col for col in df.columns if 
        ("IN" in col or "FROM" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    outflow_columns = [
        col for col in df.columns if 
        ("OUT" in col or "TO" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col and "WEL" not in col
    ]
    storage_out_columns = [
        col for col in df.columns if "STO" in col and "OUT" in col
    ]
    storage_in_columns = [
        col for col in df.columns if "STO" in col and "IN" in col
    ]
    pumped_columns = [
        col for col in df.columns if "WEL" in col and "OUT" in col
    ]

    # Process each zone
    for zone in zones:
        zone_data = df[df['zone'] == zone]

        if time_units == 'days':
            time_data = zone_data["totim"]  # Assuming input time is in days
            time_axis_label = 'Time [days]'
        elif time_units == 'years':
            time_data = zone_data["totim"] / 360  # Convert days to years
            time_axis_label = 'Time [years]'
        else:
            raise ValueError("time_units must be 'days' or 'years'")

        # Calculate reference inflow and outflow at time zero (reference state)
        #reference_inflow = zone_data.loc[zone_data['totim'] == 0, inflow_columns].sum(axis=1).values[0]
        #reference_outflow = zone_data.loc[zone_data['totim'] == 0, outflow_columns].sum(axis=1).values[0]
        reference_inflow = zone_data[inflow_columns].iloc[0].sum()
        reference_outflow = zone_data[outflow_columns].iloc[0].sum()

        # Compute components using vectorized operations
        induced_recharge = zone_data[inflow_columns].sum(axis=1) - reference_inflow
        captured_discharge = reference_outflow - zone_data[outflow_columns].sum(axis=1)
        storage_in = zone_data[storage_in_columns].sum(axis=1)
        storage_out = zone_data[storage_out_columns].sum(axis=1)
        from_storage = storage_in - storage_out
        total_pumped = zone_data[pumped_columns].sum(axis=1)
        capture = induced_recharge + captured_discharge

        # Compute percentages (handle division by zero)
        induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
        captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
        from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
        capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

        # Create plots for the current zone
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Zone {zone} Analysis', fontsize=16)

        # Subplot 1: From Storage, Induced Recharge, Captured Discharge
        axs[0, 0].plot(time_data, from_storage, label='Storage release', color='green')
        axs[0, 0].plot(time_data, induced_recharge, label='Induced Inflows', color='blue')
        axs[0, 0].plot(time_data, captured_discharge, label='Captured Outflows', color='red')
        axs[0, 0].set_title('WATER TO WELLS', fontsize=fontsize)
        axs[0, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[0, 0].set_ylabel('Flow (m³/day)', fontsize=fontsize/1.2)
        axs[0, 0].legend(fontsize=fontsize/1.2)
        axs[0, 0].grid()

        # Subplot 2: From Storage and Capture
        axs[0, 1].plot(time_data, from_storage, label='Storage release', color='green')
        axs[0, 1].plot(time_data, capture, label='Capture', color='purple')
        axs[0, 1].set_title('CAPTURE AND STORAGE', fontsize=fontsize)
        axs[0, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[0, 1].set_ylabel('Flow (m³/day)', fontsize=fontsize/1.2)
        axs[0, 1].legend(fontsize=fontsize/1.2)
        axs[0, 1].grid()

        # Subplot 3: Percentages of Total Pumped (Flows)
        axs[1, 0].plot(time_data, from_storage_pct, label='From Storage (%)', color='purple')
        axs[1, 0].plot(time_data, induced_recharge_pct, label='Induced Recharge (%)', color='blue')
        axs[1, 0].plot(time_data, captured_discharge_pct, label='Captured Discharge (%)', color='green')
        axs[1, 0].set_title('WATER TO WELLS PERCENTAGE', fontsize=fontsize)
        axs[1, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[1, 0].set_ylabel('Percent (%)', fontsize=fontsize/1.2)
        axs[1, 0].legend(fontsize=fontsize/1.2)
        axs[1, 0].grid()

        # Subplot 4: Percentages of Total Pumped (From Storage and Capture)
        axs[1, 1].plot(time_data, from_storage_pct, label='From Storage (%)', color='purple')
        axs[1, 1].plot(time_data, capture_pct, label='Capture (%)', color='orange')
        axs[1, 1].set_title('CAPTURE AND STORAGE PERCENTAGE', fontsize=fontsize)
        axs[1, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[1, 1].set_ylabel('Percent (%)', fontsize=fontsize/1.2)
        axs[1, 1].legend(fontsize=fontsize/1.2)
        axs[1, 1].grid()

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Adjust layout and show plot
        plt.ioff()
        if show:
            plt.tight_layout()
            plt.show()

        #Save plot
        if save:
            image_path = os.path.join(output_dir, f'Zone {zone} water to wells.png')
            fig.savefig(image_path, dpi=300)
            plt.close(fig) 

def plot_storage_change_rate(file_path, 
                            output_path, 
                            show=False, 
                            save=False, 
                            figsize=(14, 12), 
                            fontsize=14,
                            xlim=None,  # Tuple for x-axis limits
                            ylim=None,
                            time_units="days"):  # Tuple for y-axis limits
    """
    Creates a time series plot of the storage change rate.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns for storage components after being processed with process_csv_budget.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        xlim (tuple or None): Limits for x-axis (e.g., (0, 500)). If None, default matplotlib behavior.
        ylim (tuple or None): Limits for y-axis (e.g., (-10, 10)). If None, default matplotlib behavior.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'. Assumes model inputs in days by default.

    Outputs:
        A figure with the storage change rate time series.
    """

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare data
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")
    
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_data, storage_change_rate, label="STORAGE CHANGE RATE", color="green")

    ax.set_title("Storage change rate", fontsize=fontsize)
    ax.set_xlabel(time_axis_label, fontsize=fontsize / 1.2)
    ax.set_ylabel("m³/day", fontsize=fontsize / 1.2)
    ax.legend(fontsize=fontsize / 1.2)
    ax.grid()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_storage_change(file_path, 
                        output_path, 
                        show=False, 
                        save=False, 
                        figsize=(14, 12), 
                        fontsize=14, 
                        xlim=None,
                        ylim=None,
                        time_units="days"):
    """
    Creates a time series plot for change in storage (cumulative).

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns for storage components after being processed with process_csv_budget.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        xlim (tuple or None): x-axis limits (min, max).
        ylim (tuple or None): y-axis limits (min, max).
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'. Assumes model inputs in days by default.

    Outputs:
        A figure with the cumulative change in storage time series.
    """

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare data for time series
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    # Compute the storage change rate (STORAGE OUT - STORAGE IN)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in

    # Compute the cumulative storage change
    storage_change = np.cumsum(storage_change_rate)

    # Plot cumulative storage change
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_data, storage_change, label="CUMULATIVE STORAGE CHANGE", color="green")

    ax.set_title("Cumulative Change in Storage", fontsize=fontsize)
    ax.set_xlabel(time_axis_label, fontsize=fontsize / 1.2)
    ax.set_ylabel("m³", fontsize=fontsize / 1.2)
    ax.legend(fontsize=fontsize / 1.2)
    ax.grid()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def compute_time_steps(stress_period_data):
    """
    Return a list of lists with the time-step lengths for each stress period.

    Parameters
    ----------
    stress_period_data : list of tuples
        [(PERLEN, NSTP, TSMULT), ...] for each stress period.

    Returns
    -------
    time_steps : list of list of floats
        time_steps[sp][ts] = length of that time step
    """
    time_steps = []
    for perlen, nstp, tsmult in stress_period_data:
        if perlen == 0:
            # steady-state or zero-length period: produce zero-length steps
            time_steps.append([0.0] * nstp)
            continue

        if abs(tsmult - 1.0) < 1e-12:
            # uniform steps
            dt = float(perlen) / float(nstp)
            time_steps.append([dt] * nstp)
        else:
            # geometric progression: dt_i = dt0 * tsmult**i, sum_i dt_i = perlen
            r = float(tsmult)
            n = int(nstp)
            dt0 = float(perlen) * (r - 1.0) / (r**n - 1.0)
            steps = [dt0 * (r**i) for i in range(n)]
            time_steps.append(steps)

    return time_steps

def timestep_index_from_totim(stress_period_data, totim, tol=1e-9):
    """
    Map a MODFLOW totim (time at the END of a time step) to cumulative time-step index.

    Parameters
    ----------
    stress_period_data : list of (PERLEN, NSTP, TSMULT)
        As used in the TDIS block.
    totim : float
        Cumulative simulation time (MODFLOW totim) — expected to be the time at the end of some step.
    tol : float
        Relative tolerance for matching totim to an end-of-step time (default 1e-9).

    Returns
    -------
    ts_global : int
        Cumulative time-step index (0-based, counts every step including any zero-length steady steps).
    sp_num : int
        Stress-period index (0-based).
    ts_num : int
        Time-step index within the stress period (0-based).

    Raises
    ------
    ValueError
        If totim does not correspond (within tolerance) to an end-of-step time, or totim is outside simulation time.
    """
    # Handle totim = 0 immediately
    if abs(totim) <= tol:
        return 0, 0, 0
    
    time_steps = compute_time_steps(stress_period_data)

    elapsed = 0.0
    ts_global = 0
    eps = max(1e-12, abs(totim) * tol)  # combined small absolute + relative tolerance

    for sp_num, steps in enumerate(time_steps):
        for ts_num, dt in enumerate(steps):
            elapsed += float(dt)
            # exact/near match
            if abs(elapsed - totim) <= eps:
                return ts_global, sp_num, ts_num
            # if elapsed passed totim (and wasn't close), totim falls inside the step => error
            if elapsed > totim + eps:
                # Compare distance to current elapsed and to previous elapsed
                prev_elapsed = elapsed - float(dt)
                dist_prev = abs(totim - prev_elapsed)
                dist_curr = abs(totim - elapsed)

                if dist_curr < dist_prev:
                    return ts_global, sp_num, ts_num
                else:
                    return ts_global - 1, sp_num, ts_num - 1
            ts_global += 1

    # Finished loop; totim beyond final elapsed time?
    if abs(elapsed - totim) <= eps:
        # return last step
        # compute last indices
        last_sp = len(time_steps) - 1
        last_ts = len(time_steps[-1]) - 1
        return ts_global - 1, last_sp, last_ts

    raise ValueError(f"totim {totim} is beyond the end of the simulation (final totim = {elapsed}).")

def plot_transient_heads(
    gwf,
    idomain: np.ndarray,
    time: float,
    perioddata,
    nrow: int,
    transient_heads: np.ndarray,
    title: str = "Transient head cross section",
    label: str = "Head (m)",
    vmin: float = None,
    vmax: float = None,
    save: bool = True,
    output_folder: str = None,
    plot_name: str = "transient_heads.png",
    boundary_keywords=None,
    ve: float = 10,
    interfaces: np.ndarray = None):
    """
    Plots a cross section of transient heads from a MODFLOW simulation.

    Parameters
    ----------
    gwf : flopy GroundwaterFlowModel
        The FloPy groundwater model object.
    idomain : np.ndarray
        The model's active/inactive cell array.
    time : float
        Simulation time (in model units) to plot the transient heads.
    perioddata : list
        MODFLOW period data used to find the correct timestep.
    nrow : int
        Row index to plot the cross section.
    transient_heads : np.ndarray
        Transient head array (typically from hobj.get_alldata()).
    title : str
        Plot title.
    label : str
        Colorbar label.
    vmin : float or None
        Minimum value for colormap.
    vmax : float or None
        Maximum value for colormap.
    save : bool
        Whether to save the figure.
    output_folder : str or None
        Path to save the figure. Required if save=True.
    plot_name : str
        Name of the plot file (e.g., "transient_heads.png").
    boundary_keywords : list or None
        List of boundary condition keywords to plot (e.g., ["RIV", "WEL", "GHB", "DRN", "CHD"]).
    ve : float
        Vertical exaggeration for the cross-section plot.
    interfaces : np.ndarray or None
        3D array of interface elevations (nlay, nrow, ncol) to plot on top of heads.

    Returns
    -------
    array : np.ndarray
        2D head array (cross-section) plotted.
    """
    
    # --- Determine timestep index ---
    step = timestep_index_from_totim(perioddata, time)[0]

    # --- Extract transient head array at given time ---
    array = transient_heads[step]
    array = np.where(idomain == 0, np.nan, array)
    array = np.where(array == 1E30, np.nan, array)
    array = np.where(array == -1E30, np.nan, array)

    # --- Set up figure ---
    fig = plt.figure(figsize=(19, 5))
    ax = fig.add_subplot(1, 1, 1)
    mx = flopy.plot.PlotCrossSection(ax=ax, model=gwf, line={"row": nrow})

    # --- Plot array ---
    pa = mx.plot_array(array, head=None, vmin=vmin, vmax=vmax)

    # --- Plot grid ---
    mx.plot_grid(color="0.5", alpha=0.2)

    # --- Default color mapping for boundaries ---
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    # --- Dynamically plot boundaries if specified ---
    if boundary_keywords:
        for bc in boundary_keywords:
            bc_color = None
            for key in color_map:
                if key in bc:
                    bc_color = color_map[key]
                    break
            if bc_color:
                mx.plot_bc(bc, color=bc_color)

    if interfaces is not None:
        try:
            ncol = gwf.modelgrid.ncol
            dcol = gwf.modelgrid.delr if np.isscalar(gwf.modelgrid.delr) else np.mean(gwf.modelgrid.delr)
            x = np.arange(ncol) * dcol

            # Plot each interface
            for k in range(interfaces.shape[0]):
                ax.plot(x, interfaces[k, nrow, :], "k-", lw=1.0)

        except Exception as e:
            print(f"Could not plot interfaces: {e}")

    # --- Colorbar and labels ---
    cb = plt.colorbar(pa, ax=ax)
    cb.set_label(label)
    ax.set_title(title)
    ax.set_aspect(ve)

    plt.tight_layout()

    # --- Save or show ---
    if save:
        if output_folder is None:
            raise ValueError("output_folder must be provided if save=True")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(f"{output_folder}/{plot_name}", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return array

def plot_transient_heads_tr(
    gwf,
    idomain: np.ndarray,
    time: float,
    perioddata,
    nrow: int,
    transient_heads: np.ndarray,
    times_list: list,
    cell: tuple,
    start_time: float = None,
    end_time: float = None,
    title: str = "Transient head cross section and time series",
    label: str = "Head (m)",
    vmin: float = None,
    vmax: float = None,
    save: bool = True,
    output_folder: str = None,
    plot_name: str = "transient_heads_tr.png",
    boundary_keywords=None,
    ve: float = 10,
    interfaces: np.ndarray = None, 
    surface: bool = False):
    """
    Plots a cross section of transient heads at a given time (top subplot),
    and a time series of head evolution at a given cell (bottom subplot).

    Parameters
    ----------
    gwf : flopy GroundwaterFlowModel
        The FloPy groundwater model object.
    idomain : np.ndarray
        The model's active/inactive cell array.
    time : float
        Simulation time (in model units) to plot the transient heads cross section.
    perioddata : list
        MODFLOW period data used to find the correct timestep.
    nrow : int
        Row index to plot the cross section.
    transient_heads : np.ndarray
        Transient head array (from hobj.get_alldata()).
        Shape: (n_times, nlay, nrow, ncol)
    times_list : list
        List of simulation times corresponding to each time step.
    cell : tuple
        (lay, row, col) tuple indicating the cell for time series extraction.
    start_time : float or None
        Start time (in model units) for the time series plot. If None, includes from beginning.
    end_time : float or None
        End time (in model units) for the time series plot. If None, includes until the end.
    title : str
        Overall plot title.
    label : str
        Colorbar label.
    vmin, vmax : float or None
        Limits for the color scale.
    save : bool
        Whether to save the figure.
    output_folder : str or None
        Path to save the figure. Required if save=True.
    plot_name : str
        Name of the plot file (e.g., "transient_heads_tr.png").
    boundary_keywords : list or None
        Boundary condition types to plot (e.g., ["RIV", "WEL", "GHB"]).
    ve : float
        Vertical exaggeration for the cross-section.
    interfaces : np.ndarray or None
        3D array of interface elevations (nlay, nrow, ncol) to plot on top of heads.
    surface : bool
        Whether to plot layer surfaces with gradient colors.

    Returns
    -------
    dict
        {
            "cross_section_array": 2D np.ndarray of plotted heads,
            "times": list of times (possibly subset),
            "head_series": np.ndarray of extracted head time series (possibly subset)
        }
    """
    plt.rcParams.update({
        "font.size": 14,           # Base font size
        "axes.titlesize": 16,      # Title font size
        "axes.labelsize": 14,      # Axis label font size
        "xtick.labelsize": 12,     # X tick font
        "ytick.labelsize": 12,     # Y tick font
        "legend.fontsize": 12,     # Legend font
        "figure.titlesize": 18     # Overall figure title font
    })

    # --- Determine timestep index for the requested time ---
    step = timestep_index_from_totim(perioddata, time)[0]

    # --- Extract transient head array at that time ---
    array = transient_heads[step]
    array = np.where(array == 1E30, np.nan, array)
    array = np.where(array == -1E30, np.nan, array)
    array = np.where(idomain == 0, np.nan, array)

    # --- Extract time series for the given cell ---
    lay, row, col = cell
    full_head_series = transient_heads[:, lay, row, col]

    # --- Handle start/end times for subsetting ---
    if start_time is not None:
        start_idx = timestep_index_from_totim(perioddata, start_time)[0]
    else:
        start_idx = 0

    if end_time is not None:
        end_idx = timestep_index_from_totim(perioddata, end_time)[0] + 1  # include end time
    else:
        end_idx = len(times_list)

    # --- Slice the arrays ---
    head_series = full_head_series[start_idx:end_idx]

    # --- Cover times to years from start
    times_list = (np.array(times_list) - start_time) / 360
    times_subset = times_list[start_idx:end_idx]

    # --- Set up figure with two panels ---
    fig, axes = plt.subplots(
        2, 1, figsize=(19, 10), height_ratios=[1, 1], sharex=False
    )
    ax1, ax2 = axes

    # ==============================
    # TOP: Cross-section
    # ==============================
    mx = flopy.plot.PlotCrossSection(ax=ax1, model=gwf, line={"row": nrow})
    pa = mx.plot_array(array, head=None, vmin=vmin, vmax=vmax)
    mx.plot_grid(color="0.5", alpha=0.2)

    # --- Default color mapping for boundaries ---
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    if boundary_keywords:
        for bc in boundary_keywords:
            bc_color = next((color_map[key] for key in color_map if key in bc), None)
            if bc_color:
                mx.plot_bc(bc, color=bc_color)

    # Plot surface for each layer with a gradient of blues
    if surface:
        cmap = get_cmap("Blues")
        num_layers = array.shape[0]
        layer_colors = []  # Store colors for legend
        for layer in range(num_layers):
            # Assign a color based on the layer index
            color = cmap((layer + 1) / num_layers)  # Normalize the layer index
            mx.plot_surface(array[layer, :, :], color=color, lw=1)
            layer_colors.append((color, f"Layer {layer + 1}"))

    if interfaces is not None:
        try:
            ncol = gwf.modelgrid.ncol
            dcol = gwf.modelgrid.delr if np.isscalar(gwf.modelgrid.delr) else np.mean(gwf.modelgrid.delr)
            x = np.arange(ncol) * dcol

            # Plot each interface
            for k in range(interfaces.shape[0]):
                ax1.plot(x, interfaces[k, nrow, :], "k-", lw=1.0)

        except Exception as e:
            print(f"Could not plot interfaces: {e}")

    cb = fig.colorbar(pa, ax=ax1, location="right", fraction=0.03, pad=0.02)
    cb.set_label(label)

    ax1.set_title(f"{title}")
    ax1.set_aspect(ve, adjustable="box", anchor="C")

    # ==============================
    # BOTTOM: Time series
    # ==============================
    ax2.plot(times_subset, head_series, color="black", lw=1.5)
    ax2.axvline((time - start_time)/360, color="red", linestyle="--", lw=1.5)
    ax2.set_ylabel(label)
    ax2.set_xlabel("Time [years]")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_title(f"Hydraulic head at cell (L{lay}, R{row}, C{col})")

    plt.tight_layout()

    # --- Save or show ---
    if save:
        if output_folder is None:
            raise ValueError("output_folder must be provided if save=True")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(f"{output_folder}/{plot_name}", dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return array

def plot_transient_heads_capture(
    gwf,
    idomain: np.ndarray,
    time: float,
    perioddata,
    nrow: int,
    transient_heads: np.ndarray,
    csv_path: str,
    start_time: float,
    end_time: float,
    title: str = "Transient head cross section and capture/storage time series",
    label: str = "Head (m)",
    vmin: float = None,
    vmax: float = None,
    save: bool = True,
    output_folder: str = None,
    plot_name: str = "transient_heads_capture.png",
    boundary_keywords=None,
    ve: float = 10,
    interfaces: np.ndarray = None
):
    """
    Plots:
      (1) A cross-section of transient heads at a given time (top subplot).
      (2) Time series of Storage Release (%) and Capture (%) from CSV (bottom subplot).

    Parameters
    ----------
    gwf : flopy GroundwaterFlowModel
        FloPy groundwater model object.
    idomain : np.ndarray
        Active/inactive cell array.
    time : float
        Simulation time (in model units) for the cross-section plot.
    perioddata : list
        MODFLOW period data for timestep lookup.
    nrow : int
        Row index for cross-section.
    transient_heads : np.ndarray
        Transient head array (shape: n_times, nlay, nrow, ncol).
    times_list : list
        Simulation times corresponding to each time step.
    csv_path : str
        Path to CSV with columns: time (days), Storage_Release_Pct, Capture_Pct.
    start_time : float
        Start time (in days) for plotting capture/storage (typically pumping start).
    end_time : float
        End time (in days) for plotting capture/storage.
    title : str
        Overall plot title.
    label : str
        Colorbar label for head.
    vmin, vmax : float
        Color scale limits for head.
    save : bool
        Whether to save the figure.
    output_folder : str or None
        Directory for saving the figure (required if save=True).
    plot_name : str
        Output filename.
    boundary_keywords : list or None
        List of BC packages to plot (e.g., ["RIV", "GHB", "DRN"]).
    ve : float
        Vertical exaggeration for cross-section.
    interfaces : np.ndarray or None
        3D array of interface elevations (nlay, nrow, ncol) to plot on top of heads.
    Returns
    -------
    dict
        {
            "cross_section_array": 2D np.ndarray,
            "capture_data": pd.DataFrame (subset to time range)
        }
    """

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })

    # --- Determine timestep index for requested time ---
    step = timestep_index_from_totim(perioddata, time)[0]

    # --- Extract transient head array for that time ---
    array = transient_heads[step]
    array = np.where(idomain == 0, np.nan, array)

    # --- Set up figure ---
    fig, axes = plt.subplots(2, 1, figsize=(19, 10), height_ratios=[1, 1])
    ax1, ax2 = axes

    # ==============================
    # TOP: Cross-section
    # ==============================
    mx = flopy.plot.PlotCrossSection(ax=ax1, model=gwf, line={"row": nrow})
    pa = mx.plot_array(array, head=None, vmin=vmin, vmax=vmax)
    mx.plot_grid(color="0.5", alpha=0.2)

    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    if boundary_keywords:
        for bc in boundary_keywords:
            bc_color = next((color_map[key] for key in color_map if key in bc.upper()), "k")
            try:
                mx.plot_bc(bc.lower(), color=bc_color)
            except Exception:
                pass

    if interfaces is not None:
        try:
            ncol = gwf.modelgrid.ncol
            dcol = gwf.modelgrid.delr if np.isscalar(gwf.modelgrid.delr) else np.mean(gwf.modelgrid.delr)
            x = np.arange(ncol) * dcol

            # Plot each interface
            for k in range(interfaces.shape[0]):
                ax1.plot(x, interfaces[k, nrow, :], "k-", lw=1.0)

        except Exception as e:
            print(f"Could not plot interfaces: {e}")

    cb = fig.colorbar(pa, ax=ax1, location="right", fraction=0.03, pad=0.02)
    cb.set_label(label)

    ax1.set_title(f"{title}")
    ax1.set_aspect(ve, adjustable="box", anchor="C")

    # ==============================
    # BOTTOM: Capture/Storage time series
    # ==============================
    df = pd.read_csv(csv_path)
    required_cols = {"time", "Storage_Release_Pct", "Capture_Pct"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Filter time range (days)
    mask = (df["time"] >= start_time) & (df["time"] <= end_time)
    df_subset = df.loc[mask].copy()

    # Convert time to years after start of pumping
    df_subset["time_yrs"] = (df_subset["time"] - start_time) / 360.0

    # Plot
    ax2.plot(df_subset["time_yrs"], df_subset["Storage_Release_Pct"],
             label="Storage release (%)", color="darkorange", lw=1.8)
    ax2.plot(df_subset["time_yrs"], df_subset["Capture_Pct"],
             label="Capture (%)", color="steelblue", lw=1.8)

    # Vertical line for current time
    ax2.axvline((time - start_time)/360.0, color="red", linestyle="--", lw=1.5)

    ax2.set_xlabel("Time since pumping start [years]")
    ax2.set_ylabel("Percentage of total pumping rate (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc='lower right')
    ax2.set_title("Storage release and capture evolution")

    plt.tight_layout()

    # --- Save or show ---
    if save:
        if output_folder is None:
            raise ValueError("output_folder must be provided if save=True")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, plot_name), dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return array

def plot_residual_diffusion(
    gwf,
    time: float,
    perioddata,
    nrow: int,
    transient_heads: np.ndarray,
    steady_state_heads: np.ndarray,
    title: str = "Absolute residual diffusion cross section",
    label: str = "Absolute residual diffusion (m)",
    vmin: float = None,
    vmax: float = None,
    save: bool = True,
    show: bool = False,
    output_folder: str = None,
    plot_name : str = "residual_diffusion.png",
    boundary_keywords=None,
    ve: float = 10,
    interfaces: np.ndarray = None):

    """
    Plots the absolute residual diffusion between transient and steady-state heads
    for analyzing transient response after a step change in stress.

    Parameters
    ----------
    gwf : flopy GroundwaterFlowModel
        The FloPy groundwater model object.
    time : float
        Time in model units to plot the residual diffusion. 
    perioddata : list
        MODFLOW period data.
    nrow : int
        Row index to plot the cross section.
    transient_heads : np.ndarray
        Transient head array.
    steady_state_heads : np.ndarray
        Steady-state head array.
    hobj : flopy.utils.HeadFile
        FloPy headfile object.
    title : str
        Plot title.
    label : str
        Colorbar label.
    vmin : float or None
        Minimum value for colormap.
    vmax : float or None
        Maximum value for colormap.
    save : bool
        Whether to save the figure.
    show : bool
        Whether to display the figure.
    output_folder : str or None
        Path to save the figure. Required if save=True.
    plot_name : str
        Name of the plot file (e.g., "residual_diffusion.png").
    boundary_keywords : list or None
        List of boundary condition keywords to plot (e.g., ["RIV", "WEL", "GHB", "DRN", "CHD"]).
    ve : float
        Vertical exaggeration for the cross-section plot.
    interfaces : np.ndarray or None
        3D array of interface elevations (nlay, nrow, ncol) to plot on top of heads.
    """

    # Determine steps
    step = timestep_index_from_totim(perioddata, time)[0]
    
    # Compute residual
    array = np.abs(transient_heads[step] - steady_state_heads)
    
    # Set up figure
    fig = plt.figure(figsize=(19, 5))
    ax = fig.add_subplot(1, 1, 1)
    mx = flopy.plot.PlotCrossSection(ax=ax, model=gwf, line={"row": nrow})
    
    # Plot array
    pa = mx.plot_array(array, alpha=1, masked_values=[1.0e30], cmap="viridis", vmin=vmin, vmax=vmax)
    
    # Plot grid and colorbar
    mx.plot_grid(color="0.5", alpha=0.2)

    # Default color mapping based on boundary condition type
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    # Dynamically plot boundary conditions based on keywords
    if boundary_keywords:
        for bc in boundary_keywords:
            # Determine color based on the keyword
            bc_color = None
            for key in color_map:
                if key in bc:  # Check if the keyword contains the key
                    bc_color = color_map[key]
                    break
            # Plot the boundary condition with the appropriate color
            if bc_color:
                mx.plot_bc(bc, color=bc_color)

    if interfaces is not None:
        try:
            ncol = gwf.modelgrid.ncol
            dcol = gwf.modelgrid.delr if np.isscalar(gwf.modelgrid.delr) else np.mean(gwf.modelgrid.delr)
            x = np.arange(ncol) * dcol

            # Plot each interface
            for k in range(interfaces.shape[0]):
                ax.plot(x, interfaces[k, nrow, :], "k-", lw=1.0)

        except Exception as e:
            print(f"Could not plot interfaces: {e}")

    cb = plt.colorbar(pa, ax=ax)
    cb.set_label(label)
    
    # Title
    ax.set_title(title)
    ax.set_aspect(ve)
    
    # Layout
    plt.tight_layout()
    
    # Save or show
    if save:
        if output_folder is None:
            raise ValueError("output_folder must be provided if save=True")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(f"{output_folder}/{plot_name}", dpi=300)
        plt.close(fig)
    if show:
        plt.show()
    
    return array

def plot_residual_diffusion_tr(
    gwf,
    time: float,
    perioddata,
    nrow: int,
    transient_heads: np.ndarray,
    steady_state_heads: np.ndarray,
    times_list: list,
    cell: tuple,
    start_time: float = None,
    end_time: float = None,
    title: str = "Absolute residual diffusion cross section and time series",
    label: str = "Absolute residual diffusion (m)",
    vmin: float = None,
    vmax: float = None,
    save: bool = True,
    show: bool = False,
    output_folder: str = None,
    plot_name : str = "residual_diffusion_tr.png",
    boundary_keywords=None,
    ve: float = 10,
    interfaces: np.ndarray = None,
    relative_diffusion: bool = False,
    stability_threshold: float = 1.0e-3
    ):
    """
    Plots the absolute residual diffusion between transient and steady-state heads
    both as a cross section (top subplot) and a time series at a given cell (bottom subplot).
    """
    plt.rcParams.update({
        "font.size": 14,           # Base font size
        "axes.titlesize": 16,      # Title font size
        "axes.labelsize": 14,      # Axis label font size
        "xtick.labelsize": 12,     # X tick font
        "ytick.labelsize": 12,     # Y tick font
        "legend.fontsize": 12,     # Legend font
        "figure.titlesize": 18     # Overall figure title font
    })

    # --- Determine timestep ---
    step = timestep_index_from_totim(perioddata, time)[0]

    # --- Compute residual diffusion ---
    if relative_diffusion:
        denominator = np.abs(transient_heads[0] - steady_state_heads)
        denominator = np.where(denominator > stability_threshold, denominator, np.nan)
        array = 100 * np.abs(transient_heads[step] - steady_state_heads) / denominator
    else:
        array = np.abs(transient_heads[step] - steady_state_heads)

    # --- Time series extraction ---
    lay, row, col = cell

    if relative_diffusion:
        denominator = np.abs(transient_heads[0, lay, row, col] - steady_state_heads[lay, row, col])
        denominator = denominator if denominator > stability_threshold else np.nan
        full_series =100 * np.abs(transient_heads[:, lay, row, col] - steady_state_heads[lay, row, col]) / denominator
    else:
        full_series = np.abs(transient_heads[:, lay, row, col] - steady_state_heads[lay, row, col])

    # --- Handle start/end times ---
    if start_time is not None:
        start_idx = timestep_index_from_totim(perioddata, start_time)[0]
    else:
        start_idx = 0
    if end_time is not None:
        end_idx = timestep_index_from_totim(perioddata, end_time)[0] + 1
    else:
        end_idx = len(times_list)

    times_subset = np.array(times_list[start_idx:end_idx])
    series = full_series[start_idx:end_idx]

    # --- Convert to years for x-axis ---
    times_years = (times_subset - (start_time or 0)) / 360

    # --- Set up figure with 2 subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(19, 10), height_ratios=[1, 1])
    ax1, ax2 = axes

    # ==========================================================
    # TOP: Cross section plot
    # ==========================================================
    mx = flopy.plot.PlotCrossSection(ax=ax1, model=gwf, line={"row": nrow})
    pa = mx.plot_array(array, alpha=1, masked_values=[1.0e30],
                       cmap="viridis", vmin=vmin, vmax=vmax)
    mx.plot_grid(color="0.5", alpha=0.2)

    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }
    if boundary_keywords:
        for bc in boundary_keywords:
            bc_color = None
            for key in color_map:
                if key in bc:
                    bc_color = color_map[key]
                    break
            if bc_color:
                mx.plot_bc(bc, color=bc_color)

    if interfaces is not None:
        try:
            ncol = gwf.modelgrid.ncol
            dcol = gwf.modelgrid.delr if np.isscalar(gwf.modelgrid.delr) else np.mean(gwf.modelgrid.delr)
            x = np.arange(ncol) * dcol

            # Plot each interface
            for k in range(interfaces.shape[0]):
                ax1.plot(x, interfaces[k, nrow, :], "k-", lw=1.0)

        except Exception as e:
            print(f"Could not plot interfaces: {e}")

    cb = fig.colorbar(pa, ax=ax1, location="right", fraction=0.03, pad=0.02)
    cb.set_label(label)

    ax1.set_title(title)
    ax1.set_aspect(ve, adjustable="box", anchor="C")

    # ==========================================================
    # BOTTOM: Time series of residual diffusion
    # ==========================================================
    ax2.plot(times_years, series, color="black", lw=1.5)
    ax2.axvline((time - (start_time or 0)) / 360, color="red", linestyle="--", lw=1.5)
    ax2.set_xlabel("Time [years]")
    ax2.set_ylabel(label)
    ax2.set_title(f"Residual diffusion at cell (L{lay}, R{row}, C{col})")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    # --- Save or show ---
    if save:
        if output_folder is None:
            raise ValueError("output_folder must be provided if save=True")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(f"{output_folder}/{plot_name}", dpi=300)
        plt.close(fig)
    if show:
        plt.show()

    return array

def tr_storage_change_rate(csvbudfile, csv_output_folder, fig_output_folder,
                           show=False, save_csv=True, save_fig=True,
                           figsize=(14, 8), fontsize=14,
                           xlim=None, ylim=None, threshold=None, threshold_type="absolute",
                           start_time=0.0, step_size=360, csv_name="total_storage_change_rate.csv",
                           fig_name="total_storage_change_rate.png"):
    """
    Plot the total (summed across all zones) storage change rate 
    from a zone budget file.

    Parameters
    ----------
    csvbudfile : str
        Path to the budget CSV file. Must contain columns: 'time', 'STO-TOTAL'.
    csv_output_folder : str
        Folder to save processed CSV file.
    fig_output_folder : str
        Folder to save figure.
    show : bool, default=False
        If True, display the plot.
    save_csv : bool, default=True
        If True, save the processed CSV file.
    save_fig : bool, default=True
        If True, save the plot figure.
    figsize : tuple, default=(14, 8)
        Figure size for the plot.
    fontsize : int, default=14
        Font size for labels and legend.
    xlim : tuple or None, default=None
        Limits for x-axis (years).
    ylim : tuple or None, default=None
        Limits for y-axis.
    threshold : float or None, default=None
        If provided, marks the first time abs(total) < threshold (after start_time).
        If threshold_type='relative', interpreted as percentage (e.g. 1 = 1% of max STO-TOTAL).
    threshold_type : {'absolute', 'relative'}, default='absolute'
        How to interpret the threshold value.
    start_time : float, default=0.0
        Starting time (in days). Plot begins here, with x-axis reset so this = 0.
    step_size : float, default=360
        Time step size in days of the first time step after start_time in the zone budget file.
    """

    # Read file
    df = pd.read_csv(csvbudfile)

    # Check required columns
    required_cols = {'time', 'STO-TOTAL'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")
    
    # Subset from start_time onward
    df = df[df['time'] >= (start_time + step_size)].copy()

    # Shift so that start_time becomes 0 (in days) and convert to years
    df['time_since_start'] = (df['time'] - start_time) / 360.0

    # Aggregate by time (sum across all zones)
    df_total = df.groupby('time_since_start')['STO-TOTAL'].sum()

    # Determine effective threshold
    if threshold is not None:
        if threshold_type not in ["absolute", "relative"]:
            raise ValueError("threshold_type must be 'absolute' or 'relative'")
        
        if threshold_type == "relative":
            max_val = df_total.abs().max()
            effective_threshold = (threshold / 100.0) * max_val
        else:
            effective_threshold = threshold
    else:
        effective_threshold = None

    # Plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    line, = ax.plot(df_total.index, df_total.values, color="blue", label=None)

    legend_label = "Total"

    # Threshold crossing
    t_cross = None # Initialize 
    if effective_threshold is not None:
        crossing = df_total[df_total.abs() <= effective_threshold]
        if not crossing.empty:
            t_cross = crossing.index[0]
            v_cross = crossing.iloc[0]

            # Marker + vertical line
            ax.plot(t_cross, v_cross, 'o', color=line.get_color(), markersize=8, label=None)
            ax.axvline(t_cross, linestyle="--", color=line.get_color(), alpha=0.6, label=None)

            legend_label = f"Total, tr = {t_cross:.0f} years"
        else:
            legend_label = "Total, tr = none"

    ax.set_xlabel("Time since step change (years)", fontsize=fontsize)
    ax.set_ylabel("Storage Change Rate", fontsize=fontsize)
    ax.set_title("Storage Change Rate full system", fontsize=fontsize+2)
    ax.grid(True)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        if t_cross is not None and np.isfinite(t_cross):
            xlim_right = 2 * t_cross
        else:
            xlim_right = df_total.index.max()
        ax.set_xlim(0, xlim_right) 
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend([line], [legend_label], fontsize=fontsize-2)

    # Save outputs
    if save_csv:
        os.makedirs(csv_output_folder, exist_ok=True)
        csv_path = os.path.join(csv_output_folder, csv_name)
        df_total.to_csv(csv_path, header=["STO-TOTAL"])
    
    if save_fig:
        os.makedirs(fig_output_folder, exist_ok=True)
        fig_path = os.path.join(fig_output_folder, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return t_cross

def tr_storage_change_rate_zones(zonebudfile, csv_output_folder, fig_output_folder, 
                                 show=False, save_csv=True, save_fig=True, 
                                 figsize=(14, 12), fontsize=14,
                                 xlim=None, ylim=None, threshold=None, threshold_type="absolute",
                                 start_time=0.0, step_size=0.0, csv_name="storage_change_rate_per_zone.csv",
                                 fig_name="storage_change_rate_per_zone.png", summary_csv_name="tr_zones_storage.csv"):
    """
    Plot storage change rate for each zone from a zone budget file.

    Parameters
    ----------
    zonebudfile : str
        Path to the zone budget CSV file. Must contain columns: 'zone', 'totim', 'STO-TOTAL'.
    csv_output_folder : str
        Folder to save processed CSV file.
    fig_output_folder : str
        Folder to save figure.
    show : bool, default=False
        If True, display the plot.
    save_csv : bool, default=True
        If True, save the processed CSV file.
    save_fig : bool, default=True
        If True, save the plot figure.
    figsize : tuple, default=(14, 12)
        Figure size for the plot.
    fontsize : int, default=14
        Font size for labels and legend.
    xlim : tuple or None, default=None
        Limits for x-axis (years).
    ylim : tuple or None, default=None
        Limits for y-axis.
    threshold : float or None, default=None
        If provided, marks the first time abs(STO-TOTAL) < threshold (after start_time).
        If threshold_type='relative', interpreted as percentage (e.g. 1 = 1% of max STO-TOTAL for that zone).
    threshold_type : {'absolute', 'relative'}, default='absolute'
        How to interpret the threshold value.
    start_time : float, default=0.0
        Starting time (in days). Plot begins here, with x-axis reset so this = 0.
    step_size : float, default=0.0
        Time step size in days for defining the starting time step.
        The stress is applied as input at the start of a given time step, but MODFLOW6 poutputs
        refer to the end of the time step.
    csv_name : str, default="storage_change_rate_per_zone.csv"
        Name of the CSV file with storage change rates per zone.
    fig_name : str, default="storage_change_rate_per_zone.png"
        Name of the figure file.
    summary_csv_name : str, default="tr_zones_storage.csv"
        Name of the summary CSV file with response times per zone.
    """
    
    # Read file
    df = pd.read_csv(zonebudfile)
    
    # Check required columns
    required_cols = {'zone', 'totim', 'STO-TOTAL'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")
    
    # Subset from start_time onward (one step after since zonebud reports results at the end of the time step)
    df = df[df['totim'] >= (start_time + step_size)].copy()

    # Shift so that start_time becomes 0 (in days) and convert to years
    df['time_since_start'] = (df['totim'] - start_time) / 360.0

    # Aggregate by time (sum across all zones)
    df_total = df.groupby('time_since_start')['STO-TOTAL'].sum()
    
    # Pivot data: rows = shifted time, columns = zone, values = STO-TOTAL
    df_pivot = df.pivot_table(index='time_since_start', columns='zone', values='STO-TOTAL')
    
    # Plot
    plt.figure(figsize=figsize)
    ax = plt.gca()
    legend_labels = []
    legend_handles = []

    # --- Store t_cross results here ---
    tr_results = []
    
    # Loop through each zone
    for zone in df_pivot.columns:
        y = df_pivot[zone].dropna()
        x = y.index
        
        # Determine effective threshold for this zone
        if threshold is not None:
            if threshold_type not in ["absolute", "relative"]:
                raise ValueError("threshold_type must be 'absolute' or 'relative'")
            
            if threshold_type == "relative":
                max_val = df_total.abs().max() # global max
                effective_threshold = (threshold / 100.0) * max_val
            else:
                effective_threshold = threshold
        else:
            effective_threshold = None
        
        # Plot main line
        line, = ax.plot(x, y, label=None)
        tr_label = f"Zone {zone}"
        t_cross = None  # initialize

        # Apply threshold detection AFTER the time of the maximum value
        if effective_threshold is not None:
            # Find time of max absolute storage change rate in this zone
            t_max = y.abs().idxmax()
            # Subset data after the time of maximum
            y_after_max = y[y.index > t_max]
            
            # Find first crossing below threshold after the max
            crossing = y_after_max[y_after_max.abs() <= effective_threshold]
            
            if not crossing.empty:
                t_cross = crossing.index[0]
                v_cross = crossing.iloc[0]
                
                # Add marker + vertical line
                ax.plot(t_cross, v_cross, 'o', color=line.get_color(), markersize=8, label=None)
                ax.axvline(t_cross, linestyle="--", color=line.get_color(), alpha=0.6, label=None)
                
                tr_label = f"Zone {zone}, tr = {t_cross:.0f} years"
            else:
                tr_label = f"Zone {zone}, tr = none"

        legend_handles.append(line)
        legend_labels.append(tr_label)

        # --- Save t_cross info ---
        tr_results.append({"zone": zone, "tr_years": t_cross})
    
    df_tr = pd.DataFrame(tr_results)

    ax.set_xlabel("Time since step change (years)", fontsize=fontsize)
    ax.set_ylabel("Storage Change Rate", fontsize=fontsize)
    ax.set_title("Storage Change Rate per Zone", fontsize=fontsize+2)
    ax.grid(True)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    else: 
        max_t_cross = df_tr["tr_years"].max()
        if max_t_cross is not None and np.isfinite(max_t_cross):
            xlim_right = 2 * max_t_cross
        else:
            xlim_right = df_total.index.max()
        ax.set_xlim(0, xlim_right)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.legend(legend_handles, legend_labels, fontsize=fontsize-2)
    
    # Save outputs
    if save_csv:
        os.makedirs(csv_output_folder, exist_ok=True)
        csv_path = os.path.join(csv_output_folder, csv_name)
        df_pivot.to_csv(csv_path)

        tr_csv_path = os.path.join(csv_output_folder, summary_csv_name)
        df_tr.to_csv(tr_csv_path, index=False)
    
    if save_fig:
        os.makedirs(fig_output_folder, exist_ok=True)
        fig_path = os.path.join(fig_output_folder, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()

def absolute_head_diffusion_zones(transient_heads, 
                                  steady_state_heads, 
                                  times, 
                                  zone_array,
                                  start_step=0, 
                                  threshold=1, 
                                  threshold_type= "relative",
                                  stability_threshold=0.01,
                                  csv_output_folder=".", 
                                  summary_csv_name="tr_zones_absolute_diffusion.csv",
                                  save_fig=True, show_fig=False,
                                  fig_output_folder=".",
                                  fig_name = "diff_absolute_zones.png",
                                  zone_descriptions=None,
                                  center="mean",
                                  bounds="95p"):
														  
    """
    Plots per-zone head differences (transient - steady state) with either mean or median
    and either min-max, 95% interval or mean ± std as shaded bounds. Annotates the response times.

    Parameters:
    -----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads.
        Assumes input in days.
    zone_array : np.ndarray
        3D array identifying zones (n_layer, n_row, n_col)
    start_step : int
        Time step index to start analysis (default=0)
    threshold : float
        Value below which cells are considered steady (absolute or relative as percentage)
    threshold_type : str
        "absolute" or "relative" (default="relative")
    stability_threshold : float
        Threshold to exclude "irresponsive" cells to the step change.
    csv_output_folder : str
        Folder path to save the csv summary output.
    summary_csv_name : str
        Name of the summary CSV file with response times per zone.
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
    fig_output_folder : str
        Folder path to save figures
    fig_name : str
        Name of the figure file
    zone_descriptions : dict, optional
        Dictionary mapping zone numbers to descriptive names
    center : str, default "mean"
        Center line method: "mean" or "median"
    bounds : str, default "95p"
        Method for shaded bounds (for visualization purposes only): "95p" = 2.5th–97.5th percentiles,
									 
        "stdev" = mean ± standard deviation
        "full" = min–max range
																														  
    Output:
    Saves a csv summary of response times statistics per zone.
    Saves a figure with per-zone plots of absolute head differences over time. (if save_fig=True)
    """

    # Ensure output folders exist
    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute initial absolute residual diffusion
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)

    # Mask for irresponsive cells
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan
    initial_diff_max = np.nanmax(initial_diff)
    if np.all(np.isnan(initial_diff)):
        print("No residual diffusion higher than the stability threshold — returning NaN")
        return np.nan

    # Compute absolute residual diffusion array								   
    diff_array = np.abs(transient_heads - steady_state_heads)  # (ntsp, nlay, nrow, ncol)														 
    diff_array[:, zero_diff_mask] = np.nan
																				
    # Unique zones (exclude background if needed)
    zones = np.unique(zone_array)
    zones = zones[zones > 0]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]  # zero at start_step
    time_in_years = time_since_start / 360  # convert to years

    # Prepare subplots
    n_zones = len(zones)
    fig, axes = plt.subplots(n_zones, 1, figsize=(14, 4 * n_zones), sharex=True)
    if n_zones == 1:
        axes = [axes]
    
    # Store response time results
    tr_records = []

    for ax, zone in zip(axes, zones):
        # Mask: zone selection
        zone_mask = (zone_array == zone)

        # Exclude "irresponsive" cells

        combined_mask = np.logical_or(~zone_mask, zero_diff_mask)

        # Apply combined mask
        diff_zone = np.where(combined_mask, np.nan, diff_array)

        # Flatten spatial dimensions
        diff_zone_flat = diff_zone.reshape(diff_zone.shape[0], -1) # 2D array (n_time, n_cells)

        # Select only times after start_step
        selected_diff_zone = diff_zone_flat[start_step:]

        # Compute stats per timestep
        means = np.nanmean(selected_diff_zone, axis=1)
        medians = np.nanmedian(selected_diff_zone, axis=1)
        std = np.nanstd(selected_diff_zone, axis=1)
        maxs = np.nanmax(selected_diff_zone, axis=1)
        mins = np.nanmin(selected_diff_zone, axis=1)
        p_95 = np.nanpercentile(selected_diff_zone, 95, axis=1)
        p_2_5 = np.nanpercentile(selected_diff_zone, 2.5, axis=1)
        p_97_5 = np.nanpercentile(selected_diff_zone, 97.5, axis=1)

        if center == "median":
            centers = medians
        elif center == "mean":
            centers = means

        if bounds == "95p":
            lower = p_2_5
            upper = p_97_5
        elif bounds == "stdev":
            lower = np.maximum(centers - std, 0)
            upper = np.minimum(centers + std, maxs)
        elif bounds == "full":
            lower = mins
            upper = maxs
        else:
            raise ValueError("bounds must be '95p' or 'stdev'")

        # Determine threshold
        if threshold_type == "relative":
            threshold_zone = initial_diff_max * (threshold / 100)
        elif threshold_type == "absolute":
            threshold_zone = threshold
        else:
            raise ValueError("threshold_type must be 'absolute' or 'relative'")

        # Plot lines
        ax.plot(time_in_years, centers, color="blue", linestyle="--", label="Mean", linewidth=1.5)
        ax.fill_between(time_in_years, lower, upper, color="lightblue", alpha=0.3,
                        label="95% interval" if bounds=="95p" else "Standard Deviation")
        # Add borders
        ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Mean response time
        t_cross_mean = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(means <= threshold_zone)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_mean = time_in_years[idx]
            mean_value = means[idx]
            if center == "mean":
                ax.scatter(t_cross_mean, mean_value, color="green", s=50, zorder=5)
                ax.axvline(t_cross_mean, color="green", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_mean, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_mean))} yr",
                        color="green", rotation=0, va='top', fontweight='bold')
        
        # Median response time
        t_cross_median = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(medians <= threshold_zone)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_median = time_in_years[idx]
            median_value = medians[idx]
            if center == "median":
                ax.scatter(t_cross_median, median_value, color="green", s=50, zorder=5)
                ax.axvline(t_cross_median, color="green", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_median, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_median))} yr",
                        color="green", rotation=0, va='top', fontweight='bold')
            
        # 95pth response time
        t_cross_p_95 = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(p_95 <= threshold_zone)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_p_95 = time_in_years[idx]
            p_95_value = p_95[idx]
            if bounds == "95p":
                ax.scatter(t_cross_p_95, p_95_value, color="red", s=50, zorder=5)
                ax.axvline(t_cross_p_95, color="red", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_p_95, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_p_95))} yr",
                        color="red", rotation=0, va='top', fontweight='bold')
            
        # Max response time
        t_cross_max = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(maxs <= threshold_zone)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_max = time_in_years[idx]
            max_value = maxs[idx]
            if bounds == "full" or bounds == "stdev":
                ax.scatter(t_cross_max, max_value, color="red", s=50, zorder=5)
                ax.axvline(t_cross_max, color="red", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_max, ax.get_ylim()[1]*0.8, f"tr={int(round(t_cross_max))} yr",
                        color="red", rotation=0, va='top', fontweight='bold')
            
        # Save results
        tr_records.append({
            "zone": zone,
            "tr_mean_years": t_cross_mean,
            "tr_median_years": t_cross_median,
            "tr_95p_years": t_cross_p_95,
            "tr_max_years": t_cross_max
        })
        
        # Labels
        ax.set_ylabel("Absolute residual diffusion (m)")
        if zone_descriptions and zone in zone_descriptions:
            ax.set_title(f"Zone {zone}: {zone_descriptions[zone]}")
        else:
            ax.set_title(f"Zone {zone}")

    # Create DataFrame for response time results and save csv
    df_tr = pd.DataFrame(tr_records)
    tr_csv_path = os.path.join(csv_output_folder, summary_csv_name)
    df_tr.to_csv(tr_csv_path, index=False)

    # Apply same limit to all subplots
    for ax in axes:
        ax.set_ylim(0, np.nanmax(diff_array)) 
        max_tr_max = df_tr["tr_max_years"].max()
        xlim_right = min(2 * max_tr_max, time_in_years[-1])
        ax.set_xlim(0, xlim_right)

    axes[-1].set_xlabel("Time since step change (years)")
    axes[0].legend()
    plt.tight_layout()
    
    # Save/show figure
    if save_fig:
        fig_path = os.path.join(fig_output_folder, fig_name)
        plt.savefig(fig_path, dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

def absolute_head_diffusion(transient_heads, 
                            steady_state_heads, 
                            times,
                            start_step=0, 
                            threshold=1, 
                            threshold_type= "relative", 
                            stability_threshold=0.01,
                            save_array=True, 
                            save_fig=True, 
                            show_fig=False,
                            array_output_folder=".",
                            array_name = "diff_array_absolute.npy",
                            fig_output_folder=".",
                            fig_name="diff_absolute_total.png",
                            center="mean",
                            bounds="95p"):
    """
    Plots absolute residual diffusion timeseries.

    Parameters:
    -----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads
        Assumes input in days.
    start_step : int
        Time step index to start analysis (default=0)
    threshold : float
        Value below which cells are considered steady (absolute or relative as percentage)
    threshold_type : str
        "absolute" or "relative" (default="relative")
    stability_threshold : float
        Threshold to exclude "irresponsive" cells to the step change.
    save_array : bool
        Whether to save the 4D absolute residual diffusion array
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
    array_output_folder : str
        Folder path to save the array
    array_name : str
        Name of the saved array file
    fig_output_folder : str
        Folder path to save figure
    fig_name : str
        Name of the figure file
    center : str, default "mean"
        Center line method: "mean" or "median"
    bounds : str, default "95p"
        Method for shaded bounds: "95p" = 2.5th - 97.5th percentiles,
        "stdev" = mean ± standard deviation
        "full" = min–max range
    
        Returns:
    Statistics of response times
    Saves the absolute residual diffusion array (if save_array=True)
    Saves a figure with absolute residual diffusion time series (if save_fig=True)
    """

    # Ensure output folder exists
    os.makedirs(array_output_folder, exist_ok=True)
    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute initial difference
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)
    # Mask for small (near zero) initial differences
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan
    initial_diff_max = np.nanmax(initial_diff)
    if np.all(np.isnan(initial_diff)):
        print("No residual diffusion higher than the stability threshold — returning NaN")
        return np.nan, np.nan, np.nan

    # Compute absolute differences
    diff_array = np.abs(transient_heads - steady_state_heads)  # (ntsp, nlay, nrow, ncol)
    diff_array[:, zero_diff_mask] = np.nan

    # Save diff_array
    if save_array:
        np.save(os.path.join(array_output_folder, array_name), diff_array)

    # Flatten spatial dimensions
    diff_flat = diff_array.reshape(diff_array.shape[0], -1)
    selected_diff = diff_flat[start_step:]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]
    time_in_years = time_since_start / 360  # convert to years

    # Compute stats per timestep
    means = np.nanmean(selected_diff, axis=1)
    medians = np.nanmedian(selected_diff, axis=1)
    std = np.nanstd(selected_diff, axis=1)
    maxs = np.nanmax(selected_diff, axis=1)
    mins = np.nanmin(selected_diff, axis=1)
    p_95 = np.nanpercentile(selected_diff, 95, axis=1)
    p_2_5 = np.nanpercentile(selected_diff, 2.5, axis=1)
    p_97_5 = np.nanpercentile(selected_diff, 97.5, axis=1)

    if center == "median":
        centers = medians
    elif center == "mean":
        centers = means

    if bounds == "95p":
        lower = p_2_5
        upper = p_97_5
    elif bounds == "stdev":
        lower = np.maximum(centers - std, 0)
        upper = np.minimum(centers + std, maxs)
    elif bounds == "full":
        lower = mins
        upper = maxs
    else:
        raise ValueError("bounds must be '95p' or 'stdev'")

    # Determine threshold
    if threshold_type == "relative":
        threshold = initial_diff_max * (threshold / 100)
    elif threshold_type == "absolute":
        threshold = threshold   
    else:
        raise ValueError("threshold_type must be 'absolute' or 'relative'")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_in_years, centers, color="blue", linestyle="--", label="Mean", linewidth=1.5)
    ax.fill_between(time_in_years, np.where(np.isnan(lower), np.nan, lower),
                    np.where(np.isnan(upper), np.nan, upper),
                    color="lightblue", alpha=0.3,
                    label="95% interval" if bounds=="95p" else "Standard Deviation")
    ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Mean response time
    t_cross_mean = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(means <= threshold)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_mean = time_in_years[idx]
        mean_value = means[idx]
        if center == "mean":
            ax.scatter(t_cross_mean, mean_value, color="green", s=50, zorder=5)
            ax.axvline(t_cross_mean, color="green", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_mean, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_mean))} yr",
                    color="green", rotation=0, va='top', fontweight='bold')

    # Median response time
    t_cross_median = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(medians <= threshold)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_median = time_in_years[idx]
        median_value = medians[idx]
        if center == "median":
            ax.scatter(t_cross_median, median_value, color="green", s=50, zorder=5)
            ax.axvline(t_cross_median, color="green", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_median, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_median))} yr",
                    color="green", rotation=0, va='top', fontweight='bold')
        
    # 95pth response time
    t_cross_p_95 = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(p_95 <= threshold)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_p_95 = time_in_years[idx]
        p_95_value = p_95[idx]
        if bounds == "95p":
            ax.scatter(t_cross_p_95, p_95_value, color="red", s=50, zorder=5)
            ax.axvline(t_cross_p_95, color="red", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_p_95, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_p_95))} yr",
                    color="red", rotation=0, va='top', fontweight='bold')
        
    # Max response time
    t_cross_max = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(maxs <= threshold)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_max = time_in_years[idx]
        max_value = maxs[idx]
        if bounds == "full" or bounds == "stdev":
            ax.scatter(t_cross_max, max_value, color="red", s=50, zorder=5)
            ax.axvline(t_cross_max, color="red", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_max, ax.get_ylim()[1]*0.8, f"tr={int(round(t_cross_max))} yr",
                    color="red", rotation=0, va='top', fontweight='bold')

    # Labels
    ax.set_xlabel("Time since step change (years)")
    ax.set_ylabel("Absolute residual diffusion (m)")
    ax.set_title("Absolute residual diffusion time series")
    ax.set_ylim(bottom=0)
    xlim_right = min(2 * t_cross_max, time_in_years[-1])
    ax.set_xlim(0, xlim_right)
    ax.legend()
    plt.tight_layout()

    # Save figure
    if save_fig:
        fig_path = os.path.join(fig_output_folder, fig_name)
        plt.savefig(fig_path, dpi=300)

    # Show figure
    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return t_cross_mean, t_cross_median, t_cross_p_95, t_cross_max

def relative_head_diffusion_zones(transient_heads, 
                                  steady_state_heads, 
                                  times, 
                                  zone_array,
                                  start_step=0, 
                                  threshold_percent= 5,						 
                                  stability_threshold=0.01,
                                  csv_output_folder=".", 
                                  summary_csv_name="tr_zones_relative_diffusion.csv",
                                  save_fig=True, show_fig=False,
                                  fig_output_folder=".",
                                  fig_name= "diff_relative_zones.png",
                                  zone_descriptions=None,
                                  center="mean",
                                  bounds="95p",
                                  max_initial_diff=False):
    """
    Plots per-zone relative head differences (transient - steady state) with either mean or median
    and either min-max, 95% interval or mean ± std as shaded bounds. Annotates the response times.

    Parameters:
    -----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads
        Assumes input in days.					  
    zone_array : np.ndarray
        3D array identifying zones (n_layer, n_row, n_col)
    start_step : int
        Time step index to start analysis (default=0)
    threshold_percent : float
        Value below which cells are considered															 
    stability_threshold : float
        Threshold to exclude "irresponsive" cells to the step change.
    csv_output_folder : str
        Folder path to save the csv summary output.
    summary_csv_name : str
        Name of the summary CSV file with response times per zone.
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
	fig_output_folder : str
        Folder path to save figures
	fig_name : str
        Name of the figure file
    zone_descriptions : dict, optional
        Dictionary mapping zone numbers to descriptive names
	center : str, default "mean"
        Center line method: "mean" or "median"																	  
    bounds : str, default "95p"
        Method for shaded bounds (for visualization purposes only): 
		"95p" = 2.5th–97.5th percentiles,
        "stdev" = mean ± standard deviation
        "full" = min–max range
    max_initial_diff : bool
        If True, use the maximum initial difference across all cells for normalization.
        If False, use cell-specific initial differences: This corresponds to the head relaxation described by Carr et al 2018.
	
    Output:
    Saves a csv summary of response times statistics per zone.
    Saves a figure with per-zone plots of absolute head differences over time. (if save_fig=True)
    """

    # Ensure output folders exist
    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute initial initial absolute residual diffusion
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)

    # Mask for irresponsive cells
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan
    initial_diff_max = np.nanmax(initial_diff)
    if np.all(np.isnan(initial_diff)):
        print("No residual diffusion higher than the stability threshold — returning NaN")
        return np.nan

    if max_initial_diff:
        # Compute normalized difference
        diff_array = np.abs(transient_heads - steady_state_heads)
        diff_array[:, zero_diff_mask] = np.nan
        relative_diff = diff_array * 100 / initial_diff_max
    else:
        # Compute normalized difference
        diff_array = np.abs(transient_heads - steady_state_heads)
        diff_array[:, zero_diff_mask] = np.nan
        relative_diff = diff_array * 100 / initial_diff  # element-wise division

    # Unique zones (exclude background if needed)
    zones = np.unique(zone_array)
    zones = zones[zones > 0]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]  # zero at start_step
    time_in_years = time_since_start / 360  # convert to years

    # Prepare subplots
    n_zones = len(zones)
    fig, axes = plt.subplots(n_zones, 1, figsize=(14, 4 * n_zones), sharex=True)
    if n_zones == 1:
        axes = [axes]

    # Store response time results
    tr_records = []

    for ax, zone in zip(axes, zones):
        # Mask: zone selection
        zone_mask = (zone_array == zone)

        # Exclude "irresponsive" cell
        combined_mask = np.logical_or(~zone_mask, zero_diff_mask)

        # Apply combined mask
        diff_zone = np.where(combined_mask, np.nan, relative_diff)

        # Flatten spatial dimensions
        diff_zone_flat = diff_zone.reshape(diff_zone.shape[0], -1)

        # Select only times after start_step
        selected_diff_zone = diff_zone_flat[start_step:]

        # Compute stats per timestep
        means = np.nanmean(selected_diff_zone, axis=1)
        medians = np.nanmedian(selected_diff_zone, axis=1)
        std = np.nanstd(selected_diff_zone, axis=1)
        maxs = np.nanmax(selected_diff_zone, axis=1)
        mins = np.nanmin(selected_diff_zone, axis=1)
        p_95 = np.nanpercentile(selected_diff_zone, 95, axis=1)
        p_2_5 = np.nanpercentile(selected_diff_zone, 2.5, axis=1)
        p_97_5 = np.nanpercentile(selected_diff_zone, 97.5, axis=1)

        if center == "median":
            centers = medians
        elif center == "mean":
            centers = means									   

        if bounds == "95p":
            lower = p_2_5
            upper = p_97_5
        elif bounds == "stdev":
            lower = np.maximum(centers - std, 0)
            upper = np.minimum(centers + std, maxs)
        elif bounds == "full":
            lower = mins
            upper = maxs
        else:
            raise ValueError("bounds must be '95p' or 'stdev'")				 																	   

        # Plot lines
        ax.plot(time_in_years, centers, color="blue", linestyle="--", label="Mean", linewidth=1.5)
        ax.fill_between(time_in_years, lower, upper, color="lightblue", alpha=0.3,
                        label="95% interval" if bounds=="95p" else "Standard Deviation") 
        ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Mean response time
        t_cross_mean = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(means <= threshold_percent)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_mean = time_in_years[idx]
            mean_value = means[idx]
            if center == "mean":
                ax.scatter(t_cross_mean, mean_value, color="green", s=50, zorder=5)
                ax.axvline(t_cross_mean, color="green", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_mean, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_mean))} yr",
                        color="green", rotation=0, va='top', fontweight='bold')
        
        # Median response time
        t_cross_median = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(medians <= threshold_percent)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_median = time_in_years[idx]
            median_value = medians[idx]
            if center == "median":
                ax.scatter(t_cross_median, median_value, color="green", s=50, zorder=5)
                ax.axvline(t_cross_median, color="green", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_median, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_median))} yr",
                        color="green", rotation=0, va='top', fontweight='bold')
            
        # 95pth response time
        t_cross_p_95 = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(p_95 <= threshold_percent)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_p_95 = time_in_years[idx]
            p_95_value = p_95[idx]
            if bounds == "95p":
                ax.scatter(t_cross_p_95, p_95_value, color="red", s=50, zorder=5)
                ax.axvline(t_cross_p_95, color="red", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_p_95, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_p_95))} yr",
                        color="red", rotation=0, va='top', fontweight='bold')
            
        # Max response time
        t_cross_max = time_in_years[-1]  # initialize as the max time step
        below_threshold_idx = np.where(maxs <= threshold_percent)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross_max = time_in_years[idx]
            max_value = maxs[idx]
            if bounds == "full" or bounds == "stdev":
                ax.scatter(t_cross_max, max_value, color="red", s=50, zorder=5)
                ax.axvline(t_cross_max, color="red", linestyle=":", linewidth=1.5)
                ax.text(1.01*t_cross_max, ax.get_ylim()[1]*0.8, f"tr={int(round(t_cross_max))} yr",
                        color="red", rotation=0, va='top', fontweight='bold')
        
        # Save results
        tr_records.append({
            "zone": zone,
            "tr_mean_years": t_cross_mean,
            "tr_median_years": t_cross_median,
            "tr_95p_years": t_cross_p_95,
            "tr_max_years": t_cross_max
        })

        # Labels
        ax.set_ylabel("Relative residual diffusion (%)")
        if zone_descriptions and zone in zone_descriptions:
            ax.set_title(f"Zone {zone}: {zone_descriptions[zone]}")
        else:
            ax.set_title(f"Zone {zone}")

    # Create DataFrame for response time results and save csv
    df_tr = pd.DataFrame(tr_records)
    tr_csv_path = os.path.join(csv_output_folder, summary_csv_name)
    df_tr.to_csv(tr_csv_path, index=False)															   

    # Apply same limit to all subplots
    for ax in axes:
        ax.set_ylim(0, 100)
        max_tr_max = df_tr["tr_max_years"].max()
        xlim_right = min(2 * max_tr_max, time_in_years[-1])
        ax.set_xlim(0, xlim_right)

    axes[-1].set_xlabel("Time since step change (years)")
    axes[0].legend()
    plt.tight_layout()
    
    # Save/show figure
    if save_fig:
        fig_path = os.path.join(fig_output_folder, fig_name)
        plt.savefig(fig_path, dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

def relative_head_diffusion(transient_heads, 
                            steady_state_heads, 
                            times,
                            start_step=0, 
                            threshold_percent=5, 					
                            stability_threshold=0.01,
                            save_array=True,
                            save_fig=True, 
                            show_fig=False,
                            array_output_folder=".",
                            array_name="diff_array_relative.npy",
                            fig_output_folder=".",
                            fig_name="diff_relative_total.png",
                            center="mean", 
                            bounds="95p", 
                            max_initial_diff=False,
                            ):
    """
    Plots residual head differences (transient - steady state) normalized by initial difference.
    Small initial differences below threshold_value are excluded to avoid division by zero.
    Can save and/or display the figure.

    Parameters
    ----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads				  
    start_step : int
        Time step index to start analysis (default=0)
    threshold_percent : float
        Threshold to calculate response time (default=5)									 
    stability_threshold : float
        Percent threshold for stability used to mask initial differences near steady state (default=0.01).
    save_array : bool
        Whether to save the 4D relative residual diffusion array
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
    array_output_folder : str
        Folder path to save the array
    array_name : str
        Name of the saved array file
    fig_output_folder : str
        Folder path to save figure
    fig_name : str
        Name of the figure file
    center : str, default "mean"
        Center line method: "mean" or "median"
    bounds : str, default "95p"
        Method for shaded bounds: "95p" = 2.5th–97.5th percentiles,
        "stdev" = mean ± standard deviation
        "full" = min–max range
    max_initial_diff : bool
        If True, use the maximum initial difference across all cells for normalization.
        If False, use cell-specific initial differences: This corresponds to the head relaxation described by Carr et al 2018.
    
    Returns:
    Statistics of response times
    Saves the absolute residual diffusion array (if save_array=True)
    Saves a figure with absolute residual diffusion time series (if save_fig=True)
    """

	# Ensure output folder exists
    os.makedirs(array_output_folder, exist_ok=True)							 								   
    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute initial difference
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)
    # Mask for small (near zero) initial differences
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan
    initial_diff_max = np.nanmax(initial_diff)
    if np.all(np.isnan(initial_diff)):
        print("No residual diffusion higher than the stability threshold — returning NaN")
        return np.nan, np.nan, np.nan

    if max_initial_diff:
        # Compute global normalized difference
        diff_array = np.abs(transient_heads - steady_state_heads)
        diff_array[:, zero_diff_mask] = np.nan
        relative_array = diff_array * 100 / initial_diff_max
    else:
        # Compute local normalized difference
        diff_array = np.abs(transient_heads - steady_state_heads)
        diff_array[:, zero_diff_mask] = np.nan
        relative_array = diff_array * 100 / initial_diff

	# Save diff_array
    if save_array:
        np.save(os.path.join(array_output_folder, array_name), relative_array)				 																  

    # Flatten spatial dimensions
    relative_flat = relative_array.reshape(relative_array.shape[0], -1)
    selected_relative = relative_flat[start_step:]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]
    time_in_years = time_since_start / 360  # convert to years assuming time in days

    # Compute stats per timestep
    means = np.nanmean(selected_relative, axis=1)
    medians = np.nanmedian(selected_relative, axis=1)
    std = np.nanstd(selected_relative, axis=1)
    maxs = np.nanmax(selected_relative, axis=1)
    mins = np.nanmin(selected_relative, axis=1)
    p_95 = np.nanpercentile(selected_relative, 95, axis=1)
    p_2_5 = np.nanpercentile(selected_relative, 2.5, axis=1)
    p_97_5 = np.nanpercentile(selected_relative, 97.5, axis=1)

    if center == "median":
        centers = medians
    elif center == "mean":
        centers = means

    if bounds == "95p":
        lower = p_2_5
        upper = p_97_5
    elif bounds == "stdev":
        lower = np.maximum(centers - std, 0)
        upper = np.minimum(centers + std, maxs)
    elif bounds == "full":
        lower = mins
        upper = maxs
    else:
        raise ValueError("bounds must be '95p' or 'stdev'")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_in_years, centers, color="blue", linestyle="--", label="Mean", linewidth=1.5)
    ax.fill_between(time_in_years, lower, upper, color="lightblue", alpha=0.3,									 
                    label="95% interval" if bounds=="95p" else "Standard Deviation")
    ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Mean response time
    t_cross_mean = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(means <= threshold_percent)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_mean = time_in_years[idx]
        mean_value = means[idx]
        if center == "mean":
            ax.scatter(t_cross_mean, mean_value, color="green", s=50, zorder=5)
            ax.axvline(t_cross_mean, color="green", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_mean, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_mean))} yr",
                    color="green", rotation=0, va='top', fontweight='bold')

    # Median response time
    t_cross_median = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(medians <= threshold_percent)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_median = time_in_years[idx]
        median_value = medians[idx]
        if center == "median":
            ax.scatter(t_cross_median, median_value, color="green", s=50, zorder=5)
            ax.axvline(t_cross_median, color="green", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_median, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_median))} yr",
                    color="green", rotation=0, va='top', fontweight='bold')
        
    # 95pth response time
    t_cross_p_95 = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(p_95 <= threshold_percent)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_p_95 = time_in_years[idx]
        p_95_value = p_95[idx]
        if bounds == "95p":
            ax.scatter(t_cross_p_95, p_95_value, color="red", s=50, zorder=5)
            ax.axvline(t_cross_p_95, color="red", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_p_95, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross_p_95))} yr",
                    color="red", rotation=0, va='top', fontweight='bold')
        
    # Max response time
    t_cross_max = time_in_years[-1]  # initialize as the max time step
    below_threshold_idx = np.where(maxs <= threshold_percent)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross_max = time_in_years[idx]
        max_value = maxs[idx]
        if bounds == "full" or bounds == "stdev":
            ax.scatter(t_cross_max, max_value, color="red", s=50, zorder=5)
            ax.axvline(t_cross_max, color="red", linestyle=":", linewidth=1.5)
            ax.text(1.01*t_cross_max, ax.get_ylim()[1]*0.8, f"tr={int(round(t_cross_max))} yr",
                    color="red", rotation=0, va='top', fontweight='bold')

    # Labels
    ax.set_xlabel("Time since step change (years)")
    ax.set_ylabel("Relative residual diffusion (%)")
    ax.set_title("Relative head difference relative to initial difference")
    ax.set_ylim(0, 100)
    xlim_right = min(2 * t_cross_max, time_in_years[-1])
    ax.set_xlim(0, xlim_right)
    ax.legend()
    ax.set_autoscale_on(False) 
    plt.tight_layout()


	# Save figure			 
    if save_fig:
        fig_path = os.path.join(fig_output_folder, fig_name)
        plt.savefig(fig_path, dpi=300)

	# Show figure			 
    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return t_cross_mean, t_cross_median, t_cross_p_95, t_cross_max

def response_time_array_absolute(
    gwf,
    steady_state_heads,
    transient_heads,
    times_list,
    threshold=1,
    threshold_type="relative",
    stability_threshold=0.01,
    start_step=0,
    save_array=True,
    save_plot=True,
    show_plot=False,
    boundary_keywords=None,
    fill="nan",
    ve=10,
    array_output_folder=None,
    array_name="response_time_absolute.npy",
    fig_output_folder=None,
    fig_name="Response_time_absolute.png",
    histogram = False,
    histogram_bins = None,
    histogram_name ="Response_time_absolute_histogram.png",
    interfaces=None
):
    """
    Compute the response time from absolute residual diffusion.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model object (for plotting purposes)
    steady_state_heads : ndarray
        3D array of steady-state heads (nlay, nrow, ncol)
        representing the final steady state condition after a step change.
    transient_heads : ndarray
        4D array of transient heads (ntime, nlay, nrow, ncol)
        representing the transient simulation results from the initial state to 
        the final steady state.
    times_list : array-like
        Simulation times corresponding to transient simulation.
        Assumes input in days.
    threshold: float, optional
        Threshold to calculate response time (default=1)
    threshold_type : {'absolute', 'relative'}, optional
        Type of threshold: 'absolute' uses the threshold as an absolute value, 
        'relative' uses the input threshold as a percentage (0-100).
    stability_threshold : float, optional
        Threshold to treat differences as NaN, used to prevent division by zero
        and instability caused by numerical noise (default=0.01). This cells will 
        be masked and considered "irresponsive" to the given step change.
    start_step : int, optional
        Step to start computing response time. It should correspond to the time step
        index where the step change occurs.
    save_array : bool, optional
        Whether to save the response time array as .npy file
    save_plot : bool, optional
        Whether to save the plot
    show_plot : bool, optional
        Whether to display the plot    
    boundary_keywords : list of str, optional
        List of boundary condition keywords to plot (e.g. ["RIV", "WEL", "GHB"])
    fill : {'max', 'start', 'nan'}, optional
        How to fill cells with zero initial difference. Used only for plotting 
        visualization purposes. Options:
        - 'max': Fill with the maximum found response time
        - 'start': Fill with the start time
        - 'nan': Fill with NaN. 
        'nan' is recommended.
    ve : float, optional
        Vertical exaggeration for plotting
    array_output_folder : str, optional
        Folder to save the response time array as .npy
    array_name : str, optional
        Name of the response time array file
    fig_output_folder : str, optional
        Folder to save the plot
    fig_name : str, optional
        Name of the plot file
    histogram : bool, optional
        Whether to plot a histogram of response times   
    histogram_bins : int or sequence, optional
        Number of bins or bin edges for the histogram
    histogram_name : str, optional
        Name of the histogram plot file
    interfaces : ndarray, optional
        Array of interface elevations for plotting  

    Returns
    -------
    statistics of response times.
    saves response_time_array : ndarray
        Array of response times in same shape as steady_state_heads (if save_array=True)
    saves plot of response times cross-section (if save_plot=True)
    saves histogram of response times (if histogram=True)
    """

    end_step = transient_heads.shape[0] - 1
    times = np.array(times_list)

    # ------------------------------ Compute absolute response time ----------------------------------- #
    nlay, nrow, ncol = steady_state_heads.shape

    # Precompute initial absolute residual diffusion
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan
    if np.all(np.isnan(initial_diff)):
        print("No residual diffusion higher than the stability threshold — returning NaN")
        return np.nan, np.nan, np.nan, np.nan

    # Initialize response time array with end_time as default
    response_time_array = np.full((nlay, nrow, ncol), times[end_step])

    # Boolean array to track assigned cells
    assigned = np.zeros((nlay, nrow, ncol), dtype=bool)

    # Determine threshold
    if threshold_type == "relative":
        max_initial_diff = np.nanmax(initial_diff) # Max innitial diff across all cells
        threshold = max_initial_diff * (threshold / 100)
    elif threshold_type == "absolute":
        threshold = threshold   
    else:
        raise ValueError("threshold_type must be 'absolute' or 'relative'")

    # Loop through transient times and compute absolute residual diffusion (relaxation)
    for t in range(start_step, end_step + 1):
        relaxation = np.abs(transient_heads[t] - steady_state_heads)
        mask = (relaxation <= threshold) & (~assigned)
        response_time_array[mask] = times[t] - times[start_step]
        assigned[mask] = True
    
    # Save response time array if requested (is saved in the original units of times_list, eg. days)
    response_time_array[np.isnan(initial_diff)] = np.nan
    if save_array and array_output_folder:
        np.save(f"{array_output_folder}/{array_name}", response_time_array)
    
    # Fill irresponsive cells to either the max found response time, start time, or Nan (for visualization)
    if fill == "max":
        response_time_array[np.isnan(initial_diff)] = np.nan
        max_response_time = np.nanmax(response_time_array)
        response_time_array[np.isnan(initial_diff)] = max_response_time
    elif fill == "start":
        response_time_array[np.isnan(initial_diff)] = 0
    elif fill == "nan":
        response_time_array[np.isnan(initial_diff)] = np.nan
    else:
        raise ValueError("fill must be 'max', 'start', or 'nan'")

    # Plotting
    fig = plt.figure(figsize=(19, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Use middle row for cross-section
    mx = flopy.plot.PlotCrossSection(ax=ax, model=gwf, line={"row": nrow // 2})
    pa = mx.plot_array((response_time_array) / 360, alpha=1, cmap="viridis", vmin=0) #Plots in years
    mx.plot_grid(color="0.5", alpha=0.2)

    # Default color mapping based on boundary condition type
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"}

    # Dynamically plot boundary conditions based on keywords
    if boundary_keywords:
        for bc in boundary_keywords:
            # Determine color based on the keyword
            bc_color = None
            for key in color_map:
                if key in bc:  # Check if the keyword contains the key
                    bc_color = color_map[key]
                    break
            # Plot the boundary condition with the appropriate color
            if bc_color:
                mx.plot_bc(bc, color=bc_color)

    if pa is not None:
            cb = plt.colorbar(pa, ax=ax)
            cb.set_label("Response time - MAT + VAT (years)")
            cb.set_label("Response time (years)")

    if interfaces is not None:
        try:
            ncol = gwf.modelgrid.ncol
            dcol = gwf.modelgrid.delr if np.isscalar(gwf.modelgrid.delr) else np.mean(gwf.modelgrid.delr)
            x = np.arange(ncol) * dcol

            # Plot each interface
            for k in range(interfaces.shape[0]):
                ax.plot(x, interfaces[k, nrow // 2, :], "k-", lw=1.0)

        except Exception as e:
            print(f"Could not plot interfaces: {e}")

    ax.set_title(f"Response time to absolute residual diffusion of {threshold} m")
    ax.set_aspect(ve)
    plt.tight_layout()

    if save_plot and fig_output_folder:
        fig.savefig(f"{fig_output_folder}/{fig_name}", dpi=300)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)    

    # ------------------------------ Histogram plotting ------------------------------ #

    # Flatten, remove NaNs, and convert to years
    hist_data = (response_time_array) / 360
    hist_data[np.isnan(initial_diff)] = np.nan # Re set irresponsive cells to NaN
    flat_data = hist_data.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]

    # Compute response time statistics (in years)
    mean_value = np.nanmean(flat_data)
    median_value = np.nanmedian(flat_data)
    percentile_95 = np.nanpercentile(flat_data, 95)
    max_value = np.nanmax(flat_data)

    if histogram:
        plt.figure(figsize=(8, 5))
        if flat_data.size == 0:
            print("Warning: No valid data for histogram — skipping histogram plot.")
        else:
            counts, bin_edges, patches = plt.hist(flat_data, bins=histogram_bins, edgecolor='black')

        # Automatically cut the x-axis from the upper limit of the first bin
        first_bin_max = bin_edges[1]
        plt.xlim(first_bin_max, None)

        # Find the max count among bins that are still visible
        visible_mask = bin_edges[:-1] >= first_bin_max
        visible_max = counts[visible_mask].max() if np.any(visible_mask) else counts.max()
        plt.ylim(0, visible_max * 1.05)  # add 5% padding

        # Add annotation box in upper-left corner
        textstr = f"Max response time: {max_value:.2f} years"
        plt.text(
            0.02, 0.97, textstr,
            transform=plt.gca().transAxes,  # position in axes coordinates
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.8))

        # Label and style
        plt.xlabel("Response time (years)")
        plt.ylabel("Frequency")
        plt.title("Histogram of response times")
        plt.tight_layout()

        # Save and/or show
        if fig_output_folder:
            plt.savefig(f"{fig_output_folder}/{histogram_name}", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()
                
    return mean_value, median_value, percentile_95, max_value # return statistics of response time

def response_time_array_relative(
    gwf,
    steady_state_heads,
    transient_heads,
    times_list,
    threshold_percent=5,
    stability_threshold=0.01,
    start_step=0,
    save_array=True,
    save_plot=True,
    show_plot=False,
    boundary_keywords=None,
    max_initial_diff=False,
    fill="nan",
    ve=10,
    bounds=None,
    vmin=None, vmax=None,
    array_output_folder=None,
    array_name="response_time_relative.npy",
    fig_output_folder=None,
    fig_name="Response_time_relative.png",
    histogram=False,
    histogram_bins=None,
    histogram_name="Response_time_relative_histogram.png",
    interfaces=None, 
    log = False):
    """
    Compute the response time from relative residual diffusion.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model object (for plotting purposes)
    steady_state_heads : ndarray
        3D array of steady-state heads (nlay, nrow, ncol)
        representing the final steady state condition after a step change.
    transient_heads : ndarray
        4D array of transient heads (ntime, nlay, nrow, ncol)
        representing the transient simulation results from the initial state to 
        the final steady state.
    times_list : array-like
        Simulation times corresponding to transient_heads
        Assumes input in days.
    threshold_percent : float
        Percent threshold to calculate response time
    stability_threshold : float, optional
        Threshold to treat zero initial differences as NaN
    start_step : int, optional
        Step to start computing response time. It should correspond to the time step
        index where the step change occurs.
    save_array : bool, optional
        Whether to save the response time array as .npy file
    save_plot : bool, optional
        Whether to save the plot
    show_plot : bool, optional
        Whether to display the plot
    boundary_keywords : list of str, optional
        List of boundary condition keywords to plot (e.g. ["RIV", "WEL", "GHB"])
    max_initial_diff : bool, optional
        If True, use the maximum initial difference across all cells for normalization.
        If False, use cell-specific initial differences: This corresponds to the response time described by Carr et al 2018.
    fill : {'max', 'start', 'nan'}, optional
        How to fill cells with zero initial difference:
        - 'max': Fill with the maximum found response time
        - 'start': Fill with the start time
        - 'nan': Fill with NaN
        'nan' is recommended.
    ve : float, optional
        Vertical exaggeration for plotting
    bounds : tuple of float, optional
        Bounds for the response time array (min, max)
        If None, defaults to the min and max of the response time array.
        If "95p" is provided, cuts to the 2.5th and 97.5th percentiles.
    array_output_folder : str, optional
        Folder to save the response time array as .npy
    array_name : str, optional  
        Name of the response time array file
    fig_output_folder : str, optional
        Folder to save the plot
    fig_name : str, optional
        Name of the plot file
    histogram : bool, optional
        Whether to create a histogram of the response times
    histogram_bins : int or sequence, optional
        Number of bins or bin edges for the histogram
    histogram_name : str, optional
        Name of the histogram plot file
    interfaces : ndarray, optional
        Array of interface elevations for plotting 
    log: Boolean
        Uses logarithmic scale for array plotting 

    Returns
    -------
    statistics of response times.
    saves response_time_array : ndarray
        Array of response times in same shape as steady_state_heads (if save_array=True)
    saves plot of response times cross-section (if save_plot=True)
    saves histogram of response times (if histogram=True)
    """

    end_step = transient_heads.shape[0] - 1
    times = np.array(times_list)

    # ------------------------------ Compute relative response time ----------------------------------- #
    nlay, nrow, ncol = steady_state_heads.shape

    # Precompute initial difference (denominator)
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)

    # Mask for "irresponsive" cells
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan
    initial_diff_max = np.nanmax(initial_diff)
    if np.all(np.isnan(initial_diff)):
        print("No residual diffusion higher than the stability threshold — returning NaN")
        return np.nan, np.nan, np.nan, np.nan

    # Initialize response time array with end_time as default
    response_time_array = np.full((nlay, nrow, ncol), times[end_step])

    # Boolean array to track assigned cells
    assigned = np.zeros((nlay, nrow, ncol), dtype=bool)

    # Loop through transient times and compute relative residual diffusion (relaxation)
    for t in range(start_step, end_step+1):
        if max_initial_diff:
            relaxation = np.abs(transient_heads[t] - steady_state_heads) * 100.0 / initial_diff_max
        else:
            relaxation = np.abs(transient_heads[t] - steady_state_heads) * 100.0 / initial_diff
        mask = (relaxation <= threshold_percent) & (~assigned)
        response_time_array[mask] = times[t] - times[start_step]
        assigned[mask] = True

    if bounds == "95p":
        valid_values = response_time_array[~np.isnan(response_time_array)]
        if valid_values.size > 0:
            upper_p = np.nanpercentile(valid_values, 95)
            outlier_mask = response_time_array > upper_p
            response_time_array[outlier_mask] = np.nan

    # Save response time array if requested (is saved in the original units of times_list, eg. days)
    response_time_array[np.isnan(initial_diff)] = np.nan
    if save_array and array_output_folder:
        np.save(f"{array_output_folder}/{array_name}", response_time_array)

    # Fill irresponsive cells to either the max found response time, start time, or Nan (for visualization)
    if fill == "max":
        response_time_array[np.isnan(initial_diff)] = np.nan
        max_response_time = np.nanmax(response_time_array)
        response_time_array[np.isnan(initial_diff)] = max_response_time
    elif fill == "start":
        response_time_array[np.isnan(initial_diff)] = 0
    elif fill == "nan":
        response_time_array[np.isnan(initial_diff)] = np.nan
    else:
        raise ValueError("fill must be 'max', 'start', or 'nan'")

    # Plotting
    fig = plt.figure(figsize=(19, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Use middle row for cross-section
    mx = flopy.plot.PlotCrossSection(ax=ax, model=gwf, line={"row": nrow // 2})

    if log: 
        pa = mx.plot_array((response_time_array) / 360, alpha=1, cmap="viridis", vmin=vmin, vmax=vmax, 
                           norm = LogNorm(vmin=vmin, vmax=vmax))
    else:
        pa = mx.plot_array((response_time_array) / 360, alpha=1, cmap="viridis", vmin=vmin, vmax=vmax)

    mx.plot_grid(color="0.5", alpha=0.2)

    # Default color mapping based on boundary condition type
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"}

    # Dynamically plot boundary conditions based on keywords
    if boundary_keywords:
        for bc in boundary_keywords:
            # Determine color based on the keyword
            bc_color = None
            for key in color_map:
                if key in bc:  # Check if the keyword contains the key
                    bc_color = color_map[key]
                    break
            # Plot the boundary condition with the appropriate color
            if bc_color:
                mx.plot_bc(bc, color=bc_color)

    if interfaces is not None:
        try:
            ncol = gwf.modelgrid.ncol
            dcol = gwf.modelgrid.delr if np.isscalar(gwf.modelgrid.delr) else np.mean(gwf.modelgrid.delr)
            x = np.arange(ncol) * dcol

            # Plot each interface
            for k in range(interfaces.shape[0]):
                ax.plot(x, interfaces[k, nrow // 2, :], "k-", lw=1.0)

        except Exception as e:
            print(f"Could not plot interfaces: {e}")

    if pa is not None:
            cb = plt.colorbar(pa, ax=ax)
            cb.set_label("Response time - MAT + VAT (years)")
            cb.set_label("Response time (years)")

    ax.set_title(f"Response time to {threshold_percent}% relative residual diffusion")
    ax.set_aspect(ve)
    plt.tight_layout()

    if save_plot and fig_output_folder:
        fig.savefig(f"{fig_output_folder}/{fig_name}", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # ------------------------------ Histogram plotting ------------------------------ #
    
    # Flatten, remove NaNs, and convert to years
    hist_data = (response_time_array) / 360
    hist_data[np.isnan(initial_diff)] = np.nan # Re set irresponsive cells to NaN
    flat_data = hist_data.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]

    # Compute response time statistics (in years)
    mean_value = np.nanmean(flat_data)
    median_value = np.nanmedian(flat_data)
    percentile_95 = np.nanpercentile(flat_data, 95)
    max_value = np.nanmax(flat_data)

    if histogram:
        plt.figure(figsize=(8, 5))
        if flat_data.size == 0:
            print("Warning: No valid data for histogram — skipping histogram plot.")
        else:
            counts, bin_edges, patches = plt.hist(flat_data, bins=histogram_bins, edgecolor='black')

        # Automatically cut the x-axis from the upper limit of the first bin (zoom visualization of longer response times)
        first_bin_max = bin_edges[1]
        plt.xlim(first_bin_max, None)
    
        # Find the max count among bins that are still visible
        visible_mask = bin_edges[:-1] >= first_bin_max
        visible_max = counts[visible_mask].max() if np.any(visible_mask) else counts.max()
        plt.ylim(0, visible_max * 1.05)  # add 5% padding

        # Add annotation box in upper-left corner
        textstr = f"Max response time: {max_value:.2f} years"
        plt.text(
            0.02, 0.97, textstr,
            transform=plt.gca().transAxes,  # position in axes coordinates
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.8))

        # Label and style
        plt.xlabel("Response time (years)")
        plt.ylabel("Frequency")
        plt.title("Histogram of response times")
        plt.tight_layout()

        # Save and/or show
        if fig_output_folder:
            plt.savefig(f"{fig_output_folder}/{histogram_name}", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()

    return mean_value, median_value, percentile_95, max_value # return statistics of response time

def perform_mat_analysis(csv_path, time_col=0, var_col=1, fig_output_folder=".", fig_name="mat_analysis.png", sep=","):
    """
    Perform Mean Arrival Time (MAT) analysis on a variable time series.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the data.
    time_col : int
        Index of the time column in the CSV file.
    var_col : int
        Index of the variable column for which to perform the MAT analysis.
    fig_output_folder : str, optional
        Directory where the output figure will be saved. Default is current directory.
    fig_name : str, optional
        Name of the output plot file (e.g., 'result.png'). Default is 'mat_analysis.png'.
    sep : str, optional
        Column separator used in the CSV file (e.g., ',' or ';'). Default is ','.

    Returns
    -------
    tr : float
        The computed characteristic time (in years).
    """
    # === 1. Load data ===
    df = pd.read_csv(csv_path, sep=sep)
    time = df.iloc[:, time_col]
    var = df.iloc[:, var_col]

    # === 2. Compute G(t) ===
    h_0, h_f = var.iloc[0], var.iloc[-1]
    G_t = 1 - (np.abs(var - h_f) / np.abs(h_f - h_0))

    # === 3. Compute g(t) = dG/dt ===
    g_t = G_t.diff() / time.diff()

    # Clean NaN values
    df_clean = pd.DataFrame({
        "time": time,
        "G_t": G_t,
        "g_t": g_t
    }).dropna()

    # === 4. Compute temporal moments ===
    M = np.trapz(df_clean["time"] * df_clean["g_t"], df_clean["time"])
    V = np.trapz((df_clean["time"] - M)**2 * df_clean["g_t"], df_clean["time"])
    tr = M + np.sqrt(V) / 360  # in years

    # === 5. Plot results ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # G(t)
    axes[0].plot(df_clean["time"], df_clean["G_t"], color="purple", label="G(t)")
    axes[0].axvline(tr, color="red", linestyle="--", label=f"tr = {tr:.2f} years")
    axes[0].set_title("Function G(t)")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("G(t)")
    axes[0].legend()
    if tr is not None and np.isfinite(tr):
        xlim_right = 5 * tr
    else:
        xlim_right = df_clean["time"].max()
    axes[0].set_xlim(0, xlim_right)

    # g(t)
    axes[1].plot(df_clean["time"], df_clean["g_t"], color="orange", label="g(t) = dG/dt")
    axes[1].axvline(tr, color="red", linestyle="--", label=f"tr = {tr:.2f} years")
    axes[1].set_title("Derivative g(t)")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("g(t)")
    axes[1].legend()
    if tr is not None and np.isfinite(tr):
        xlim_right = 5 * tr
    else:
        xlim_right = df_clean["time"].max()
    axes[1].set_xlim(0, xlim_right)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(fig_output_folder, exist_ok=True)

    # Save figure
    fig_path = os.path.join(fig_output_folder, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    
    return tr

# --------- Postprocessing response times

def volume_weighted_percentile(values, volumes, percentile):
    mask = np.isfinite(values) & np.isfinite(volumes) & (volumes > 0)
    values = values[mask]
    volumes = volumes[mask]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values_sorted = values[order]
    volumes_sorted = volumes[order]

    cum_vol = np.cumsum(volumes_sorted)
    cum_frac = cum_vol / cum_vol[-1]

    return np.interp(percentile / 100.0, cum_frac, values_sorted)

def volume_weighted_mean(values, volumes):
    mask = np.isfinite(values) & np.isfinite(volumes) & (volumes > 0)
    if np.sum(volumes[mask]) == 0:
        return np.nan
    return np.sum(values[mask] * volumes[mask]) / np.sum(volumes[mask])

def load_cell_volumes(dis_path):
    """
    Load MF6 DIS and compute cell volumes.
    """
    sim_ws = os.path.dirname(dis_path)

    sim = flopy.mf6.MFSimulation.load(
        sim_ws=sim_ws,
        load_only=["dis"],
        verbosity_level=0,)

    model = sim.get_model()
    dis = model.dis

    delr = dis.delr.array
    delc = dis.delc.array
    top = dis.top.array
    botm = dis.botm.array

    nlay, nrow, ncol = botm.shape

    thickness = np.zeros((nlay, nrow, ncol))
    for k in range(nlay):
        if k == 0:
            thickness[k] = top - botm[k]
        else:
            thickness[k] = botm[k - 1] - botm[k]

    area = np.outer(delc, delr)
    volumes = thickness * area[np.newaxis, :, :]

    return volumes

def analyze_results(main_folder, thickness_dict, length_dict, unc_length_dict,
                     subfolder_keyword="parv_", volume_weighted=True,
                     B=None, L=None, B_threshold=None):
    """
    Collect results and append statistics per zone and total.

    main_folder : str
        Path to the main folder containing subfolders with simulation results.
    thickness_dict : dict
        Dictionary mapping zone numbers to their thickness values.
    length_dict : dict
        Dictionary mapping zone numbers to their length values.
    unc_length_dict : dict
        Dictionary mapping zone numbers to their uncertainty in length values.
    subfolder_keyword : str
        Keyword to identify subfolders containing simulation results.
    volume_weighted : bool
        If True (default), tr_ statistics (mean/percentiles) are computed as cell volume-weighted.
        If False, plain (unweighted) mean/percentiles are computed instead.
    B : float, optional
        Global system thickness. If provided, used instead of the
        per-sequence B_seq in the analytical timescale formulas below.
    L : float, optional
        Global system length. If provided, used instead of the
        per-sequence L_seq in the analytical timescale formulas below.
    B_threshold : float, optional
        Global threshold thickness. If provided, used instead of the
        per-sequence threshold_thickness in the analytical timescale
        formulas below.

    Notes on B_seq / L_seq
    -----------------------
    For each zone z, B_seq and L_seq are derived per sequence (zones
    zz <= z) directly from the input dictionaries:

    - ``B_seq`` = sum of ``thickness_dict`` values over zones zz <= z.
    - ``L_seq`` = max of ``length_dict`` values over zones zz <= z.

    So zone 3's B_seq/L_seq are based on zones 1, 2, 3; zone 5's on zones
    1-5; zone 1's on zone 1 only. These are used in place of B and L in
    the analytical timescale formulas below, unless B or L is provided
    directly, in which case that provided value overrides its per-sequence
    counterpart (independently of the other).

    Notes on equivalent properties
    -------------------------------
    - ``Dv_eq``, ``Dh_eq``, ``kv_eq``, ``kh_eq``: for a given zone z these
      are computed using only the zones "overlying" it, i.e. all zones
      zz <= z (zone 1 up to zone z). So zone 3 uses zones 1, 2, 3; zone 5
      uses zones 1, 2, 3, 4, 5; zone 1 uses only zone 1.
    - All the analytical timescales that depend on these equivalents
      (tao_v_eq, tao_h_eq, tr_v_eq, tr_h_eq, tr_mixed_eq, tr_aquifer,
      tr_basin) are computed row-wise, i.e. each zone uses its own
      overlying-stack equivalent rather than a single system-wide value.

    Notes on response-time statistics
    ----------------------------------
    Three types of tr_* statistics (mean/percentiles/max) are computed
    from the raw response-time array:

    - ``*_zone`` (e.g. ``tr_95p_vol_zone``): computed from cells belonging
      only to that single zone (zones_array == z).
    - ``*_seq`` (e.g. ``tr_95p_vol_seq``): computed from cells belonging to
      that zone AND all overlying zones, i.e. zz <= z (same cumulative
      logic as the Dv_eq/Dh_eq/kv_eq/kh_eq equivalents above). So zone 3
      pools cells from zones 1, 2, 3; zone 5 pools cells from zones 1-5;
      zone 1 uses only zone 1.
    - unsuffixed (e.g. ``tr_95p_vol``): computed from the entire domain,
      all zones combined, one value repeated for every row.
    """
    zone_records = []

    # --- select mean/percentile implementation based on the flag ---
    if volume_weighted:
        mean_func = volume_weighted_mean
        pct_func = volume_weighted_percentile
    else:
        mean_func = lambda vals, vols: np.nanmean(vals)
        pct_func = lambda vals, vols, p: np.nanpercentile(vals, p)

    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        if not folder.startswith(subfolder_keyword):
            continue

        setup_path = os.path.join(folder_path, "setup.xlsx")
        print("Reading setup file:", setup_path)
        output_path = os.path.join(folder_path, "mf", "output", "tr_zones_relative_local.csv")

        if not os.path.exists(setup_path):
            print(f"Missing setup file: {setup_path}")
            continue
        if not os.path.exists(output_path):
            print(f"Missing zone output file: {output_path}")
            continue

        # --- READ PARAMETERS ---
        df_setup = pd.read_excel(setup_path, sheet_name="parameters")
        param_dict = df_setup.set_index("par_name")["value"].to_dict()

        zones = sorted({name.split("_")[-1] for name in param_dict.keys() if "_" in name and name.split("_")[-1].isdigit()})
        zones_h = [int(z) for z in zones if int(z) % 2 == 1]  # Aquifers
        zones_v = [int(z) for z in zones if int(z) % 2 == 0]  # Aquitards

        kv = {int(z): param_dict.get(f"kv_{z}") / 86400 for z in zones}  # Converted to m/s
        kh = {int(z): param_dict.get(f"kh_{z}") / 86400 for z in zones}  # Converted to m/s
        ss = {int(z): param_dict.get(f"ss_{z}") for z in zones}
        sy = {int(z): param_dict.get(f"sy_{z}") for z in zones}

        # --- READ INPUT DICTIONARIES ---
        thickness = {int(z): thickness_dict.get(int(z), np.nan) for z in zones}
        length = {int(z): length_dict.get(int(z), np.nan) for z in zones}
        unc_length = {int(z): unc_length_dict.get(int(z), np.nan) for z in zones}

        # --- INITIALIZE OUTPUT DATAFRAME ---
        df_zone_out = pd.DataFrame({"zone": sorted(map(int, zones))})

        # --- COMPUTE Dv, Dh ---
        Dv = {int(z): (kv[int(z)] / ss[int(z)]) for z in zones}
        Dh = {int(z): (kh[int(z)] / ss[int(z)] if int(z) != 1 else kh[int(z)] * thickness[int(z)] / sy[int(z)]) for z in zones}

        zones_sorted = sorted(int(z) for z in zones)

        # ----------------------------------------------------------------------- #
        # -------- PER-SEQUENCE (OVERLYING-ZONES) EQUIVALENTS ------------------- #

        Dv_eq_seq, Dh_eq_seq, kv_eq_seq, kh_eq_seq = {}, {}, {}, {}
        B_seq, L_seq = {}, {}
        for z in zones_sorted:
            subset = [zz for zz in zones_sorted if zz <= z]
            thickness_subset_sum = sum(thickness[zz] for zz in subset)
            length_subset_max = max(length[zz] for zz in subset)

            Dv_eq_seq[z] = thickness_subset_sum / sum(thickness[zz] / Dv[zz] for zz in subset)
            Dh_eq_seq[z] = sum(Dh[zz] * thickness[zz] for zz in subset) / thickness_subset_sum

            kv_eq_seq[z] = thickness_subset_sum / sum(thickness[zz] / kv[zz] for zz in subset)
            kh_eq_seq[z] = sum(kh[zz] * thickness[zz] for zz in subset) / thickness_subset_sum

            B_seq[z] = thickness_subset_sum
            L_seq[z] = length_subset_max

        # Merge computed parameters
        df_zone_out["kv"] = df_zone_out["zone"].map(kv)
        df_zone_out["kh"] = df_zone_out["zone"].map(kh)
        df_zone_out["ss"] = df_zone_out["zone"].map(ss)
        df_zone_out["sy"] = df_zone_out["zone"].map(sy)
        df_zone_out["Dv"] = df_zone_out["zone"].map(Dv)
        df_zone_out["Dh"] = df_zone_out["zone"].map(Dh)
        df_zone_out["thickness"] = df_zone_out["zone"].map(thickness)
        df_zone_out["length"] = df_zone_out["zone"].map(length)
        df_zone_out["unc_length"] = df_zone_out["zone"].map(unc_length)
        df_zone_out["conf_length"] = df_zone_out["length"] - df_zone_out["unc_length"]

        df_zone_out["Dv_eq"] = df_zone_out["zone"].map(Dv_eq_seq)
        df_zone_out["Dh_eq"] = df_zone_out["zone"].map(Dh_eq_seq)
        df_zone_out["kv_eq"] = df_zone_out["zone"].map(kv_eq_seq)
        df_zone_out["kh_eq"] = df_zone_out["zone"].map(kh_eq_seq)
        df_zone_out["B_seq"] = df_zone_out["zone"].map(B_seq)
        df_zone_out["L_seq"] = df_zone_out["zone"].map(L_seq)

        df_zone_out["anisotropy"] = df_zone_out["kh"] / df_zone_out["kv"]

        # ----------------------------------------------------------------------- #
        # -------------------- COMPUTE RESPONSE TIME STATISTICS ----------------- #
        # ----------------------------------------------------------------------- #
        dis_path = os.path.join(folder_path, "mf", "DEESAC.dis")
        zone_path = os.path.join(folder_path, "mf", "zone_array.npy")
        tr_array_path = os.path.join(folder_path, "mf", "output", "response_time_relative_local.npy")

        if os.path.exists(dis_path) and os.path.exists(zone_path) and os.path.exists(tr_array_path):
            volumes_3d = load_cell_volumes(dis_path)
            volumes = volumes_3d.flatten()
            tr_array_3d = np.load(tr_array_path) / 360.0
            tr_array = tr_array_3d.flatten()
            zones_array = np.load(zone_path).flatten()

            # Unconfined (top most active cells) response times
            nlay, nrow, ncol = tr_array_3d.shape
            valid_mask = ~np.isnan(tr_array_3d)
            top_active = np.argmax(valid_mask, axis=0)
            has_active = np.any(valid_mask, axis=0)
            i_idx, j_idx = np.indices((nrow, ncol))
            mask = (j_idx < ncol - 2) & has_active
            k_idx = top_active[mask]
            i_idx = i_idx[mask]
            j_idx = j_idx[mask]
            tr_unc = tr_array_3d[k_idx, i_idx, j_idx]
            vol_unc = volumes_3d[k_idx, i_idx, j_idx]
            df_zone_out["tr_unc_mean_vol"] = mean_func(tr_unc, vol_unc)
            df_zone_out["tr_unc_5p_vol"] = pct_func(tr_unc, vol_unc, 5)
            df_zone_out["tr_unc_median_vol"] = pct_func(tr_unc, vol_unc, 50)
            df_zone_out["tr_unc_95p_vol"] = pct_func(tr_unc, vol_unc, 95)
            df_zone_out["tr_unc_max"] = np.nanmax(tr_unc)
            tr_unc_valid = tr_unc[~np.isnan(tr_unc)]
            kde = gaussian_kde(tr_unc_valid)
            x = np.linspace(tr_unc_valid.min(), tr_unc_valid.max(), 2000)
            pdf = kde(x)
            tr_unc_mode = x[np.argmax(pdf)]
            df_zone_out["tr_unc_mode"] = tr_unc_mode

            # --- per-zone stats ---
            tr_mean_vol = []
            tr_5p_vol = []
            tr_median_vol = []
            tr_90p_vol = []
            tr_95p_vol = []
            tr_max = []

            for z in df_zone_out["zone"]:
                mask = zones_array == z
                if not np.any(mask):
                    tr_mean_vol.append(np.nan)
                    tr_5p_vol.append(np.nan)
                    tr_median_vol.append(np.nan)
                    tr_90p_vol.append(np.nan)
                    tr_95p_vol.append(np.nan)
                    tr_max.append(np.nan)
                    continue

                tr_zone_vals = tr_array[mask]
                vol_zone = volumes[mask]

                tr_mean_vol.append(mean_func(tr_zone_vals, vol_zone))
                tr_5p_vol.append(pct_func(tr_zone_vals, vol_zone, 5))
                tr_median_vol.append(pct_func(tr_zone_vals, vol_zone, 50))
                tr_90p_vol.append(pct_func(tr_zone_vals, vol_zone, 90))
                tr_95p_vol.append(pct_func(tr_zone_vals, vol_zone, 95))
                tr_max.append(np.nanmax(tr_zone_vals))

            df_zone_out["tr_mean_vol_zone"] = tr_mean_vol
            df_zone_out["tr_5p_vol_zone"] = tr_5p_vol
            df_zone_out["tr_median_vol_zone"] = tr_median_vol
            df_zone_out["tr_90p_vol_zone"] = tr_90p_vol
            df_zone_out["tr_95p_vol_zone"] = tr_95p_vol
            df_zone_out["tr_max_zone"] = tr_max

            # --- sequence stats (cumulative over overlying zones, zz <= z) ---
            tr_mean_vol_seq = []
            tr_5p_vol_seq = []
            tr_median_vol_seq = []
            tr_90p_vol_seq = []
            tr_95p_vol_seq = []
            tr_max_seq = []

            for z in df_zone_out["zone"]:
                seq_zones = [zz for zz in zones_sorted if zz <= z]
                mask = np.isin(zones_array, seq_zones)
                if not np.any(mask):
                    tr_mean_vol_seq.append(np.nan)
                    tr_5p_vol_seq.append(np.nan)
                    tr_median_vol_seq.append(np.nan)
                    tr_90p_vol_seq.append(np.nan)
                    tr_95p_vol_seq.append(np.nan)
                    tr_max_seq.append(np.nan)
                    continue

                tr_seq_vals = tr_array[mask]
                vol_seq = volumes[mask]

                tr_mean_vol_seq.append(mean_func(tr_seq_vals, vol_seq))
                tr_5p_vol_seq.append(pct_func(tr_seq_vals, vol_seq, 5))
                tr_median_vol_seq.append(pct_func(tr_seq_vals, vol_seq, 50))
                tr_90p_vol_seq.append(pct_func(tr_seq_vals, vol_seq, 90))
                tr_95p_vol_seq.append(pct_func(tr_seq_vals, vol_seq, 95))
                tr_max_seq.append(np.nanmax(tr_seq_vals))

            df_zone_out["tr_mean_vol_seq"] = tr_mean_vol_seq
            df_zone_out["tr_5p_vol_seq"] = tr_5p_vol_seq
            df_zone_out["tr_median_vol_seq"] = tr_median_vol_seq
            df_zone_out["tr_90p_vol_seq"] = tr_90p_vol_seq
            df_zone_out["tr_95p_vol_seq"] = tr_95p_vol_seq
            df_zone_out["tr_max_seq"] = tr_max_seq

            # --- total/system stats ---
            df_zone_out["tr_mean_vol"] = mean_func(tr_array, volumes)
            df_zone_out["tr_5p_vol"] = pct_func(tr_array, volumes, 5)
            df_zone_out["tr_median_vol"] = pct_func(tr_array, volumes, 50)
            df_zone_out["tr_90p_vol"] = pct_func(tr_array, volumes, 90)
            df_zone_out["tr_95p_vol"] = pct_func(tr_array, volumes, 95)
            df_zone_out["tr_max"] = np.nanmax(tr_array)

            # --- grouped stats: aquifers vs aquitards ---
            mask_aqf = np.isin(zones_array, zones_h) & (zones_array != 1)
            mask_aqt = np.isin(zones_array, zones_v)

            # Aquifers (h)
            if np.any(mask_aqf):
                tr_aqf = tr_array[mask_aqf]
                vol_aqf = volumes[mask_aqf]

                df_zone_out["tr_mean_vol_aqf"] = mean_func(tr_aqf, vol_aqf)
                df_zone_out["tr_5p_vol_aqf"] = pct_func(tr_aqf, vol_aqf, 5)
                df_zone_out["tr_median_vol_aqf"] = pct_func(tr_aqf, vol_aqf, 50)
                df_zone_out["tr_90p_vol_aqf"] = pct_func(tr_aqf, vol_aqf, 90)
                df_zone_out["tr_95p_vol_aqf"] = pct_func(tr_aqf, vol_aqf, 95)
                df_zone_out["tr_max_aqf"] = np.nanmax(tr_aqf)

            # Aquitards (v)
            if np.any(mask_aqt):
                tr_aqt = tr_array[mask_aqt]
                vol_aqt = volumes[mask_aqt]

                df_zone_out["tr_mean_vol_aqt"] = mean_func(tr_aqt, vol_aqt)
                df_zone_out["tr_5p_vol_aqt"] = pct_func(tr_aqt, vol_aqt, 5)
                df_zone_out["tr_median_vol_aqt"] = pct_func(tr_aqt, vol_aqt, 50)
                df_zone_out["tr_90p_vol_aqt"] = pct_func(tr_aqt, vol_aqt, 90)
                df_zone_out["tr_95p_vol_aqt"] = pct_func(tr_aqt, vol_aqt, 95)
                df_zone_out["tr_max_aqt"] = np.nanmax(tr_aqt)

        else:
            print(f"Missing volume or tr data in folder {folder}")
            df_zone_out["tr_mean_vol_zone"] = np.nan
            df_zone_out["tr_5p_vol_zone"] = np.nan
            df_zone_out["tr_median_vol_zone"] = np.nan
            df_zone_out["tr_90p_vol_zone"] = np.nan
            df_zone_out["tr_95p_vol_zone"] = np.nan
            df_zone_out["tr_max_zone"] = np.nan
            df_zone_out["tr_mean_vol_seq"] = np.nan
            df_zone_out["tr_5p_vol_seq"] = np.nan
            df_zone_out["tr_median_vol_seq"] = np.nan
            df_zone_out["tr_90p_vol_seq"] = np.nan
            df_zone_out["tr_95p_vol_seq"] = np.nan
            df_zone_out["tr_max_seq"] = np.nan
            df_zone_out["tr_mean_vol"] = np.nan
            df_zone_out["tr_5p_vol"] = np.nan
            df_zone_out["tr_median_vol"] = np.nan
            df_zone_out["tr_90p_vol"] = np.nan
            df_zone_out["tr_95p_vol"] = np.nan
            df_zone_out["tr_max"] = np.nan
            df_zone_out["tr_mean_vol_aqf"] = np.nan
            df_zone_out["tr_5p_vol_aqf"] = np.nan
            df_zone_out["tr_median_vol_aqf"] = np.nan
            df_zone_out["tr_90p_vol_aqf"] = np.nan
            df_zone_out["tr_95p_vol_aqf"] = np.nan
            df_zone_out["tr_max_aqf"] = np.nan
            df_zone_out["tr_mean_vol_aqt"] = np.nan
            df_zone_out["tr_5p_vol_aqt"] = np.nan
            df_zone_out["tr_median_vol_aqt"] = np.nan
            df_zone_out["tr_90p_vol_aqt"] = np.nan
            df_zone_out["tr_95p_vol_aqt"] = np.nan
            df_zone_out["tr_max_aqt"] = np.nan

        zone_records.append(df_zone_out)

        # ----------------------------------------------------------------------- #
        # -------------------- COMPUTE ANALYTICAL TIMESCALES -------------------- #
        # ----------------------------------------------------------------------- #

        conversion = 1 / (86400 * 360)

        # --- Threshold thickness per sequence: thickness of the layer with the ---
        # --- highest vertical diffusive resistance (b_i / Dv_i) among zz <= z ---
        threshold_thickness_seq = {}
        Dv_threshold_seq = {}
        for z in zones_sorted:
            subset = [zz for zz in zones_sorted if zz <= z]
            resistances = {zz: thickness[zz] / Dv[zz] for zz in subset}
            zz_max_resistance = max(resistances, key=resistances.get)
            threshold_thickness_seq[z] = thickness[zz_max_resistance]
            Dv_threshold_seq[z] = Dv[zz_max_resistance]
        threshold_thickness = df_zone_out["zone"].map(threshold_thickness_seq)
        Dv_threshold = df_zone_out["zone"].map(Dv_threshold_seq)
        df_zone_out["threshold_thickness"] = threshold_thickness
        df_zone_out["Dv_threshold"] = Dv_threshold

        # --- Geometry used in the analytical formulas: each of B, L, B_threshold ---
        # --- overrides its per-sequence counterpart independently if provided ---
        B_geom = B if B is not None else df_zone_out["B_seq"]
        L_geom = L if L is not None else df_zone_out["L_seq"]
        B_threshold_geom = B_threshold if B_threshold is not None else threshold_thickness

        # --- Equivalent homogeneous timescales (per sequence, using that zone's own overlying-stack Dv_eq/Dh_eq) ---
        tao_v_eq = (B_geom**2 / df_zone_out["Dv_eq"]) * conversion
        tao_h_eq = (L_geom**2 / df_zone_out["Dh_eq"]) * conversion
        tr_v_eq = (12 / np.pi**2) * tao_v_eq
        tr_h_eq = (3 / np.pi**2) * tao_h_eq

        # --- Timescales per zone ---
        tao_v_zone = conversion * df_zone_out["thickness"]**2 / df_zone_out["Dv"].values
        tao_h_zone = conversion * df_zone_out["length"]**2 / df_zone_out["Dh"].values
        tr_v_zone = (3 / np.pi**2) * tao_v_zone
        tr_h_zone = (3 / np.pi**2) * tao_h_zone
        tr_zone = np.where(df_zone_out["zone"].isin(zones_h), tr_h_zone, tr_v_zone)

        # Cumulative max of tr_v_zone over the sequence (zz <= z)
        tr_v_zone_seq_max = pd.Series(tr_v_zone).cummax().values

        # --- Timescales for the mixed aquifer formulation (uses each zone's own kh_eq) --- #
        tr_mixed_zone = 3 * conversion * df_zone_out["unc_length"] * df_zone_out["sy"] \
            * (df_zone_out["conf_length"] + df_zone_out["unc_length"] / 2) / (df_zone_out["thickness"] * df_zone_out["kh"])

        tr_mixed_eq = 3 * conversion * df_zone_out["unc_length"] * df_zone_out["sy"] \
            * (df_zone_out["conf_length"] + df_zone_out["unc_length"] / 2) / (df_zone_out["thickness"] * df_zone_out["kh_eq"])

        # --- Revised analytical response time for the deepest confined aquifer (per zone) ---#
        # Penultimate zone in the entire system
        penultimate_zone = zones_sorted[-2]
        tr_v_zone_penultimate = tr_v_zone[zones_sorted.index(penultimate_zone)]
        tr_v_zone_penultimate_all = np.full(
            len(df_zone_out),
            tr_v_zone_penultimate)

        # Revised analytical response time for the deepest confined aquifer
        tr_aquifer = np.where(
            tr_h_eq >= tr_v_zone_penultimate_all,
            1 / ((1 / tr_h_eq) + (1 / tr_v_eq)),
            tr_h_zone)

        tr_basin = np.where( tr_h_eq >= tr_v_zone_seq_max, 
                             1 / ((1 / tr_h_eq) + (1 / tr_v_eq)),
                             tr_v_zone_seq_max)

        # --- Append analytical timescales to dataframe ---
        df_zone_out["r"] = tr_h_eq/tr_v_zone_seq_max
        df_zone_out["tr_v_eq"] = tr_v_eq
        df_zone_out["tr_h_eq"] = tr_h_eq

        df_zone_out["tr_h_zone"] = tr_h_zone
        df_zone_out["tr_v_zone"] = tr_v_zone
        df_zone_out["tr_zone"] = tr_zone

        df_zone_out["tr_v_zone_seq_max"] = tr_v_zone_seq_max

        df_zone_out["tr_mixed_zone"] = tr_mixed_zone
        df_zone_out["tr_mixed_eq"] = tr_mixed_eq

        df_zone_out["tr_aquifer"] = tr_aquifer
        df_zone_out["tr_basin"] = tr_basin

        df_zone_out["folder"] = folder

        df_zone_out["sim"] = np.log10(df_zone_out["tr_95p_vol_seq"])
        df_zone_out["an"] = np.log10(df_zone_out["tr_basin"])
        df_zone_out["diff"] = df_zone_out["sim"] - df_zone_out["an"]

    df_analysis = pd.concat(zone_records, ignore_index=True)
    df_analysis = df_analysis.sort_values(by="Dv_eq").reset_index(drop=True)
    return df_analysis

def loglog_scatter_df(
    df,
    x_column,
    y_column,
    zone_column="zone",
    zone_value=None,
    color_column=None,         
    color_zone_value=None, 
    color_bar_label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    cmap="viridis_r",
    marker_size=50,
    plot_metrics = True,
    min_val = None,
    max_val= None,
    SAVE = False,
    output_path = "loglog_scatter.png"
):
    """
    Creates a log-log scatter plot from a DataFrame with consistent styling.
    Allows separate zones for plotting and for color values.

    Args:
        df: pandas DataFrame.
        x_column: Column name for x-axis values.
        y_column: Column name for y-axis values.
        color_column: Column name for coloring points (optional).
        zone_column: Column name for zone filtering.
        zone_value: Zone value to filter points for plotting.
        color_zone_value: Zone value to select color values.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        cmap: Colormap for points.
        marker_size: Scatter marker size.
        plot_metrics: Whether to compute and display metrics (R², MAE, RMSE, KGE).
        min_val: Minimum value for axis limits.
        max_val: Maximum value for axis limits.
        SAVE: Whether to save the plot.
        output_path: Path to save the plot if SAVE is True.
    """

    # Filter points to plot
    if zone_value is not None:
        df_plot = df[df[zone_column] == zone_value]
    else:
        df_plot = df.copy()

    x = df_plot[x_column].values
    y = df_plot[y_column].values

    # --- Compute metrics in log space ---
    mask = (x > 0) & (y > 0)

    if np.sum(mask) > 1:
        logx = np.log10(x[mask])
        logy = np.log10(y[mask])

        # Linear regression in log space
        slope, intercept = np.polyfit(logx, logy, 1)
        logy_pred = slope * logx + intercept

        # Residuals
        residuals = logy_pred - logy

        # --- Metrics ---
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((logy - np.mean(logy)) ** 2)
        r2 = 1 - ss_res / ss_tot

        mae = np.mean(np.abs(logx-logy))
        rmse = np.sqrt(np.mean((logx-logy) ** 2))
        bias = np.mean(logx-logy)

        # --- KGE (log space) ---
        r = np.corrcoef(logy, logx)[0, 1]
        alpha = np.std(logx) / np.std(logy)
        beta = np.mean(logx) / np.mean(logy)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    else:
        r2 = mae = rmse = bias = kge = np.nan

    # Select color values
    if color_column:
        if color_zone_value is not None:
            # Match color values from another zone
            df_color = df[df[zone_column] == color_zone_value]
            # Ensure alignment by index if same size; otherwise fallback
            if len(df_color) == len(df_plot):
                c = df_color[color_column].values
            else:
                c = df_plot[color_column].values  # fallback to same zone
        else:
            c = df_plot[color_column].values
    else:
        c = 'blue'

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    if color_column:
        # Ensure positive values for LogNorm
        c_pos = np.array(c, dtype=float)
        c_pos[c_pos <= 0] = np.nan
        sc = ax.scatter(
            x, y, c=c_pos, cmap=cmap, s=marker_size,
            edgecolors='black', linewidths=0.5,
            norm=LogNorm(vmin=np.nanmin(c_pos), vmax=np.nanmax(c_pos))
        )
        # cb = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.85)
        # cb.set_label(color_bar_label if color_bar_label else (f"{color_column} (zone {color_zone_value})" if color_zone_value else color_column))

        cb = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.85)
        cb.set_label(color_bar_label if color_bar_label else (f"{color_column} (zone {color_zone_value})" if color_zone_value else color_column))
        #cb.ax.invert_yaxis()   # <-- this flips the colorbar visually

    else:
        sc = ax.scatter(
            x, y, c=c, cmap=cmap, s=marker_size,
            edgecolors='black', linewidths=0.5
        )

    # 1:1 dashed reference line
    if min_val is not None:
        min_val = min_val
    else: 
        min_val = min(np.nanmin(x), np.nanmin(y))
    
    if max_val is not None:
        max_val = max_val
    else:
        max_val = max(np.nanmax(x), np.nanmax(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    # Log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels & title
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)

    ax.set_aspect('equal', adjustable='box')

    # --- Metrics box ---
    if plot_metrics: 
        if not np.isnan(r2):
            ax.text(
                0.05, 0.95,
                (
                    f"$R^2$ = {r2:.3f}\n"
                    f"MAE = {mae:.3f}\n"
                    f"RMSE = {rmse:.3f}\n"
                    f"BIAS = {bias:.3f}\n"
                    f"KGE = {kge:.3f}"
                ),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white',
                        alpha=0.85, edgecolor='black')
            )

    plt.tight_layout()
    if SAVE:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

def loglog_contours_df(
    df,
    x_col,
    y_col,
    z_col,
    zone=None,
    B=None,
    L=None,
    B_threshold=None,
    anis=None,
    plot_threshold=True,
    plot_B_threshold=True,
    plot_anis=True,
    x_label="X variable",
    y_label="Y variable",
    z_label="Response time [years]",
    y_max_log=None,
    y_min_log=None,
    x_min_log=None,
    x_max_log=None, 
    grid_n=80, 
    regression = False,
    SAVE = False,
    output_path_regression = None,
    output_path_interpolation = None):
    
    """
    Fits log10(Z) = a·log10(X) + b·log10(Y) + c
    and produces:
      - Regression contour plot
      - Interpolated contour plot

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe. Must contain a "zone" column when `zone` is
        provided, and, if B/L/B_threshold are to default from it, the
        "B_seq"/"L_seq"/"threshold_thickness" columns produced by
        analyze_results.
    x_col, y_col, z_col : str or array-like
        Column names (when `zone` is provided, these are looked up in the
        zone-subset of df) or, when `zone` is None, array-like values
        (e.g. an already-subsetted pandas Series) for X, Y, Z directly.
    zone : int, optional
        If provided, df is subset to rows where df["zone"] == zone before
        x_col/y_col/z_col are looked up as column names, and B, L, and
        B_threshold default to that row's "B_seq", "L_seq", and
        "threshold_thickness" columns (as produced by analyze_results).
    B, L : float, optional
        Reference system thickness/length: plots the line Y/X = B²/L².
        If `zone` is provided, B and L default to that row's "B_seq"/
        "L_seq" columns unless given directly, in which case the given
        value overrides its counterpart independently of the other.
    B_threshold : float, optional
        Reference threshold thickness: plots the line Y/X = B_threshold²/L².
        If `zone` is provided, defaults to that row's "threshold_thickness"
        column unless given directly, in which case it overrides that value.
    plot_threshold : bool
        If True (default), the diagonal reference lines for B/L,
        B_threshold/L, and anis (whichever are provided) are plotted. If
        False, no diagonal reference lines are plotted regardless of
        B, L, B_threshold, or anis.
    x_label, y_label, z_label : str
        Axis / colorbar labels
    y_max_log : float, optional
        Upper limit in log10(Y); defaults to data max
    grid_n : int
        Grid resolution
    regression : bool
        If True, fits and plots the regression surface; otherwise only interpolation
    SAVE : bool
        If True, saves the figures
    output_path_regression : str
        Path to save the regression figure if SAVE is True
    output_path_interpolation : str
        Path to save the interpolation figure if SAVE is True
    """
    # --- Zone subsetting and geometry lookup ---
    if zone is not None:
        df_zone = df[df["zone"] == zone]
        x_col = df_zone[x_col]
        y_col = df_zone[y_col]
        z_col = df_zone[z_col]

        # This selects the first row of the zone subset, assuming B_seq, L_seq, and threshold_thickness are consistent within the zone.
        # This is the case, since this function is meant to plot outputs of simulations sharing the same geometry through a
        # systematic sensitivity analysis of diffusivities, guaranteeing that B_seq, L_seq, and threshold_thickness are the same for all rows of a given zone.
        if "B_seq" in df_zone.columns:
            B = B if B is not None else df_zone["B_seq"].iloc[0]
        if "L_seq" in df_zone.columns:
            L = L if L is not None else df_zone["L_seq"].iloc[0]
        if "threshold_thickness" in df_zone.columns:
            B_threshold = B_threshold if B_threshold is not None else df_zone["threshold_thickness"].iloc[0]

    # --- Log transforms ---
    d = pd.DataFrame({
    "X": np.asarray(x_col),
    "Y": np.asarray(y_col),
    "Z": np.asarray(z_col),})
    d["logX"] = np.log10(d["X"])
    d["logY"] = np.log10(d["Y"])
    d["logZ"] = np.log10(d["Z"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Regression ---
    Xmat = d[["logX", "logY"]].values
    yvec = d["logZ"].values

    model = LinearRegression().fit(Xmat, yvec)
    a, b = model.coef_
    c = model.intercept_
    r2 = r2_score(yvec, model.predict(Xmat))
    K = 10 ** c

    print(f"logZ = {a:.3f}·logX + {b:.3f}·logY + {c:.3f}")
    print(f"R² = {r2:.4f}")
    print(f"Physical form: Z = {K:.3e} · X^{a:.3f} · Y^{b:.3f}")

    # --- Grids ---
    x_log_min = x_min_log if x_min_log is not None else d["logX"].min()
    x_log_max = x_max_log if x_max_log is not None else d["logX"].max()
    y_log_min = y_min_log if y_min_log is not None else d["logY"].min()
    y_log_max = y_max_log if y_max_log is not None else d["logY"].max()

    Xg = np.linspace(x_log_min, x_log_max, grid_n)
    Yg = np.linspace(y_log_min, y_log_max, grid_n)
    Xgrid, Ygrid = np.meshgrid(Xg, Yg)

    # --- Surfaces ---
    Z_pred = a * Xgrid + b * Ygrid + c

    Z_interp = griddata(
        (d["logX"], d["logY"]),
        d["logZ"],
        (Xgrid, Ygrid),
        method="cubic",)

    # --- Fill NaNs ---
    def _fill_nans(Z):
        Zf = Z.copy()
        if np.isnan(Zf).any():
            Z_lin = griddata(
                (d["logX"], d["logY"]),
                d["logZ"],
                (Xgrid, Ygrid),
                method="linear",)
            Zf = np.where(np.isnan(Zf), Z_lin, Zf)
        if np.isnan(Zf).any():
            Z_near = griddata(
                (d["logX"], d["logY"]),
                d["logZ"],
                (Xgrid, Ygrid),
                method="nearest",)
            Zf = np.where(np.isnan(Zf), Z_near, Zf)
        return Zf

    Z_interp_filled = _fill_nans(Z_interp)

    # --- Levels ---
    vmin = np.floor(d["logZ"].min())
    vmax = np.ceil(d["logZ"].max())
    levels_fill = np.linspace(vmin, vmax, 100)
    levels_contour = np.arange(vmin, vmax + 1)

    # --- Reference line ---
    if B is not None and L is not None:
        ratio_log = np.log10(4 * (B ** 2) / (L ** 2))
        logX_line = np.linspace(x_log_min, x_log_max, 200)
        logY_line = logX_line + ratio_log
    
    if B_threshold is not None and L is not None:
        ratio_threshold_log = np.log10(4 * (B_threshold ** 2) / (L ** 2))
        logX_line_threshold = np.linspace(x_log_min, x_log_max, 200)
        logY_line_threshold = logX_line_threshold + ratio_threshold_log
    
    if anis is not None:
        ratio_anis_log = np.log10((np.sqrt(anis)))
        logX_line_anis = np.linspace(x_log_min, x_log_max, 200)
        logY_line_anis = logX_line_anis + ratio_anis_log

    # --- Helpers ---
    def _clean_axes(ax):
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("black")
            s.set_linewidth(1)
        ax.tick_params(direction="out", colors="black", labelsize=11)
        # Only place ticks at whole orders of magnitude (integers in log10
        # space), so every tick corresponds to an exact 10^n and none are
        # produced by truncating an arbitrary non-integer tick position.
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        return ax

    def log_formatter(val, pos):
        return rf"$10^{{{int(round(val))}}}$"

    def fmt_pow10(val):
        return f"{int(10 ** val):g}"

    def format_cb(cb):
        ticks = np.arange(vmin, vmax + 1)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([f"{int(10 ** i):,}" for i in ticks])
        cb.set_label(z_label, fontsize=11)

    def add_ratio_legend(cb):
        h = Line2D([0], [0], color="red", linestyle="--", linewidth=1,
                   label=r"$Y/X = B^2/L^2$")
        cb.ax.legend(handles=[h], loc="upper left",
                     bbox_to_anchor=(0, -0.05), fontsize=10)

    # ============================
    # Regression plot
    # ============================
    if regression:

        fig1, ax1 = plt.subplots(figsize=(6, 6.5), dpi=100)

        cf1 = ax1.contourf(
            Xg, Yg, Z_pred,
            levels=levels_fill, cmap="viridis",
            vmin=vmin, vmax=vmax, extend="both"
        )

        cont1 = ax1.contour(
            Xg, Yg, Z_pred,
            levels=levels_contour,
            colors="black", linewidths=0.6, linestyles="dashed"
        )
        ax1.clabel(cont1, fmt=fmt_pow10, fontsize=10)

        ax1.scatter(
            d["logX"], d["logY"],
            c=d["logZ"], cmap="viridis",
            vmin=vmin, vmax=vmax,
            s=40, edgecolors="black", linewidths=0.5
        )

        if B is not None and L is not None and plot_threshold: 
            ax1.plot(logX_line, logY_line, "r--", lw=1)
        
        if B_threshold is not None and L is not None and plot_B_threshold:
            ax1.plot(logX_line_threshold, logY_line_threshold, "b--", lw=1)   
        
        if anis is not None and plot_anis:       
            ax1.plot(logX_line_anis, logY_line_anis, "g--", lw=1)

        ax1 = _clean_axes(ax1)
        ax1.set_aspect("equal")
        ax1.set_xlim(x_log_min, x_log_max)
        ax1.set_ylim(y_log_min, y_log_max)
        ax1.xaxis.set_major_formatter(FuncFormatter(log_formatter))
        ax1.yaxis.set_major_formatter(FuncFormatter(log_formatter))
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)

        cb1 = fig1.colorbar(cf1, ax=ax1, shrink=0.85)
        format_cb(cb1)
        if B is not None and L is not None:
            add_ratio_legend(cb1)

        plt.tight_layout()
        if SAVE and output_path_regression is not None:
            plt.savefig(output_path_regression, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    # ============================
    # Interpolated plot
    # ============================
    fig2, ax2 = plt.subplots(figsize=(6, 6.5), dpi=200)

    cf2 = ax2.contourf(
        Xg, Yg, Z_interp_filled,
        levels=levels_fill, cmap="viridis",
        vmin=vmin, vmax=vmax, extend="both"
    )

    cont2 = ax2.contour(
        Xg, Yg, Z_interp_filled,
        levels=levels_contour,
        colors="black", linewidths=0.6, linestyles="dashed"
    )
    ax2.clabel(cont2, fmt=fmt_pow10, fontsize=10)

    ax2.scatter(
        d["logX"], d["logY"],
        c=d["logZ"], cmap="viridis",
        vmin=vmin, vmax=vmax,
        s=40, edgecolors="black", linewidths=0.5
    )

    if B is not None and L is not None and plot_threshold:
        ax2.plot(logX_line, logY_line, "r--", lw=1)
    
    if B_threshold is not None and L is not None and plot_B_threshold:
        ax2.plot(logX_line_threshold, logY_line_threshold, "b--", lw=1)
    
    if anis is not None and plot_anis:       
        ax2.plot(logX_line_anis, logY_line_anis, "g--", lw=1)

    ax2 = _clean_axes(ax2)
    ax2.set_aspect("equal")
    ax2.set_xlim(x_log_min, x_log_max)
    ax2.set_ylim(y_log_min, y_log_max)
    ax2.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)

    cb2 = fig2.colorbar(cf2, ax=ax2, shrink=0.85)
    format_cb(cb2)
    if B is not None and L is not None:
        add_ratio_legend(cb2)

    plt.tight_layout()
    if SAVE and output_path_interpolation is not None:
        plt.savefig(output_path_interpolation, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return {
        "a": a,
        "b": b,
        "c": c,
        "K": K,
        "R2": r2,
        "model": model,
        "X_grid": Xgrid,
        "Y_grid": Ygrid,
        "Z_pred": Z_pred,
        "Z_interp": Z_interp_filled,
    }