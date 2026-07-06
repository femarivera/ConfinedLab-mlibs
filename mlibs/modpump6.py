# ==========================================================================================
#  modpump6.py - Steady State Pumping Analysis Utilities for MODFLOW 6 Groundwater Models
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  This module provides utilities for analyzing well pumping scenarios in steady state MODFLOW 6 
#  models. It automates pumping rate iteration and generates plots and animations for flow budgets 
#  and well abstraction analysis.
#
#  MAIN FEATURES:
#  --------------
#  - Update and iterate well pumping rates for MODFLOW 6 steady state simulations.
#  - Analyze induced recharge, natural discharge, and captured discharge.
#  - Visualize cross-sections and create pumping scenario animations.
#  - Generate water budget plots and well abstraction summaries.
#
# ==========================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import flopy
import os
import sys
import imageio
import re
from pathlib import Path
import shutil
import subprocess
import gc
import time
from mlibs import modplot6 # type: ignore

def simplify_name(name):
    """
    Simplifies a column or component name for display in plots and legends.

    If the input string contains parentheses, extracts the text inside and combines it
    with the text after the next underscore. Otherwise, replaces underscores with spaces.

    Args:
        name (str): The input string to simplify (e.g., a column name).

    Returns:
        str: Simplified name for display or legend.
    """
    # If parentheses are present
    if '(' in name and ')' in name:
        # Extract text inside parentheses
        simplified = name.split('(')[1].split(')')[0].strip()
        # Extract text after the underscore and strip spaces
        after_underscore = name.split(')')[1].split('_')[1].strip()
        # Combine both parts with a space
        simplified = simplified + " " + after_underscore
        return simplified
    else:
    # If no parentheses, replace the underscore with a space
        simplified = name.replace('_', ' ')  
        return simplified
       
def update_well_pumping_rate_steady(gwf, 
                                    wel_spd, 
                                    wel, 
                                    q):
    """
    Updates the pumping rates for all wells in wel_spd[0] and modifies 
    the corresponding well package (wel).
    Used for STEADY STATE SIMULATIONS.
    
    Parameters:
        gwf (flopy.mf6.ModflowGwf): The groundwater flow model object.
        wel_spd (dict): The dictionary containing well stress period data.
        wel (flopy.mf6.ModflowGwfwel): The well package object.
        q (tuple or list): Pumping rates for each well. Length must match
                           number of wells in wel_spd[0].
    
    Returns:
        None (modifies wel_spd in place and updates the wel object).
    """

    n_wells = len(wel_spd[0])

    # Check input length
    if len(q) != n_wells:
        raise ValueError(f"Length of q ({len(q)}) must equal number of wells ({n_wells})")

    # Update each well with its corresponding pumping rate
    for i in range(n_wells):
        wel_spd[0][i] = (wel_spd[0][i][0],  # layer
                         wel_spd[0][i][1],  # row
                         wel_spd[0][i][2],  # column
                         q[i])              # pumping rate
    
    # Update the well package (wel) with the new pumping rate for stress period 0
    wel = flopy.mf6.ModflowGwfwel(gwf, 
                                 pname = "wel",
                                 save_flows = True,
                                 stress_period_data = wel_spd)

def iterate_pumping_rate_steady(model_ws,
                                sim, 
                                gwf, 
                                wel_spd, 
                                wel, 
                                q_values,
                                q_ref,
                                budget_csv_file,
                                head_path_file,
                                row, 
                                figure_dir,
                                csv_output_path, 
                                boundary_keywords = None,
                                layers=False,
                                animate=False,
                                animation_name = "cross_section_animation_ss.gif",
                                duration=0.5,
                                ve=10, 
                                save_budget = False, 
                                save_wells = False,
                                save_csv = False, 
                                interfaces=None):
    """
    Function to iterate through different pumping rates, run simulations, and generate plots.
    Used for STEADY STATE SIMULATIONS.

    Parameters:
        model_ws (str): Path to the model workspace directory.
        output_folder (str): Path to the output folder where head and budget files are written.
        sim (flopy.mf6.MFSimulation): The simulation object.
        gwf (flopy.mf6.ModflowGwf): The groundwater flow model object.
        wel_spd (dict): The dictionary containing well stress period data.
        wel (flopy.mf6.ModflowGwfwel): The well package object.
        q_values (list of tuples): List of n_well tuples. Each tuple contains pumping rates for each iteration for each well.
                                    len(q_values) = n_wells and len(q_values[0]) = n_iterations.
        q_ref (tuple): Reference pumping rate for initial simulation (normally 0 for natural conditions).
                       Length must match number of wells.
        budget_csv_file (str): Path to the CSV file containing budget data.
        head_path_file (str): Path to the head output file (.hds).
        row (int): Row index for cross-section plotting.
        figure_dir (str): Directory to save figures.
        csv_output_path (str): Path to the CSV output file.
        boundary_keywords (list of str, optional): List of boundary condition keywords to include in cross-section plots.
        layers (bool): Whether to plot layers legend in cross-section plots.
        animate (bool): Whether to create an animation of cross-sections.
        animation_name (str): Name of the output animation file.
        duration (float): Duration (in seconds) for each frame in the animation.
        ve (int): Vertical exaggeration for cross-section plots.
        save_budget (bool): Whether to save the water budget plot.
        save_wells (bool): Whether to save the water to wells plot.
        save_csv (bool): Whether to save the pumping analysis results to a CSV file.

    Returns:
        Plot of induced recharge, natural discharge, and captured discharge vs pumping rates
    """

    # --------------------------------------------------------------------- #
    # ------------------- REFERENCE PUMPING SCENARIO ---------------------- #
    # --------------------------------------------------------------------- #
    # A default reference pumping scenario of no pumping (natural conditions) is used.

    # Read initial simulation with q_ref to get the reference inflow and reference outflow
    update_well_pumping_rate_steady(gwf, wel_spd, wel, q_ref) 
    sim.write_simulation()
    success, buff = sim.run_simulation()
    if not success:
        print(f"Simulation failed for pumping rate {q_ref}")
        return

    # Path to the reference CSV file
    csv_file_path = os.path.join(budget_csv_file)
    data = pd.read_csv(csv_file_path)

    # Get reference inflow and outflow from TOTAL_IN and TOTAL_OUT (excluding well abstractions)
    reference_inflow = data['TOTAL_IN'].iloc[-1]
    reference_other_out = [col for col in data.columns if col.endswith("_OUT") and "WEL" not in col and col != "TOTAL_OUT"]
    reference_outflow = data[reference_other_out].sum(axis=1).iloc[-1]

    # --------------------------------------------------------------------- #
    # ----------------------------- PUMPING RATES ------------------------- #
    # --------------------------------------------------------------------- #
    
    # Initialize lists of outputs
    induced_recharge_results = []
    natural_discharge_results = []
    captured_discharge_results = []
    pumping_rates = []
    image_paths = []

    # Identify relevant columns (excluding first and last columns that correspond to time and percent difference)
    relevant_columns = data.columns[1:-1] 
    # Initialize a dictionary to store the results for each column in relevant_columns
    column_results = {col: [] for col in relevant_columns}

    n_wells = len(wel_spd[0])
    assert len(q_values) == n_wells, (
        f"q_values must have {n_wells} tuple entries (one per well), "
        f"but got {len(q_values)}")
    
    n_iterations = len(q_values[0])
    for well_idx, rates in enumerate(q_values):
        assert len(rates) == n_iterations, (
            f"Well {well_idx} has {len(rates)} pumping rates, "
            f"but expected {n_iterations}")
    
    for it in range(n_iterations):
        # Build a tuple of pumping rates for this iteration as input for the update well function
        q_tuple = tuple(q_values[well_idx][it] for well_idx in range(n_wells))

        # Update pumping rates
        update_well_pumping_rate_steady(gwf, wel_spd, wel, q_tuple)

        print(f"Running simulation {it+1} with pumping rates: {q_tuple} m³/day")
        sim.write_simulation()
        success, buff = sim.run_simulation()
        if not success:
            print(f"Simulation {it+1} failed for pumping rates {q_tuple}")
            continue
        
        # ---------------------------- ANIMATION --------------------------------- #
        if animate:
            cross_section_dir = os.path.join(figure_dir, "cross_sections_ss")
            # Create directory if it does not exist
            if cross_section_dir and not os.path.exists(cross_section_dir):
                os.makedirs(cross_section_dir)
            
            #Plot cross section
            plt.ioff()
            fig, ax = plt.subplots(figsize=(19, 4))
            modplot6.plot_cross_section_row(gwf, head_path_file, row, model_ws,
                                            boundary_keywords = boundary_keywords,
                                            flow_dir = False,
                                            surface = True, layers=layers, ve=ve,
                                            show = False, save = False, ax=ax, interfaces=interfaces)
            plt.title(f"Cross-Section for Total Pumping Rate: {abs(sum(q_tuple)):.1f} m³/day")

            # Save the plot as an image and append the path to image_paths
            image_path = os.path.join(cross_section_dir, f"cross_section_{it}.png")
            fig.savefig(image_path, dpi=300)
            image_paths.append(image_path)
            plt.close(fig)

        # ------------------- DATA FOR FLOW BUDGET PLOTS ----------------------- #

        # Path to the current CSV file
        csv_file_path = os.path.join(budget_csv_file)

        # Load the CSV file generated by the current simulation
        data = pd.read_csv(csv_file_path)

        # Append the last (and only) value of each relevant column to the corresponding list in column_results
        for col in relevant_columns:
            column_results[col].append(data[col].iloc[-1])

        # Compute induced recharge, natural discharge, and capture
        total_in = data['TOTAL_IN'].iloc[-1]
        induced_recharge = total_in - reference_inflow 
        
        columns_other_out = [col for col in data.columns if col.endswith("_OUT") and "WEL" not in col and col != "TOTAL_OUT"]
        natural_discharge = data[columns_other_out].sum(axis=1).iloc[-1]
        
        # Compute captured discharge
        captured_discharge = reference_outflow - natural_discharge

        # Store results
        pumping_rates.append(abs(sum(q_tuple))) # Total pumping rate (absolute value)
        induced_recharge_results.append(induced_recharge)
        natural_discharge_results.append(natural_discharge)
        captured_discharge_results.append(captured_discharge)

    # --------------------------------------------------------------------- #
    # -------------------------- PLOTS FLOW BUDGET ------------------------ #
    # --------------------------------------------------------------------- #

    # Split relevant columns into two groups based on "_IN" and "_OUT"
    columns_in = [col for col in relevant_columns if "_IN" in col]
    columns_out = [col for col in relevant_columns if "_OUT" in col]
    
    #Simplify names of columns
    simplified_columns_in = [simplify_name(col) for col in columns_in]
    simplified_columns_out = [simplify_name(col) for col in columns_out]
    simplified_columns_names = [simplify_name(col) for col in column_results]
    
    #Replace column names on  the column results dictionary
    simplified_column_results = {}

    # Iterate over both the original column names and their corresponding simplified names
    for old_key, new_key in zip(column_results.keys(), simplified_columns_names):
        simplified_column_results[new_key] = column_results[old_key]  # Assign the same list of results

    # Determine the maximum number of rows needed
    n_rows = max(len(columns_in), len(columns_out))

    # Create the figure and axes with 2 columns and n_rows
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 5))

    # Ensure axes are treated as a 2D array for easier indexing
    axes = axes if isinstance(axes, np.ndarray) and len(axes.shape) == 2 else np.array([axes]).reshape(n_rows, 2)

    # Plot "_IN" columns in the first column
    for i, col in enumerate(simplified_columns_in):
        ax = axes[i, 0]  # Access the subplot for this position
        ax.plot(pumping_rates, simplified_column_results[col], marker='o', label=col)
        ax.set_xlabel('Pumping Rate (m³/day)')
        ax.set_ylabel(f'{simplify_name(col)} (m³/day)')
        ax.grid(True)
        ax.legend()

    # Plot "_OUT" columns in the second column
    for i, col in enumerate(simplified_columns_out):
        ax = axes[i, 1]  # Access the subplot for this position
        ax.plot(pumping_rates, simplified_column_results[col], marker='o', label=col)
        ax.set_xlabel('Pumping Rate (|m³/day|)')
        ax.set_ylabel(f'{simplify_name(col)} (m³/day)')
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(n_rows):
        if i >= len(columns_in):
            axes[i, 0].axis('off')  # Hide unused plots in the first column
        if i >= len(columns_out):
            axes[i, 1].axis('off')  # Hide unused plots in the second column

    plt.tight_layout()
    
    if save_budget:
        image_path = os.path.join(figure_dir, f"modpump6_water budget.png")
        fig.savefig(image_path, dpi=300)
        plt.close(fig)         

    # --------------------------------------------------------------------- #
    # -------------------------- PLOTS WATER TO WELLS --------------------- #
    # --------------------------------------------------------------------- #

    # Plot induced recharge, natural discharge, and captured discharge vs pumping rate
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 8))

    # Induced Recharge vs Pumping Rate
    axs2[0].plot(pumping_rates, induced_recharge_results, marker='o', label='Induced Inflows')
    axs2[0].set_xlabel('Pumping Rate (m³/day)')
    axs2[0].set_ylabel('Induced Inflows (m³/day)')
    axs2[0].grid(True)
    axs2[0].legend()

    # Natural Discharge vs Pumping Rate
    axs2[1].plot(pumping_rates, natural_discharge_results, marker='o', label='Natural Outflows', color='green')
    axs2[1].set_xlabel('Pumping Rate (m³/day)')
    axs2[1].set_ylabel('Natural Outflows (m³/day)')
    axs2[1].grid(True)
    axs2[1].legend()

    # Captured Discharge vs Pumping Rate
    axs2[2].plot(pumping_rates, captured_discharge_results, marker='o', label='Captured Outflows', color='orange')
    axs2[2].set_xlabel('Pumping Rate (m³/day)')
    axs2[2].set_ylabel('Captured Outflows (m³/day)')
    axs2[2].grid(True)
    axs2[2].legend()

    plt.tight_layout()
    
    if save_wells:
        image_path = os.path.join(figure_dir, f"modpump6_water to wells.png")
        fig2.savefig(image_path, dpi=300)
        plt.close(fig2) 

    # Create animation from saved images
    if animate:
        if image_paths:
            with imageio.get_writer(os.path.join(figure_dir, animation_name), 
                                    mode='I', duration=duration) as writer:
                for image_path in image_paths:
                    image = imageio.imread(image_path)
                    writer.append_data(image)
        else:
            print("No successful simulations to animate.")

    # Save results as CSV
    if save_csv:
        # Prepare dictionary for DataFrame
        results_dict = {}
        results_dict['Pumping_Rate'] = pumping_rates
        # Add relevant columns
        for col in relevant_columns:
            results_dict[col] = column_results[col]
        # Add induced recharge, natural discharge, and captured discharge
        results_dict['Induced_Recharge'] = induced_recharge_results
        results_dict['Natural_Discharge'] = natural_discharge_results
        results_dict['Captured_Discharge'] = captured_discharge_results

        df_results = pd.DataFrame(results_dict)

        df_results.to_csv(csv_output_path, index=False)
        print(f"Pumping analysis results saved to {csv_output_path}")

def iterate_pumping_rate_transient(setup_file, model_file, iterations_output_dir, summary_dir, model_ws_name, 
                                   budget_file_name, zonebud_file_name, head_file_name, cbb_summary_file_name):
    """
    Function to iterate a groundwater model over different pumping rates defined in an Excel setup file
    (on sheet named q_values_tr). For each pumping rate, the model is run in a unique workspace and the 
    relevant output files are copied to a summary directory.

    This is useful for small models that do not have large input or output files. For larger models, use
    the function iterate_pumping_rate_transient_eff.

    Args:
        setup_file (str): Path to the Excel setup file containing well and pumping rate information.
        model_file (str): Path to the groundwater model Python script.
        iterations_output_dir (str): Directory where output files from model iterations will be saved.
        summary_dir (str): Directory where summary of results will be saved.
        model_ws_name (str): Base name for the model workspace directories.
        budget_file_name (str): Name of the budget output file generated by the model.
        zonebud_file_name (str): Name of the zone budget output file generated by the model.
        head_file_name (str): Name of the head observation output file generated by the model.
        cbb_summary_file_name (str): Name of the cell-by-cell budget summary output file generated by the model.
    """
    os.makedirs(iterations_output_dir, exist_ok=True)

    # --------------------------------------------------------------------------------------- #
    # ------------------------------- PREPARE ITERATION FILE -------------------------------- #
    # --------------------------------------------------------------------------------------- #
    model_file = os.path.abspath(model_file)
    folder, filename = os.path.split(model_file)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_it{ext}"
    new_file_path = os.path.join(folder, new_filename)

    # Copy the original file
    shutil.copy(model_file, new_file_path)

    # Read and modify the new file with placeholders
    with open(new_file_path, "r") as f:
        lines = f.readlines()

    setup_replaced = False
    for i, line in enumerate(lines):
        # if "sys.path.append('..')" in line:
        #     lines[i] = f"sys.path.append(r'{mlibs_path}')\n"
        if not setup_replaced and line.strip().startswith("setup_file ="):
            lines[i] = f"setup_file = r'{os.path.abspath(setup_file)}'\n" # Excel file containing model setup parameters\n"
            setup_replaced = True

    # Save back the modified file
    with open(new_file_path, "w") as f:
        f.writelines(lines)
    print("Iteration script created at:", new_file_path)

    # --------------------------------------------------------------------------------------- #
    # ------------------------------- UPDATE AND RUN MODEL ---------------------------------- #
    # --------------------------------------------------------------------------------------- #

    # Load setup
    # file contining the pumping rates used by modflow6
    well_df = pd.read_excel(setup_file, sheet_name="wells")
    well_st_df = pd.read_excel(setup_file, sheet_name="wells_st")

    # q-values for each iteration
    q_df = pd.read_excel(setup_file, sheet_name="q_values_tr")
    q_st_df = pd.read_excel(setup_file, sheet_name="q_values_st")       

    # Identify iteration columns (all except well_id + time)
    iter_cols = [c for c in q_df.columns if c not in ["well_id", "time", "comment"]]
    n_iterations = len(iter_cols)

    for i, col in enumerate(iter_cols, start=1):

        print(f"\n--- Running iteration {i}/{n_iterations} with {col} ---")

        # ------------------------------------------------------------------------------------- #
        # -------------------------------- PREPARE SETUP FILE --------------------------------- #
        # ------------------------------------------------------------------------------------- #

        # Merge well_df with the selected q column
        merged = well_df.drop(columns=["q"]).merge(
            q_df[["well_id", "time", col]],
            on=["well_id", "time"],
            how="left")
        # Rename current iteration q value column to "q"
        merged = merged.rename(columns={col: "q"})
        # Write updated wells sheet back to Excel (overwrite only that sheet)
        with pd.ExcelWriter(setup_file, mode="a", if_sheet_exists="replace") as writer:
            merged.to_excel(writer, sheet_name="wells", index=False)

        # Same for the steady state wells
        merged_st = well_st_df.drop(columns=["q"]).merge(
            q_st_df[["well_id", col]],
            on=["well_id"],
            how="left")
        merged_st = merged_st.rename(columns={col: "q"})
        with pd.ExcelWriter(setup_file, mode="a", if_sheet_exists="replace") as writer:
            merged_st.to_excel(writer, sheet_name="wells_st", index=False)

        # --------------------------------------------------------------------------------------- #
        # ------------------------------- ITERATE MODEL ----------------------------------------- #
        # --------------------------------------------------------------------------------------- #

        # # Create a unique model workspace directory name based on the parameter value
        # model_ws = os.path.join(iterations_output_dir, f"{model_ws_name}_it_{col}")
        
        # # Create the directory for model_ws if it doesn't exist
        # os.makedirs(model_ws, exist_ok=True)
        
        # # Copy the iteration script into the unique model workspace folder and get the path
        # shutil.copy(new_file_path, model_ws)
        # script_path = os.path.join(model_ws, new_filename)
        
        # # Run the script inside the unique model workspace folder
        # # You can use subprocess to execute the script in that directory
        # subprocess.run(["python", script_path], cwd=model_ws)

        # print(f"Model run completed for iteration {i} with q={col}, model_ws={model_ws}")

        model_ws = os.path.join(iterations_output_dir, f"{model_ws_name}_it_{col}")
        os.makedirs(model_ws, exist_ok=True)

        # Just run the same iteration script from its original location
        subprocess.run(
            ["python", new_file_path],
            cwd=model_ws
        )

        print(f"Model run completed for iteration {i} with q={col}, model_ws={model_ws}")


    # --------------------------------------------------------------------------------------- #
    # ------------------------------- MANAGE OUTPUT FILES ----------------------------------- #
    # --------------------------------------------------------------------------------------- #

    # Define a destination directory to summarize results of the iterations
    results_folder = summary_dir
    os.makedirs(results_folder, exist_ok=True)

    # Loop through the sub-folders in the output directory to get relevant files
    for folder_name in os.listdir(iterations_output_dir):
        # Check if the folder matches the pattern "model_ws_name_it_xxxx"
        if folder_name.startswith(f"{model_ws_name}_it_"):
            folder_path = os.path.join(iterations_output_dir, folder_name)
            mf_path = os.path.join(folder_path, model_ws_name, "output")

            # Only proceed if the unique model workspace subfolder exists
            if os.path.exists(mf_path):
                # Extract the "parameter_xxxx" part from the folder name
                code = folder_name.split(f"{model_ws_name}_")[1]

                # Define the source files
                budget_file = os.path.join(mf_path, budget_file_name)
                zonebud_file = os.path.join(mf_path, zonebud_file_name)
                head_obs_file = os.path.join(mf_path, head_file_name)
                cbb_summary_file = os.path.join(mf_path, cbb_summary_file_name)

                # Define the destination files
                budget_dest = os.path.join(results_folder, f"{os.path.splitext(budget_file_name)[0]}_{code}.csv")
                zonebud_dest = os.path.join(results_folder, f"{os.path.splitext(zonebud_file_name)[0]}_{code}.csv")
                head_obs_dest = os.path.join(results_folder, f"{os.path.splitext(head_file_name)[0]}_{code}.csv")
                cbb_summary_dest = os.path.join(results_folder, f"{os.path.splitext(cbb_summary_file_name)[0]}_{code}.csv")

                # Copy files if they exist
                if os.path.exists(budget_file):
                    shutil.copy(budget_file, budget_dest)
                    print(f"Copied {budget_file} to {budget_dest}")

                if os.path.exists(zonebud_file):
                    shutil.copy(zonebud_file, zonebud_dest)
                    print(f"Copied {zonebud_file} to {zonebud_dest}")

                if os.path.exists(head_obs_file):
                    shutil.copy(head_obs_file, head_obs_dest)
                    print(f"Copied {head_obs_file} to {head_obs_dest}")
                
                if os.path.exists(cbb_summary_file):
                    shutil.copy(cbb_summary_file, cbb_summary_dest)
                    print(f"Copied {cbb_summary_file} to {cbb_summary_dest}")

def estimate_sustainable_yield(
    input_folder: str,
    output_folder: str,
    plot_folder: str,
    pump_start: float,
    pump_zone: str,
    planning_horizon: float,
    constraints: list,
    model_name: str,
    csv_filename: str = "flow_summary.csv",
    plot_filename: str = "Q_vs_flow.png",
    plot_units: str = None,
    conversion_factor: float = 1.0
):
    """
    Estimates the sustainable yield for a groundwater system using MODFLOW 6 zonebud simulation results.
    Using a constrained maximization approach where user defined constrains on different flow components
    can be defined (river outflow, spring outflow, leakage, etc).

    Args:
        input_folder (str): Path to the folder containing zonebudget CSV output files.
        output_folder (str): Path to the folder where results will be saved.
        plot_folder (str): Path to the folder where plots will be saved.
        pump_start (float): Time (in model total simulation time units) when pumping starts.
        pump_zone (str): Zone ID or "ALL" for the pumping well(s) to analyze.
        planning_horizon (float): Time (in model time units) after pump_start to evaluate constraints.
        constraints (list): List of constraint dictionaries. Each dictionary should contain:
            - "label" (str): Descriptive label for the constraint. Used for plotting.
            - "id" (str): Unique identifier for the constraint.
            - "constrain" (str): Component acting as constrain ("LEAKAGE", "DRN", "RIV", "GHB", etc).
            - "flow" (str): Flow type ("IN", "OUT", "NET", "CBB"). If "NET" the net outflow is considered (OUT - IN).
                If "CBB", constraints from the cell budget file analysis are considered.
            Positive values indicate outflow from the system, negative values indicate inflow to the system.
            - "zone" (str): Zone ID or "ALL" for the constraint.
            - "threshold_type" (str): Type of threshold ("ABSOLUTE" or "RELATIVE").
            - "threshold" (float): Threshold value for the constraint.
            If absolute, value in model units minding its sign if "flow" is set to "NET": Positive for net outflow, negative for net inflow.
            If relative, as a fraction of a given reference (e.g. < 1, 0.1 for 10% if the variable is decreasing)
             or > 1, 1.5 for 150% if the variable is increasing).
            - "reference" (float, optional): Reference value for relative thresholds. If None, the flow data from
            the first pumping rate amongs the iterations will be used as reference. Normally the first pumping rate is 0 (natural conditions).
            Otherwise, the reference value must be provided. Just used when threshold_type is "RELATIVE".
            - "neighbour_zones" (list, optional): List of neighboring zones for leakage constraints. Just used when
            "constrain" is "LEAKAGE".
            - "color" (str, optional): Color for plotting the constraint.
        csv_filename (str): Name of the output CSV file summarizing results.
        plot_filename (str): Name of the output plot file.
        plot_units (str, optional): "years" or None. If "years", it assumes model time units are days and converts to years for plotting.
        conversion_factor (float): Factor to convert time units to years, used for plotting.
        If None, does not customize model time units.

    Returns:
        qs_value (float): Estimated sustainable yield value for the specified system.
        df (DataFrame): DataFrame containing pumping rates and corresponding flow values for each constraint.
        most_prohibitive_constraints (list): List of dicts for the binding constraint(s), each containing:
            - "id" (str): Constraint identifier.
            - "type" (str): "ABSOLUTE" or "RELATIVE".
            - "value" (float): The actual threshold value that was crossed (i.e. the limit).
            - "reference" (float): Reference value used. For RELATIVE, the explicitly provided reference
              or vals[0] if not provided. For ABSOLUTE, always vals[0] (natural conditions).
            - "impact" (float): reference - value, representing the total allowed change from
              natural conditions to the threshold.

    Extended version:
      • Supports FLOW, HEAD, and LEAKAGE constraints.
      • Automatically uses aggregated model budget file (<model_name>_budget_it_qv_X.csv)
        instead of zonebudget file whenever:
          - pump_zone == "ALL"
          - OR a flow constraint (not LEAKAGE) has zone == "ALL".
      • The budget file's time column is automatically renamed from "time" → "totim"
        to keep consistent time handling throughout.

    Naming conventions:
      Zonebudget file columns:  <pname>-<ptype>-IN/OUT  or  <ptype>-IN/OUT
      Budget file columns:      <ptype>(<pname>)_IN/OUT
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import Optional

    output_folder = os.path.abspath(output_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    # --- helper: find where constraint crosses threshold ---
    def find_threshold_crossing(x, y, threshold=0):
        for i in range(len(y) - 1):
            if pd.isna(y[i]) or pd.isna(y[i + 1]):
                continue
            if (y[i] - threshold) * (y[i + 1] - threshold) < 0:
                x0, x1 = x[i], x[i + 1]
                y0, y1 = y[i] - threshold, y[i + 1] - threshold
                return x0 - y0 * (x1 - x0) / (y1 - y0)
        return None

    # --- helper: paired head_obs file ---
    def paired_head_file(zonebud_filename: str) -> Optional[str]:
        base, ext = os.path.splitext(zonebud_filename)
        if not base.startswith("zonebud_it_qv_"):
            return None
        suffix = base[len("zonebud_"):]  # e.g. "it_qv_0"
        candidate = f"head_obs_t_{suffix}{ext}"
        path = os.path.join(input_folder, candidate)
        return path if os.path.isfile(path) else None

    # --- helper: paired aggregated budget file ---
    def paired_budget_file(zonebud_filename: str) -> Optional[str]:
        base, ext = os.path.splitext(zonebud_filename)
        if not base.startswith("zonebud_it_qv_"):
            return None
        suffix = base[len("zonebud_"):]  # e.g. "it_qv_0"
        candidate = f"{model_name}_budget_{suffix}{ext}"
        path = os.path.join(input_folder, candidate)
        return path if os.path.isfile(path) else None

    # --- helper: paired cbb summary file ---
    def paired_cbb_summary_file(zonebud_filename: str) -> Optional[str]:
        base, ext = os.path.splitext(zonebud_filename)
        if not base.startswith("zonebud_it_qv_"):
            return None
        suffix = base[len("zonebud_"):]  # e.g. "it_qv_0"
        candidate = f"cbb_summary_{suffix}{ext}"
        path = os.path.join(input_folder, candidate)
        return path if os.path.isfile(path) else None

    # --- compute target time for constraint evaluation ---
    time_target = pump_start + planning_horizon

    # --- result collector ---
    results = {c["id"]: [] for c in constraints}

    # --- loop through all zonebudget scenarios ---
    for file_name in os.listdir(input_folder):
        if not (file_name.startswith("zonebud") and file_name.endswith(".csv")):
            continue

        file_path = os.path.join(input_folder, file_name)
        try:
            data = pd.read_csv(file_path)
        except Exception:
            continue

        if "totim" not in data.columns or "WEL-OUT" not in data.columns:
            continue

        # --- try to pair related files ---
        head_path = paired_head_file(file_name)
        budget_path = paired_budget_file(file_name)

        heads_df, budget_df = None, None

        # --- load head file ---
        if head_path and os.path.isfile(head_path):
            try:
                tmp = pd.read_csv(head_path)
                for alt in ["time", "Time", "TIME"]:
                    if "totim" not in tmp.columns and alt in tmp.columns:
                        tmp = tmp.rename(columns={alt: "totim"})
                if "totim" in tmp.columns:
                    heads_df = tmp
            except Exception:
                pass

        # --- load aggregated budget file ---
        if budget_path and os.path.isfile(budget_path):
            try:
                budget_df = pd.read_csv(budget_path)
                # rename "time" -> "totim" for consistency
                if "time" in budget_df.columns and "totim" not in budget_df.columns:
                    budget_df = budget_df.rename(columns={"time": "totim"})
            except Exception:
                pass

        # --- load cbb summary file (for CBB constraints) ---
        cbb_summary_path = paired_cbb_summary_file(file_name)
        cbb_summary_df = None
        if cbb_summary_path and os.path.isfile(cbb_summary_path):
            try:
                cbb_summary_df = pd.read_csv(cbb_summary_path)
                for alt in ["time", "Time", "TIME"]:
                    if "totim" not in cbb_summary_df.columns and alt in cbb_summary_df.columns:
                        cbb_summary_df = cbb_summary_df.rename(columns={alt: "totim"})
            except Exception:
                pass

        # --- filter post-pumping period ---
        pump_data = data[data["totim"] >= pump_start].copy()
        if pump_data.empty:
            continue

        # --- compute pumping rate ---
        if pump_zone == "ALL" and budget_df is not None:
            if "WEL(WEL)_OUT" in budget_df.columns:
                pump_series = budget_df.loc[budget_df["totim"] >= pump_start, "WEL(WEL)_OUT"]
            else:
                pump_series = pump_data.groupby("totim")["WEL-OUT"].sum()
        elif pump_zone == "ALL":
            if "zone" in pump_data.columns:
                pump_series = pump_data.groupby("totim")["WEL-OUT"].sum()
            else:
                pump_series = pump_data["WEL-OUT"]
        else:
            if "zone" in pump_data.columns:
                pump_series = pump_data.loc[pump_data["zone"] == pump_zone, "WEL-OUT"]
            else:
                continue

        if pump_series.empty:
            continue

        Q_code = float(pd.to_numeric(pump_series, errors="coerce").mean())

        # --- evaluate all constraints for this scenario ---
        for c in constraints:
            cid = c["id"]
            constr = c["constrain"].upper()
            zone = c["zone"]
            flow = c["flow"].upper()
            value = None

            # ---------- HEAD constraint ----------
            if constr == "HEAD":
                head_col = c.get("head_obs")
                if heads_df is None or head_col not in heads_df.columns:
                    continue
                closest_idx = (heads_df["totim"] - time_target).abs().idxmin()
                value = heads_df.loc[closest_idx, head_col]
                if pd.notna(value):
                    results[cid].append((Q_code, float(value)))
                continue

            # ---------- CBB constraint (reads from cbb_summary file) ----------
            if flow == "CBB":
                if cbb_summary_df is None:
                    continue
                pkg_col = c["constrain"]   # raw MODFLOW name e.g. "DRN", "GHB"
                if pkg_col not in cbb_summary_df.columns or "totim" not in cbb_summary_df.columns:
                    continue
                closest_idx = (cbb_summary_df["totim"] - time_target).abs().idxmin()
                value = cbb_summary_df.loc[closest_idx, pkg_col]
                if pd.notna(value):
                    results[cid].append((Q_code, float(value)))
                continue

            # ---------- LEAKAGE constraint ----------
            if constr == "LEAKAGE":
                subset = data if zone == "ALL" else data[data["zone"] == zone]
                nz = c.get("neighbour_zones") or []
                required_cols = set([f"TO ZONE {z}" for z in nz] + [f"FROM ZONE {z}" for z in nz] + ["totim"])
                if not required_cols.issubset(subset.columns):
                    continue
                closest_idx = (subset["totim"] - time_target).abs().idxmin()
                row = subset.loc[closest_idx]
                to_sum = float(sum(row.get(f"TO ZONE {z}", 0.0) for z in nz))
                from_sum = float(sum(row.get(f"FROM ZONE {z}", 0.0) for z in nz))
                if flow == "NET":
                    value = to_sum - from_sum
                elif flow == "OUT":
                    value = to_sum
                elif flow == "IN":
                    value = from_sum

            # ---------- FLOW with zone == "ALL" (use aggregated budget) ----------
            elif zone == "ALL" and budget_df is not None:
                out_col, in_col = f"{constr}_OUT", f"{constr}_IN"
                if "totim" not in budget_df.columns:
                    continue
                closest_idx = (budget_df["totim"] - time_target).abs().idxmin()
                row = budget_df.loc[closest_idx]
                if flow == "OUT" and out_col in budget_df.columns:
                    value = float(row[out_col])
                elif flow == "IN" and in_col in budget_df.columns:
                    value = float(row[in_col])
                elif flow == "NET" and {out_col, in_col}.issubset(budget_df.columns):
                    value = float(row[out_col] - row[in_col])

            # ---------- FLOW per-zone (use zonebudget) ----------
            else:
                subset = data if zone == "ALL" else data[data["zone"] == zone]
                out_col, in_col = f"{constr}-OUT", f"{constr}-IN"
                if "totim" not in subset.columns:
                    continue
                closest_idx = (subset["totim"] - time_target).abs().idxmin()
                if zone == "ALL":
                    closest_time = subset.loc[closest_idx, "totim"]
                    time_filtered = subset[subset["totim"] == closest_time]
                    if flow == "OUT" and out_col in subset.columns:
                        value = float(time_filtered[out_col].sum())
                    elif flow == "IN" and in_col in subset.columns:
                        value = float(time_filtered[in_col].sum())
                    elif flow == "NET" and {out_col, in_col}.issubset(subset.columns):
                        value = float(time_filtered[out_col].sum() - time_filtered[in_col].sum())
                else:
                    row = subset.loc[closest_idx]
                    if flow == "OUT" and out_col in subset.columns:
                        value = float(row[out_col])
                    elif flow == "IN" and in_col in subset.columns:
                        value = float(row[in_col])
                    elif flow == "NET" and {out_col, in_col}.issubset(subset.columns):
                        value = float(row[out_col] - row[in_col])

            # --- store the value ---
            if value is not None and pd.notna(value):
                results[cid].append((Q_code, float(value)))

    # --- assemble DataFrame with all results ---
    for cid in results:
        results[cid].sort(key=lambda x: x[0])
    pumping_rates = sorted({x[0] for vals in results.values() for x in vals})
    df = pd.DataFrame({"PumpingRate": pumping_rates})
    for c in constraints:
        cid = c["id"]
        temp = pd.DataFrame(results.get(cid, []), columns=["PumpingRate", cid])
        df = pd.merge(df, temp, on="PumpingRate", how="left")

    # --- find threshold crossings ---
    thresholds, crossings = {}, {}
    constraint_info = {}  # stores detailed info per constraint

    for c in constraints:
        cid = c["id"]
        series = pd.to_numeric(df[cid], errors="coerce")
        vals = series.dropna().values
        if len(vals) == 0:
            continue

        # default reference for both ABSOLUTE and RELATIVE is vals[0] (natural conditions)
        ref = float(vals[0])

        if c["threshold_type"].upper() == "ABSOLUTE":
            thresholds[cid] = float(c["threshold"])
            # reference is always vals[0] for ABSOLUTE

        elif c["threshold_type"].upper() == "RELATIVE":
            # override ref if explicitly provided
            if c.get("reference") is not None:
                ref = float(c["reference"])
            thresholds[cid] = ref * float(c["threshold"])

        crossings[cid] = find_threshold_crossing(df["PumpingRate"].values, series.values, thresholds[cid])

        constraint_info[cid] = {
            "type": c["threshold_type"].upper(),
            "value": thresholds[cid],
            "reference": ref,
            "impact": ref - thresholds[cid],
        }

    # --- determine sustainable yield (min of all valid crossings) ---
    qs_candidates = {k: v for k, v in crossings.items() if v is not None}

    if qs_candidates:
        qs_value = min(qs_candidates.values())
        most_prohibitive_constraints = [
            {"id": k, **constraint_info[k]}
            for k, v in qs_candidates.items() if v == qs_value
        ]
    else:
        qs_value = None
        most_prohibitive_constraints = []

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(14, 12))
    ax2 = ax.twinx()
    flow_handles, head_handles = [], []

    def is_head(c): return c["constrain"].upper() == "HEAD"

    for c in constraints:
        cid = c["id"]
        color = c.get("color")
        if is_head(c):
            ln, = ax2.plot(df["PumpingRate"], df[cid], marker="o", label=c["label"], color=color)
            head_handles.append(ln)
            if cid in thresholds:
                ax2.axhline(thresholds[cid], linestyle="--", color=ln.get_color(), label=f"{c['label']} threshold")
        else:
            ln, = ax.plot(df["PumpingRate"], df[cid], marker="o", label=c["label"], color=color)
            flow_handles.append(ln)
            if cid in thresholds:
                ax.axhline(thresholds[cid], linestyle="--", color=ln.get_color(), label=f"{c['label']} threshold")

    if qs_value is not None:
        ax.axvline(qs_value, color="g", linestyle="--")
        try:
            y_top = max([h.get_ydata().max() for h in flow_handles]) if flow_handles else ax.get_ybound()[1]
        except Exception:
            y_top = ax.get_ybound()[1]
        ax.text(qs_value, y_top, f"Qs < {qs_value:.2f}", color="g", fontsize=14, ha="right",
                bbox=dict(facecolor="white", alpha=0.7))

    title_suffix = f"{int(planning_horizon / conversion_factor)} {plot_units}" if plot_units == "years" else f"{planning_horizon} time units"
    ax.set_title(f"Sustainable yield estimation - {title_suffix} after pumping")
    ax.set_xlabel("Pumping Rate")
    ax.set_ylabel("Flow Rate")
    ax2.set_ylabel("Head")
    ax.set_xlim(left=0, right=min(df["PumpingRate"].max() * 1.1, qs_value * 2 if qs_value is not None else df["PumpingRate"].max() * 1.1))
    # ax.set_yscale('log')

    # combined legend (flows + heads + thresholds)
    handles, labels = [], []
    for h in flow_handles + head_handles:
        handles.append(h)
        labels.append(h.get_label())
    handles_thr, labels_thr = ax.get_legend_handles_labels()
    handles_thr2, labels_thr2 = ax2.get_legend_handles_labels()
    threshold_pairs = [(h, l) for h, l in (list(zip(handles_thr, labels_thr)) + list(zip(handles_thr2, labels_thr2))) if "threshold" in l.lower()]
    handles += [h for h, _ in threshold_pairs]
    labels += [l for _, l in threshold_pairs]

    if handles:
        ax.legend(handles, labels, loc="best")

    ax.grid(True)

    plt.savefig(os.path.join(plot_folder, plot_filename), bbox_inches="tight")
    plt.close()
    df.to_csv(os.path.join(output_folder, csv_filename), index=False)

    return qs_value, df, most_prohibitive_constraints

def update_well_ts_file(base_ts_path, setup_file, q_column, output_path):
    """
    Update a Flopy-style .ts file with new pumping rates from a given q_column in Excel.

    Parameters
    ----------
    base_ts_path : str or Path
        Path to the existing .ts file (template).
    setup_file : str or Path
        Path to the Excel file containing 'well_id', 'time', and q_column.
    q_column : str
        Name of the column in Excel to use for pumping rates (e.g., 'qv_01').
    output_path : str or Path
        Path to save the new .ts file.
    """
    # Read Excel data
    q_df = pd.read_excel(setup_file, sheet_name="q_values_tr")

    # Read the base .ts file
    with open(base_ts_path, "r") as f:
        lines = f.readlines()

    # Locate timeseries block
    start_idx = next(i for i, l in enumerate(lines) if "BEGIN timeseries" in l)
    end_idx = next(i for i, l in enumerate(lines) if "END timeseries" in l)

    # Extract well names (order matters)
    name_line = next(l for l in lines if l.strip().startswith("NAMES"))
    well_names = name_line.strip().split()[1:]  # skip 'NAMES'

    # Extract times from the existing file
    ts_lines = lines[start_idx + 1:end_idx]
    times = [float(re.split(r"\s+", l.strip())[0]) for l in ts_lines]

    # Initialize last known q for each well
    last_q = {well: 0.0 for well in well_names}

    # Build new timeseries lines
    new_ts_lines = []
    for t in times:
        row = [f"{t:.8E}"]
        for well in well_names:
            match = q_df[(q_df["well_id"] == well) & (q_df["time"] == t)]
            if not match.empty:
                q_val = float(match[q_column].iloc[0])
                last_q[well] = q_val
            else:
                q_val = last_q[well]
            row.append(f"{q_val:14.8f}")
        new_ts_lines.append(" ".join(row) + "\n")

    # Replace timeseries block
    new_lines = lines[:start_idx+1] + new_ts_lines + lines[end_idx:]

    # Write new file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(new_lines)

    print(f"Updated .ts file saved at: {output_path}")

def update_oc_file(base_oc_path, new_output_prefix, output_path):
    """
    Update output folder paths in a MODFLOW 6 .oc file.

    Parameters
    ----------
    base_oc_path : str or Path
        Path to the existing .oc file.
    new_output_prefix : str
        New folder or prefix to replace the old path before filenames, e.g. "output/scenario1/".
    output_path : str or Path
        Path to save the updated .oc file.
    """
    # Read the OC file
    with open(base_oc_path, "r") as f:
        lines = f.readlines()

    # Ensure the prefix ends with '/'
    if not new_output_prefix.endswith("/"):
        new_output_prefix += "/"

    # Get pattern: captures lines like "FILEOUT  output/DEESACt.hds"
    pattern = re.compile(r"(FILEOUT\s+)([\w./\\-]+)")

    new_lines = []
    for line in lines:
        match = pattern.search(line)
        if match:
            old_path = match.group(2)
            filename = Path(old_path).name  # get "DEESACt.hds"
            new_line = pattern.sub(rf"\1{new_output_prefix}{filename}", line)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Write new file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(new_lines)

    print(f"Updated .oc file saved at: {output_path}")

def update_obs_file(base_obs_path, new_output_prefix, output_path):
    """
    Update output folder paths in a MODFLOW 6 .obs file.

    Parameters
    ----------
    base_obs_path : str or Path
        Path to the existing .obs file.
    new_output_prefix : str
        New folder or prefix to replace the old path before filenames,
        e.g. "output/scenario1/".
    output_path : str or Path
        Path to save the updated .obs file.
    """
    # Read the OBS file
    with open(base_obs_path, "r") as f:
        lines = f.readlines()

    # Ensure the prefix ends with '/'
    if not new_output_prefix.endswith("/"):
        new_output_prefix += "/"

    # Regex pattern to match lines with FILEOUT <path>
    # e.g., "FILEOUT  output/head_obs_t.csv"
    pattern = re.compile(r"(FILEOUT\s+)([\w./\\-]+)")

    new_lines = []
    for line in lines:
        match = pattern.search(line)
        if match:
            old_path = match.group(2)
            filename = Path(old_path).name  # Extract "head_obs_t.csv"
            # Replace with new prefix
            new_line = pattern.sub(rf"\1{new_output_prefix}{filename}", line)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Write updated file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(new_lines)

    print(f"Updated .obs file saved at: {output_path}")

def iterate_pumping_rate_transient_eff(setup_file, model_file, model_ws_name, model_name, 
                                        iterations_output_dir, summary_dir, 
                                        budget_file_name, zonebud_file_name, head_file_name, cbb_summary_file_name):
    """
    Function to iterate a groundwater model over different pumping rates defined in an Excel setup file
    (on sheet named q_values_tr). For each pumping rate, the model is run in a unique workspace and the 
    relevant output files are copied to a summary directory.
    
    This version manages memory more efficiently by calling model files from a base workspace of the first 
    model iteration. Subsequent runs do not write model input files again, just update the well and output
    control files. It does not postprocess on the go, useful for large model files.

    Args:
        setup_file (str): Path to the Excel setup file containing well and pumping rate information.
        model_file (str): Path to the transient groundwater model Python script.
        model_ws_base (str): Path to the base model workspace directory to write input model files.
        model_name (str): Name of the groundwater model.
        iterations_output_dir (str): Directory where output files from model iterations will be saved.
        summary_dir (str): Directory where summary of results will be saved.
        budget_file_name (str): Name of the budget output file generated by the model.
        zonebud_file_name (str): Name of the zone budget output file generated by the model.
        head_file_name (str): Name of the head observation output file generated by the model.
        cbb_summary_file_name (str): Name of the cell-by-cell budget summary output file generated by the model.
    """
    os.makedirs(model_ws_name, exist_ok=True)
    os.makedirs(iterations_output_dir, exist_ok=True)

    # --------------------------------------------------------------------------------------- #
    # ------------------------------- WRITE MODEL INPUT FILES ------------------------------- #
    # --------------------------------------------------------------------------------------- #
    # Run the original script inside the base model workspace folder to write input files
    # You can use subprocess to execute the script in that directory
    subprocess.run(["python", model_file])
    print(f"Base model input files written in model working directory")
    # Define zone array for zonal budget calculations
    zone_array = np.load(f"{model_ws_name}/zone_array.npy")

    # --------------------------------------------------------------------------------------- #
    # ------------------------------- UPDATE WELL MODEL FILES ------------------------------- #
    # --------------------------------------------------------------------------------------- #
    # q-values for each iteration
    q_df = pd.read_excel(setup_file, sheet_name="q_values_tr")
    # Identify iteration columns (all except well_id + time)
    iter_cols = [c for c in q_df.columns if c not in ["well_id", "time", "comment"]]
    n_iterations = len(iter_cols)

    for i, col in enumerate(iter_cols, start=1):

        print(f"\n--- Running iteration {i}/{n_iterations} with {col} ---")
        os.makedirs(f"{iterations_output_dir}/{col}", exist_ok=True)

        update_well_ts_file(f"{model_ws_name}/well_rates.ts", f"{setup_file}", f"{col}", f"{model_ws_name}/well_rates.ts")
        update_oc_file(f"{model_ws_name}/{model_name}.oc", f"../sust_yield_results/yield_iterations/{col}/", f"{model_ws_name}/{model_name}.oc")
        update_obs_file(f"{model_ws_name}/{model_name}.obs", f"../sust_yield_results/yield_iterations/{col}/", f"{model_ws_name}/{model_name}.obs")
        # Everything in the oc file is relative to the model_ws, so "../" goes to the parent folder
        sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws_name, exe_name="mf6")
        subprocess.run(["mf6"], cwd=model_ws_name)

        gwf = sim.gwf[0]

        zonebud = gwf.output.zonebudget(zone_array)
        zonebud.change_model_ws(f"{iterations_output_dir}/{col}")
        zonebud.write_input()
        zonebud.run_model()
        del zonebud

        cbb_path = f"{iterations_output_dir}/{col}/{model_name}.cbb" 
        csv_out  = f"{iterations_output_dir}/{col}/cbb_summary.csv"
        analyze_cbb_boundaries(
            cbb_path   = cbb_path,
            csv_out    = csv_out,
            delete_cbb = True,)

        print(f"Model run completed for iteration {i} with q={col}")

    # --------------------------------------------------------------------------------------- #
    # ------------------------------- MANAGE OUTPUT FILES ----------------------------------- #
    # --------------------------------------------------------------------------------------- #

    # Define a destination directory to summarize results of the iterations
    results_folder = summary_dir
    os.makedirs(results_folder, exist_ok=True)

    # Loop through the sub-folders in the output directory to get relevant files
    for folder_name in os.listdir(iterations_output_dir):
            folder_path = os.path.join(iterations_output_dir, folder_name)

            # Only proceed if the unique model workspace subfolder exists
            if os.path.exists(folder_path):
                # Extract the "parameter_xxxx" part from the folder name
                code = folder_name

                # Define the source files
                budget_file = os.path.join(folder_path, budget_file_name)
                zonebud_file = os.path.join(folder_path, zonebud_file_name)
                head_obs_file = os.path.join(folder_path, head_file_name)
                cbb_summary_file = os.path.join(folder_path, cbb_summary_file_name)

                # Define the destination files
                budget_dest = os.path.join(results_folder, f"{os.path.splitext(budget_file_name)[0]}_it_{code}.csv")
                zonebud_dest = os.path.join(results_folder, f"{os.path.splitext(zonebud_file_name)[0]}_it_{code}.csv")
                head_obs_dest = os.path.join(results_folder, f"{os.path.splitext(head_file_name)[0]}_it_{code}.csv")
                cbb_summary_dest = os.path.join(results_folder, f"{os.path.splitext(cbb_summary_file_name)[0]}_it_{code}.csv")

                # Copy files if they exist
                if os.path.exists(budget_file):
                    shutil.copy(budget_file, budget_dest)
                    print(f"Copied {budget_file} to {budget_dest}")

                if os.path.exists(zonebud_file):
                    shutil.copy(zonebud_file, zonebud_dest)
                    print(f"Copied {zonebud_file} to {zonebud_dest}")

                if os.path.exists(head_obs_file):
                    shutil.copy(head_obs_file, head_obs_dest)
                    print(f"Copied {head_obs_file} to {head_obs_dest}")

                if os.path.exists(cbb_summary_file):
                    shutil.copy(cbb_summary_file, cbb_summary_dest)
                    print(f"Copied {cbb_summary_file} to {cbb_summary_dest}")

def analyze_cbb_boundaries(
    cbb_path: str,
    csv_out: str,
    delete_cbb: bool = True,
    run_label: str = None,
) -> pd.DataFrame:
    """
    Read a MODFLOW 6 cell budget file and compute per-time-step SNAPSHOT
    statistics for all DRN, GHB, RIV and CHD packages found inside it.

    MULTIPLE PACKAGES OF THE SAME TYPE (e.g. drn1..drn5, ghb1, ghb2):
    MF6 writes these under a shared generic record "text" (e.g. "DRN"),
    so cb.get_unique_record_names() / a plain text-based read only ever
    sees "DRN" once, not five times. Per the MF6 IO documentation, list-
    based boundary flow records are written with IMETH=6, which includes
    four text identifiers: TXT1ID1, TXT2ID1, TXT1ID2, TXT2ID2. For GWF
    boundary packages, TXT1ID1 and TXT2ID1 are both just the GWF model
    name, and TXT2ID2 is "the package or model name" - i.e. the actual
    package instance name (e.g. "DRN1", "DRN2", ...).

    flopy exposes these as columns on `cb.headers`:
        modelnam  <- TXT1ID1 (model name)
        paknam    <- TXT2ID1 (ALSO model name - NOT the package name)
        modelnam2 <- TXT1ID2 (model name)
        paknam2   <- TXT2ID2 (the actual package instance name)

    So this function reads the package instance name from `paknam2`
    (not `paknam`) and uses it to label columns, giving one full set of
    stats per package instance (e.g. DRN1_*, ..., DRN5_*, GHB1_*, GHB2_*)
    instead of collapsing them into one generic <TEXT>_* block.

    For each package instance and each time step the following columns
    are written, where <PKG> is the resolved paknam2 (falling back to
    the record text if paknam2 is blank, e.g. for a model with only one
    instance of that package type):

    NET statistics  (all cells in the package)
    -------------------------------------------
    <PKG>_net_count     Total number of cells in the package (constant).
    <PKG>_net_total     Sum of all cell flows.
    <PKG>_net_mean      Mean flow per cell.
    <PKG>_net_median    Median flow per cell.
    <PKG>_net_std       Standard deviation of flows (population, ddof=0).
    <PKG>_net_5pct      5th percentile of flows.
    <PKG>_net_95pct     95th percentile of flows.
    <PKG>_net_min       Minimum flow (most negative).
    <PKG>_net_max       Maximum flow (least negative / most positive).

    OUTFLOW statistics  (cells with flow < 0, water leaving the aquifer)
    ---------------------------------------------------------------------
    <PKG>_out_count     Number of outflow cells.
    <PKG>_out_pct       out_count / net_count * 100.
    <PKG>_out_total     Sum of outflow cell flows (always <= 0).
    <PKG>_out_mean      Mean flow among outflow cells.
    <PKG>_out_median    Median flow among outflow cells.
    <PKG>_out_std       Standard deviation among outflow cells.
    <PKG>_out_5pct      5th percentile among outflow cells.
    <PKG>_out_95pct     95th percentile among outflow cells.
    <PKG>_out_min       Minimum (most negative) among outflow cells.
    <PKG>_out_max       Maximum (least negative) among outflow cells.

    INFLOW statistics  (cells with flow >= 0, water entering or inactive)
    ----------------------------------------------------------------------
    For DRN: includes dry/inactive cells (flow == 0).
    For GHB: includes cells that have reversed to inflow (flow >= 0).

    <PKG>_in_count      Number of inflow/inactive cells.
    <PKG>_in_pct        in_count / net_count * 100.
    <PKG>_in_total      Sum of inflow cell flows (always >= 0).
    <PKG>_in_mean       Mean flow among inflow cells.
    <PKG>_in_median     Median flow among inflow cells.
    <PKG>_in_std        Standard deviation among inflow cells.
    <PKG>_in_5pct       5th percentile among inflow cells.
    <PKG>_in_95pct      95th percentile among inflow cells.
    <PKG>_in_min        Minimum among inflow cells.
    <PKG>_in_max        Maximum among inflow cells.

    Note: <PKG>_net_pct is intentionally omitted (it is always 100% and
    carries no information), for every time step including ones where a
    package returned no data.

    Parameters
    ----------
    cbb_path : str
        Full path to the transient .cbb file.
    csv_out : str
        Full path where the output CSV will be written. Always overwritten.
    delete_cbb : bool
        If True, the .cbb file is removed after a successful read.
    run_label : str or None
        Optional string tag added as a "run" column. If None, omitted.

    Returns
    -------
    pd.DataFrame
        Table with one row per time step and all columns described above.
    """

    # ---------------------------------------------------------------------- #
    # 1. Open the cell budget file
    # ---------------------------------------------------------------------- #
    cb = flopy.utils.CellBudgetFile(cbb_path, precision="double")
    all_kstpkper = cb.get_kstpkper()
    times = cb.get_times()

    if len(times) != len(all_kstpkper):
        print(
            f"[analyze_cbb] WARNING - get_times() length ({len(times)}) does not "
            f"match get_kstpkper() length ({len(all_kstpkper)}); totim values may "
            f"be misaligned."
        )

    # ---------------------------------------------------------------------- #
    # 2. Discover DRN, GHB, RIV and CHD package INSTANCES via paknam2
    # ---------------------------------------------------------------------- #
    def _clean_str(val) -> str:
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        return str(val).replace("\x00", "").strip()

    hdr = cb.headers  # pandas DataFrame of every record header in the file

    hdr_text = hdr["text"].apply(_clean_str)
    if "paknam2" in hdr.columns:
        hdr_paknam2 = hdr["paknam2"].apply(_clean_str)
    else:
        # Very old flopy without paknam2: no way to split instances.
        print(
            "[analyze_cbb] WARNING - cb.headers has no 'paknam2' column "
            "(old flopy version?). Multi-instance packages of the same "
            "type cannot be distinguished; only the first instance's "
            "cells will be read per text. Consider upgrading flopy."
        )
        hdr_paknam2 = pd.Series([""] * len(hdr), index=hdr.index)

    combos = pd.DataFrame({"text": hdr_text, "paknam2": hdr_paknam2}).drop_duplicates()

    def _discover(prefix: str) -> list:
        """Return list of (text, paknam2, column_label) for a package prefix."""
        subset = combos[combos["text"].str.upper().str.startswith(prefix)]
        instances = []
        for _, r in subset.sort_values(["text", "paknam2"]).iterrows():
            txt, pak2 = r["text"], r["paknam2"]
            label = pak2 if pak2 else txt
            instances.append((txt, pak2, label))
        return instances

    package_groups = {
        "DRN": _discover("DRN"),
        "GHB": _discover("GHB"),
        "RIV": _discover("RIV"),
        "CHD": _discover("CHD"),
    }

    for prefix, instances in package_groups.items():
        labels = [lbl for _, _, lbl in instances]
        print(f"[analyze_cbb] {prefix} package instances found: {labels}")

    # ---------------------------------------------------------------------- #
    # 3. Helper: compute stats + count for flows[mask] under column prefix
    # ---------------------------------------------------------------------- #
    def flow_stats(flows: np.ndarray, mask: np.ndarray, prefix: str, n_cells: int) -> dict:
        subset = flows[mask]
        count = len(subset)
        pct = float(count / n_cells * 100) if n_cells > 0 else 0.0
        if count == 0:
            return {
                f"{prefix}_count":  0,
                f"{prefix}_pct":    pct,
                **{f"{prefix}_{s}": np.nan for s in
                   ["total", "mean", "median", "std", "5pct", "95pct", "min", "max"]},
            }
        return {
            f"{prefix}_count":  count,
            f"{prefix}_pct":    pct,
            f"{prefix}_total":  float(np.sum(subset)),
            f"{prefix}_mean":   float(np.mean(subset)),
            f"{prefix}_median": float(np.median(subset)),
            f"{prefix}_std":    float(np.std(subset)),
            f"{prefix}_5pct":   float(np.percentile(subset, 5)),
            f"{prefix}_95pct":  float(np.percentile(subset, 95)),
            f"{prefix}_min":    float(np.min(subset)),
            f"{prefix}_max":    float(np.max(subset)),
        }

    # ---------------------------------------------------------------------- #
    # 4. Helper: build all columns for one package instance at one time step
    # ---------------------------------------------------------------------- #
    def package_row(label: str, data: list) -> dict:
        if len(data) == 0:
            base = {f"{label}_net_count": 0}
            for s in ["total", "mean", "median", "std", "5pct", "95pct", "min", "max"]:
                base[f"{label}_net_{s}"] = np.nan
            for prefix in [f"{label}_out", f"{label}_in"]:
                base[f"{prefix}_count"] = 0
                base[f"{prefix}_pct"]   = np.nan
                for s in ["total", "mean", "median", "std", "5pct", "95pct", "min", "max"]:
                    base[f"{prefix}_{s}"] = np.nan
            return base

        if len(data) > 1:
            print(
                f"  [warn] {label} returned {len(data)} record arrays for this "
                f"time step even after filtering by paknam2; only the first "
                f"is used. Verify paknam2 values are unique per instance."
            )

        flows = _extract_flows(data[0])
        n_cells = len(flows)

        row = {f"{label}_net_count": n_cells}
        row.update(flow_stats(flows, np.ones(n_cells, dtype=bool), f"{label}_net", n_cells))
        row.update(flow_stats(flows, flows < 0.0,                   f"{label}_out", n_cells))
        row.update(flow_stats(flows, flows >= 0.0,                  f"{label}_in",  n_cells))

        row.pop(f"{label}_net_pct", None)  # net is always 100%, no info

        return row

    # ---------------------------------------------------------------------- #
    # 5. Loop over every time step
    # ---------------------------------------------------------------------- #
    rows = []

    for i, (kstp, kper) in enumerate(all_kstpkper):
        kstpkper = (kstp, kper)
        totim = times[i] if i < len(times) else np.nan

        row = {"run": run_label, "kper": kper, "kstp": kstp, "totim": totim}

        for prefix, instances in package_groups.items():
            for txt, pak2, label in instances:
                try:
                    data = cb.get_data(
                        text=txt,
                        kstpkper=kstpkper,
                        paknam2=pak2 if pak2 else None,
                    )
                    row.update(package_row(label, data))
                except Exception as e:
                    print(f"  [warn] Could not read {label} ({txt}) at {kstpkper}: {e}")
                    row.update(package_row(label, []))

        rows.append(row)

    # ---------------------------------------------------------------------- #
    # 6. Build DataFrame and write CSV
    # ---------------------------------------------------------------------- #
    df = pd.DataFrame(rows)

    if run_label is None:
        df.drop(columns=["run"], inplace=True)

    out_dir = os.path.dirname(csv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(csv_out, mode="w", header=True, index=False)
    print(f"[analyze_cbb] Results written to: {csv_out}")

    # ---------------------------------------------------------------------- #
    # 7. Delete the .cbb file
    # ---------------------------------------------------------------------- #
    cb.close()
    del cb
    gc.collect()

    if delete_cbb:
        for attempt in range(10):
            try:
                os.remove(cbb_path)
                print(f"[analyze_cbb] Deleted: {cbb_path}")
                break
            except OSError:
                time.sleep(0.5)
        else:
            print(f"[analyze_cbb] WARNING - could not delete {cbb_path} after 10 attempts")

    return df

def _extract_flows(record_array) -> np.ndarray:
    """
    Pull per-cell flow values from a CellBudgetFile record.
    MODFLOW 6 list-based packages use the field name 'q'.
    Falls back to the last numeric field if 'q' is absent.
    """
    names = record_array.dtype.names
    if "q" in names:
        return record_array["q"].astype(float)
    return record_array[names[-1]].astype(float)