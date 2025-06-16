#!/usr/bin/env python3
#=================================================================================================
# Sayan Adhikari | May 25, 2025 | https://github.com/sayan919
#=================================================================================================
"""
Generate Inputs for QM/MM systems for Gaussian or TeraChem

Flags:
    --single_xyz (Required) : Single frame XYZ file (or --traj_xyz).
    OR,
    --traj_xyz (Required) : Multi-frame trajectory XYZ file.

    --frames (Required with --traj_xyz) : Number of frames (default: all).
    
    --system_info (Required) : Single JSON defining all monomers and solvent properties.
    
    --aggregate (Required) : Number of core monomer units.
    
    --qm_aggregate_xyz (Optional) : Use coordinates from this file for the core aggregate, 
                                    overriding from the main input XYZ. Atom order must match.
    
    --qm_solvent (Optional) : Defines QM solvent shell by radius (Å) around core atoms.
    
    --mm_solvent (Optional) : Defines MM solvent.
                                - path provided: XYZ-like file (charge x y z per line, skips 2 headers).
                                - flag used alone: non-QM solvent with charges from system_info.json.
    
    --mm_monomer (Optional) : Defines MM charges for embedding from other monomers.
                                - '0': Embeds other monomers with zero charges at their atomic positions.
                                - File(s) provided: File(s) with "charge x y z" for MM charges.
    
    --eetg (Optional) : Flag to generate only EETG input for dimers (requires --aggregate=2).
    
    --output_com (Optional) : monomer/dimer/both; 
                              Generates Gaussian .com input files.
                              If specified, --gauss_keywords is required.
    OR,
    --output_xyz (Optional) : monomer/dimer/both/none; 
                              Generates detailed QM region and MM solvent .xyz files.

    --gauss_keywords (Conditionally Required) : Gaussian keywords (required if --output_com is used).
    
    --tag (Optional) : custom tag for generated .com filenames (e.g., ..._TAG.com).
    
    --logfile (Optional) : Specify log file name (default: run.log).

"""
# === Imports ===
import argparse
import os
import json
import numpy as np
import re
import sys
import datetime
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union

# Import utility functions and constants
import functions as fn

# === Main Function ===
def main() -> None:
    # Default log filename moved here for clarity, can be overridden by args
    current_default_log_filename = "giqpy_run.log"

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description="Generate Gaussian .com and .xyz files from single or trajectory XYZ with QM/MM options.",
        epilog="""
-------------------------------------------------------------------------------
Core Flags:
  --single_xyz <file.xyz> or --traj_xyz <file.xyz> :
                                  Input coordinate file(s). One of these is required.
  --frames <N>                    : Number of frames if --traj_xyz is used (required with --traj_xyz).
  --aggregate <M>               : Number of core monomer units (e.g., 2 for a dimer). Required.
  --system_info <file.json>     : Single JSON defining all monomers and solvent properties. Required.
  
Output Control (At least one of --output_com or --output_xyz must be specified):
  --output_com [monomer|dimer|both] : Generate Gaussian .com input files.
                                    If flag is present without a value, defaults to "both".
                                    Requires --gauss_keywords.
  --output_xyz [monomer|dimer|both] : Generate detailed .xyz files for QM regions and MM solvent.
                                    If flag is present without a value, defaults to "both".
  --gauss_keywords <file.txt>   : Text file with Gaussian route section keywords.
                                    Required if --output_com is specified.
QM/MM Setup:
  --qm_aggregate_xyz <file.xyz> : Optional: Use coordinates from this file for the core aggregate,
                                  overriding those from the main input XYZ. Atom order must match.
  --qm_solvent <radius>         : Defines QM solvent shell by radius (Å) around core atoms (default: 5.0).
  --mm_solvent [file.xyz]       : Optional. Defines MM solvent.
                                  - If path provided: XYZ-like file (charge x y z per line, skips 2 headers).
                                  - If flag used alone (i.e., --mm_solvent with no file path):
                                    Auto-detects non-QM solvent from input XYZ and uses charges from system_info.json.
  --mm_monomer [0|file1 ...]    : Optional. Defines MM charges for embedding from other monomers.
                                  - '0': Embeds other monomers with zero charges at their atomic positions.
                                  - File(s) provided: File(s) with "charge x y z" for MM charges.
                                    Provide one file per monomer in the aggregate if this mode is used.
  --eetg                        : Flag to generate only EETG input for dimers (requires --aggregate=2 and --output_com).
  
Other:
  --tag <TAG_STRING>            : Optional custom tag for generated .com filenames (e.g., ..._TAG.com).
  --logfile <filename>          : Specify log file name (default: giqpy_run.log).
-------------------------------------------------------------------------------
        """
    )
    group_xyz_input = parser.add_mutually_exclusive_group(required=True)
    group_xyz_input.add_argument('--single_xyz', type=str, help='Single-frame XYZ file for the entire system.')
    group_xyz_input.add_argument('--traj_xyz', type=str, help='Multi-frame XYZ trajectory for the entire system.')
    
    parser.add_argument('--frames', type=int,
                        help='Number of frames from trajectory (required with --traj_xyz).')
    parser.add_argument('--aggregate', type=int, required=True,
                        help='Number of core monomers (e.g., 1 for monomer, 2 for dimer).')
    parser.add_argument('--system_info', type=str, required=True,
                        help='JSON file with monomer and solvent metadata.')
    
    # --- Modified output arguments ---
    parser.add_argument('--output_com', nargs='?', choices=['monomer', 'dimer', 'both'], const='both', default=None,
                        help=('Generate Gaussian .com input files. '
                              'If flag is present without a value, defaults to "both". '
                              'Requires --gauss_keywords if specified.'))
    parser.add_argument('--output_xyz', nargs='?', choices=['monomer', 'dimer', 'both', 'none'], const='both', default=None, 
                        help=("Generate detailed .xyz files (QM regions, MM solvent). "
                              "If flag given without value, defaults to 'both'. "
                              "Choose 'none' to explicitly disable detailed XYZ output if --output_com is also given."))

    parser.add_argument('--gauss_keywords', type=str, default=None, 
                        help='File with Gaussian keywords (required if --output_com is specified).')
    # --- End of modified output arguments ---

    parser.add_argument('--qm_aggregate_xyz', type=str, default=None,
                        help='Optional: XYZ file for core aggregate coordinates, overriding main input for core.')
    parser.add_argument('--qm_solvent', type=float, default=5.0, # Ensure qm_solvent always has a default
                        help='Radius (Å) for QM solvent selection (default: 5.0). Negative value disables QM solvent.')
    parser.add_argument('--mm_monomer', type=str, nargs='*', # Can be '0' or list of files
                        help="MM embedding for other monomers: '0' for zero charges, or path(s) to charge file(s).")
    parser.add_argument('--mm_solvent', type=str, nargs='?', const=fn.AUTO_MM_SOLVENT_TRIGGER, default=None,
                        help='MM solvent: XYZ-like charge file, or flag alone for auto-detect from non-QM solvent and system_info.')
    parser.add_argument('--eetg', action='store_true',
                        help='Generate EETG input for dimer (requires --aggregate 2 and --output_com).')
    parser.add_argument('--tag', type=str, default="",
                        help='Optional custom tag for .com filenames (e.g., monomer1_qm_mm_TAG.com).')
    parser.add_argument('--logfile', type=str, default=current_default_log_filename, # Use variable for default
                        help=f'Specify log file name (default: {current_default_log_filename}).')
    
    args = parser.parse_args()

    # --- Setup Log File ---
    # Use the logfile name from args, which has a default.
    # Global LOG_FILENAME constant is not used directly here anymore for open().
    effective_log_filename = args.logfile
    try:
        fn.log_file_handle = open(effective_log_filename, 'w')
        fn.log_file_handle.write(f"GIQPy Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
        fn.log_file_handle.write(f"Arguments: {vars(args)}\n\n")
        fn.log_file_handle.flush()
    except IOError as e:
        print(f"CRITICAL ERROR: Could not open log file {effective_log_filename} for writing: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)


    # --- Argument Validation ---
    user_intends_com_output = args.output_com is not None
    user_intends_xyz_output = args.output_xyz is not None and args.output_xyz != 'none'

    if not user_intends_com_output and not user_intends_xyz_output:
        err_msg = "No output format selected. You must specify --output_com and/or --output_xyz."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        fn.write_to_log(err_msg, is_error=True)
        parser.error(err_msg)

    if args.traj_xyz and args.frames is None:
        err_msg = "--frames is required when --traj_xyz is used."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        fn.write_to_log(err_msg, is_error=True)
        parser.error(err_msg) 
    if args.aggregate < 1:
        err_msg = "--aggregate must be at least 1."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        fn.write_to_log(err_msg, is_error=True)
        parser.error(err_msg)
    if args.eetg:
        if args.aggregate != 2:
            err_msg = "--eetg calculations require --aggregate=2 (a dimer system)."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            fn.write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
        if not user_intends_com_output:
            err_msg = "--eetg calculations require .com file output. Please specify --output_com."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            fn.write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
            
    if user_intends_com_output and args.output_com == 'monomer' and args.eetg:
        warn_msg = ("Configuration: --output_com set to 'monomer' with --eetg. "
                    "EETG is an aggregate/dimer property. An EETG file for the dimer will be generated if --aggregate=2, "
                    "but individual monomer .com files will also be generated as per --output_com monomer.")
        print(f"INFO: {warn_msg}") # Changed to INFO as it's a clarification
        fn.write_to_log(warn_msg, is_warning=False) # Not strictly a warning of misconfiguration

    # Gaussian keywords handling
    gaussian_keywords_for_calcs: Optional[List[str]] = None # Variable to hold loaded keywords
    if user_intends_com_output:
        if not args.gauss_keywords:
            err_msg = "A Gaussian keywords file must be provided via --gauss_keywords when --output_com is specified."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            fn.write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
        
        if not os.path.exists(args.gauss_keywords) or os.path.getsize(args.gauss_keywords) == 0:
            err_msg = f"Gaussian keywords file '{args.gauss_keywords}' not found or is empty."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            fn.write_to_log(err_msg, is_error=True)
            # parser.error(err_msg) # This would exit before log flushes sometimes
            raise FileNotFoundError(err_msg) # More standard for file issues
        gaussian_keywords_for_calcs = fn.load_keywords_from_file(args.gauss_keywords)
    elif args.gauss_keywords: # Keywords provided, but .com output not requested
        warn_msg = ("--gauss_keywords was provided, but --output_com was not specified. "
                    "The keywords file will be ignored as no .com files are being generated.")
        print(f"WARNING: {warn_msg}")
        fn.write_to_log(warn_msg, is_warning=True)

    # --- End of Argument Validation ---

    # Load qm_aggregate_xyz if provided
    qm_aggregate_coords_override: Optional[CoordType] = None
    if args.qm_aggregate_xyz:
        fn.write_to_log(f"Loading user-defined aggregate coordinates from: {args.qm_aggregate_xyz}")
        try:
            with open(args.qm_aggregate_xyz, 'r') as f_agg_xyz:
                agg_lines = f_agg_xyz.readlines()
            if len(agg_lines) < 2: raise ValueError("QM aggregate XYZ file is too short.")
            num_atoms_in_agg_file = int(agg_lines[0].strip())
            if len(agg_lines) < num_atoms_in_agg_file + 2: raise ValueError("QM aggregate XYZ has fewer lines than atom count.")
            
            temp_agg_coords_list = []
            # Atom symbols from qm_aggregate_xyz are ignored; only coordinates are used.
            for ln_idx, ln_content in enumerate(agg_lines[2 : num_atoms_in_agg_file + 2]):
                parts = ln_content.split()
                if len(parts) >= 4: # Expect atom_symbol x y z
                    try: temp_agg_coords_list.append([float(p) for p in parts[1:4]])
                    except ValueError: raise ValueError(f"Malformed coordinate in {args.qm_aggregate_xyz} line {ln_idx+3}: '{ln_content.strip()}'")
                else: raise ValueError(f"Malformed line (expected atom_symbol x y z) in {args.qm_aggregate_xyz} line {ln_idx+3}: '{ln_content.strip()}'")
            qm_aggregate_coords_override = np.array(temp_agg_coords_list)
            fn.write_to_log(f"Successfully loaded {qm_aggregate_coords_override.shape[0]} aggregate coordinates from {args.qm_aggregate_xyz}.")
        except Exception as e: # Catch generic exceptions during file load/parse
            err_msg = f"Failed to load or parse --qm_aggregate_xyz file '{args.qm_aggregate_xyz}': {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            fn.write_to_log(err_msg, is_error=True)
            sys.exit(1) # Critical error, cannot proceed


    base_output_dir: str = os.getcwd() 

    processed_frames_list: List[Tuple[Optional[str], str, str]] = fn.split_frames(args.single_xyz, args.traj_xyz, args.frames, base_output_dir)
    total_frames_to_process = len(processed_frames_list)
    
    if total_frames_to_process == 0: # This implies an issue with split_frames or input files
        err_msg = f"No frames were found or could be processed from the input XYZ source."
        if args.traj_xyz:
             err_msg = f"No frames were found or could be processed from trajectory: {args.traj_xyz}"
        elif args.single_xyz:
             err_msg = f"Could not process single XYZ file: {args.single_xyz}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        fn.write_to_log(err_msg, is_error=True)
        sys.exit(1)


    monomers_metadata_list: List[Dict[str, Any]]
    solvent_metadata: Dict[str, Any]
    try:
        monomers_metadata_list, solvent_metadata = fn.load_system_info(args.system_info, args.aggregate)
    except Exception as e: # Catch errors from load_system_info (FileNotFound, ValueError)
        print(f"CRITICAL ERROR loading system_info: {e}", file=sys.stderr)
        fn.write_to_log(f"CRITICAL ERROR loading system_info: {e}", is_error=True)
        sys.exit(1)

    solvent_name_str = solvent_metadata.get(fn.JSON_KEY_NAME, "solvent")


    if len(monomers_metadata_list) != args.aggregate:
        # This check might be redundant if load_system_info already validates based on aggregate count
        msg = (f"Number of monomer entries in system_info ({len(monomers_metadata_list)}) "
               f"does not match --aggregate ({args.aggregate}).")
        print(f"ERROR: {msg}", file=sys.stderr)
        fn.write_to_log(msg, is_error=True)
        # raise ValueError(msg) # Or sys.exit(1)
        sys.exit(1)


    combined_system_term: str = "dimer" if args.aggregate == 2 else "aggregate"
    tag_for_filename = f"_{args.tag}" if args.tag else "" 


    if args.traj_xyz and total_frames_to_process > 0 :
        sys.stdout.write(f"\nProcessed Frames: 0 / {total_frames_to_process}\n") # Ensure this prints on a new line
        sys.stdout.flush()

    # --- Determine what types of files to generate based on user's explicit choices ---
    # .com file generation flags
    # If args.output_com is None (flag not given), these will be False.
    actual_gen_monomer_com: bool = (user_intends_com_output and args.output_com in ['monomer', 'both'])
    actual_gen_aggregate_com: bool = (user_intends_com_output and args.output_com in ['dimer', 'both'])

    # .xyz file generation flags
    # If args.output_xyz is None or 'none' (flag not given or explicitly disabled), these will be False.
    actual_gen_monomer_xyz: bool = (user_intends_xyz_output and args.output_xyz in ['monomer', 'both'])
    actual_gen_aggregate_xyz: bool = (user_intends_xyz_output and args.output_xyz in ['dimer', 'both'])


    for frame_idx, (frame_id_str, temp_xyz_input_filepath, current_frame_output_dir) in enumerate(processed_frames_list):
        # Progress indicator for trajectory processing
        if args.traj_xyz and total_frames_to_process > 0 :
            # For the first frame (idx 0), the "Processed Frames: 0 / N" is already printed.
            # For subsequent frames, update the line.
            if frame_idx > 0 :
                sys.stdout.write('\r' + ' ' * 70 + '\r') # Clear previous status line
            # Print current frame being processed, then on the next line, the progress update that will be overwritten
            # This ensures the "Processing Frame X / Y" message remains visible.
            
        # This message will be printed for each frame and stay.
        logged_output_dir_name = frame_id_str if frame_id_str else os.path.basename(current_frame_output_dir)
        frame_processing_message = f"--- Processing Frame {frame_idx + 1} / {total_frames_to_process} ({'ID: '+frame_id_str if frame_id_str else 'Single File'}) -> Output Dir: '{logged_output_dir_name}' ---"
        print(frame_processing_message) # This should be on its own line.
        fn.write_to_log(f"\n\n{frame_processing_message}")
        
        # The progress update that gets overwritten
        if args.traj_xyz and total_frames_to_process > 0:
            sys.stdout.write(f"Progress: Frame {frame_idx + 1}/{total_frames_to_process}")
            sys.stdout.flush()


        monomer_qm_regions: List[Tuple[AtomListType, CoordType]]
        aggregate_qm_region: Tuple[AtomListType, CoordType]
        non_qm_sol_groups: List[SolventGroupType]
        qm_solvent_flags: Dict[str, bool]
        core_coords_for_dist_calc: CoordType
        n_atoms_per_monomer_for_dist_calc: List[int]

        try:
            monomer_qm_regions, aggregate_qm_region, non_qm_sol_groups, qm_solvent_flags, \
            core_coords_for_dist_calc, n_atoms_per_monomer_for_dist_calc = \
                fn.localize_solvent_and_prepare_regions(
                    temp_xyz_input_filepath,
                    args.system_info,
                    args.aggregate,
                    args.qm_solvent, # qm_solvent radius
                    qm_aggregate_coords_override=qm_aggregate_coords_override 
                )
        except Exception as e:
            err_msg = f"Error during solvent localization/region preparation for frame {frame_id_str if frame_id_str else 'single'}: {e}"
            print(f"\nERROR: {err_msg}", file=sys.stderr) # Newline if progress bar was on previous line
            fn.write_to_log(err_msg, is_error=True)
            if args.traj_xyz:
                fn.write_to_log(f"Skipping frame {frame_id_str} due to error.", is_warning=True)
                # No need to rewrite progress here, loop will continue or end.
                continue 
            else:
                sys.exit(1) 


        monomer_names_from_meta: List[str] = [m.get(fn.JSON_KEY_NAME, f'm{idx+1}') for idx, m in enumerate(monomers_metadata_list)]
        base_system_name_for_title: str
        if len(set(monomer_names_from_meta)) == 1 and monomers_metadata_list : # check list not empty
            base_system_name_for_title = monomer_names_from_meta[0]
        else: 
            base_system_name_for_title = "_".join(monomer_names_from_meta) if monomer_names_from_meta else "system" # handle empty case
        
        distance_str_for_title: str = ""
        if args.aggregate >= 2:
            dist = fn.calculate_centroid_distance_between_first_two(core_coords_for_dist_calc, n_atoms_per_monomer_for_dist_calc)
            if dist is not None:
                distance_str_for_title = f"(m1-m2 centroid dist: {dist:.2f} A)" # Angstrom symbol
        
        aggregate_title_base: str = f"{base_system_name_for_title} {combined_system_term} {distance_str_for_title}".strip()
        # User requested comments not to be changed, these are the original ones for QM XYZ files
        aggregate_xyz_comment_base: str = f"{base_system_name_for_title} {combined_system_term} {distance_str_for_title}".strip()


        mm_solvent_point_charges_xyz_q: Optional[List[MMChargeTupleType]] = None # Format (x,y,z,q)
        system_has_mm_solvent: bool = False
        if args.mm_solvent == fn.AUTO_MM_SOLVENT_TRIGGER:
            fn.write_to_log("Attempting to auto-detect MM solvent from non-QM solvent molecules...") 
            if non_qm_sol_groups:
                try:
                    mm_solvent_point_charges_xyz_q = fn.assign_charges_to_solvent_molecules(non_qm_sol_groups, solvent_metadata)
                    if mm_solvent_point_charges_xyz_q:
                        fn.write_to_log(f"Generated {len(mm_solvent_point_charges_xyz_q)} MM solvent point charges (x,y,z,q) from auto-detection.") 
                        system_has_mm_solvent = True
                    else:
                        fn.write_to_log("Auto-detection found non-QM solvent groups, but no charges were assigned (check solvent definition or warnings).") 
                except ValueError as e:
                     fn.write_to_log(f"Error assigning charges to auto-detected MM solvent: {e}", is_error=True)
            else:
                fn.write_to_log("No non-QM solvent molecules found to treat as MM solvent via auto-detection.") 
        elif args.mm_solvent is not None: # Path to MM solvent file is provided
            fn.write_to_log(f"Loading MM solvent charges from file: {args.mm_solvent}") 
            loaded_mm_charges_from_file_chg_xyz: List[Tuple[float,float,float,float]] = [] # charge, x, y, z
            try:
                with open(args.mm_solvent, 'r') as f_mm_sol_file:
                    lines = f_mm_sol_file.readlines()
                    if len(lines) >=2: # Expect 2 header lines
                        for line_num, line_content in enumerate(lines[2:]): 
                            parts = line_content.strip().split()
                            if len(parts) == 4: # charge x y z
                                try:
                                    # File format is charge x y z
                                    charge_val = float(parts[0])
                                    x_val, y_val, z_val = float(parts[1]), float(parts[2]), float(parts[3])
                                    loaded_mm_charges_from_file_chg_xyz.append((charge_val, x_val, y_val, z_val))
                                except ValueError:
                                    warn_msg = f"Non-numeric value in MM solvent file {args.mm_solvent} on line {line_num + 3}: {line_content.strip()}"
                                    print(f"\nWARNING: {warn_msg}") # Ensure newline
                                    fn.write_to_log(warn_msg, is_warning=True)
                            elif parts: # Line not empty, but not 4 parts
                                warn_msg = f"Malformed MM solvent line in {args.mm_solvent} on line {line_num + 3} (expected 4 values, got {len(parts)}): {line_content.strip()}"
                                print(f"\nWARNING: {warn_msg}") # Ensure newline
                                fn.write_to_log(warn_msg, is_warning=True)
                    else:
                        warn_msg = f"MM solvent file {args.mm_solvent} is too short (expected XYZ-like format with 2 header lines). No charges loaded."
                        print(f"\nWARNING: {warn_msg}") # Ensure newline
                        fn.write_to_log(warn_msg, is_warning=True)

                if loaded_mm_charges_from_file_chg_xyz:
                    # Convert to (x,y,z,q) for internal consistency for Gaussian
                    mm_solvent_point_charges_xyz_q = [(x,y,z,q) for q,x,y,z in loaded_mm_charges_from_file_chg_xyz]
                    system_has_mm_solvent = True
                    fn.write_to_log(f"Loaded {len(mm_solvent_point_charges_xyz_q)} MM solvent point charges (x,y,z,q) from file.") 
                else:
                    # This warning might be redundant if the file was short/malformed and already warned.
                    if os.path.exists(args.mm_solvent): # Only warn if file exists but no charges loaded
                        warn_msg = f"No valid MM solvent charges loaded from file {args.mm_solvent}."
                        print(f"\nWARNING: {warn_msg}") # Ensure newline
                        fn.write_to_log(warn_msg, is_warning=True)
            except FileNotFoundError:
                warn_msg = f"MM solvent file {args.mm_solvent} not found. No MM solvent from file will be used."
                print(f"\nWARNING: {warn_msg}") # Ensure newline
                fn.write_to_log(warn_msg, is_warning=True)
        
        # MM Monomer Embedding Charges (this logic remains the same, result is used later)
        mm_embedding_charges_for_each_monomer: List[List[MMChargeTupleType]] = [[] for _ in range(args.aggregate)] 
        if args.mm_monomer is not None: 
            if args.mm_monomer == ['0']: 
                fn.write_to_log("Applying zero charges for other-monomer embedding as per --mm_monomer 0.")
                current_atom_idx = 0
                core_monomer_coords_list: List[CoordType] = []
                for num_atoms_in_mono in n_atoms_per_monomer_for_dist_calc: 
                    core_monomer_coords_list.append(
                        core_coords_for_dist_calc[current_atom_idx : current_atom_idx + num_atoms_in_mono]
                    )
                    current_atom_idx += num_atoms_in_mono
                
                if args.aggregate == 1:
                    fn.write_to_log("Note: --mm_monomer 0 specified with --aggregate=1. No other monomers to embed with zero charges.", is_warning=True)
                else: 
                    for i in range(args.aggregate): 
                        embedding_charges_for_monomer_i: List[MMChargeTupleType] = []
                        for j in range(args.aggregate): 
                            if i == j: continue 
                            other_monomer_coords = core_monomer_coords_list[j]
                            for coord_row in other_monomer_coords: 
                                embedding_charges_for_monomer_i.append((coord_row[0], coord_row[1], coord_row[2], 0.0)) 
                        mm_embedding_charges_for_each_monomer[i] = embedding_charges_for_monomer_i
            elif args.mm_monomer: 
                if len(args.mm_monomer) == args.aggregate :
                    fn.write_to_log(f"Loading MM charges for inter-monomer embedding from {args.aggregate} specified file(s).")
                    all_monomer_explicit_charges_xyz_q = [fn.load_monomer_charges_from_file(f) for f in args.mm_monomer]
                    
                    if args.aggregate == 1:
                         warn_msg = ("--mm_monomer provided with 1 file for --aggregate=1. "
                                     "This charge file will be loaded but not used for inter-monomer embedding as there are no 'other' monomers.")
                         print(f"\nWARNING: {warn_msg}") # Ensure newline
                         fn.write_to_log(warn_msg, is_warning=True)
                    else: 
                        for i in range(args.aggregate): 
                            combined_embedding_charges_for_i = []
                            for j in range(args.aggregate): 
                                if i == j: continue 
                                combined_embedding_charges_for_i.extend(all_monomer_explicit_charges_xyz_q[j])
                            mm_embedding_charges_for_each_monomer[i] = combined_embedding_charges_for_i
                else: 
                    warn_msg = (f"--mm_monomer was given {len(args.mm_monomer)} charge file(s), "
                                f"but --aggregate is {args.aggregate}. Expected {args.aggregate} file(s) "
                                f"for this mode of MM monomer embedding. MM monomer embedding charges will be skipped.")
                    print(f"\nWARNING: {warn_msg}") # Ensure newline
                    fn.write_to_log(warn_msg, is_warning=True)


        # --- XYZ File Generation Block ---
        if user_intends_xyz_output:
            fn.write_to_log("Writing detailed QM region and MM XYZ files (if applicable)...") 
            
            # QM Region XYZ files (filenames and comments are UNCHANGED)
            if actual_gen_aggregate_xyz:
                agg_qm_atoms_xyz, agg_qm_coords_xyz = aggregate_qm_region
                qm_sol_desc_agg = f" + qm {solvent_name_str}" if qm_solvent_flags.get('aggregate_has_added_qm_solvent') else ""
                # Original comment from user's script version
                xyz_comment_for_aggregate_qm = f"{aggregate_xyz_comment_base} qm{qm_sol_desc_agg}" # Corrected: was 'qm {qm_sol_desc_agg}'
                fn.write_xyz(os.path.join(current_frame_output_dir, f'{combined_system_term}_qm.xyz'), 
                          agg_qm_atoms_xyz, agg_qm_coords_xyz, 
                          comment=xyz_comment_for_aggregate_qm)
            
            if actual_gen_monomer_xyz:
                for i, (mono_qm_atoms_xyz, mono_qm_coords_xyz) in enumerate(monomer_qm_regions):
                    monomer_name_from_meta = monomers_metadata_list[i].get(fn.JSON_KEY_NAME, f'm{i+1}')
                    mono_has_qm_sol = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                    qm_sol_desc_mono = f" + its unique qm {solvent_name_str}" if mono_has_qm_sol else ""
                    # Original comment from user's script version
                    xyz_comment_for_monomer_qm = f"{monomer_name_from_meta} monomer{i+1} qm{qm_sol_desc_mono}" # Corrected: was 'qm {qm_sol_desc_mono}'
                    fn.write_xyz(os.path.join(current_frame_output_dir, f'monomer{i+1}_qm.xyz'),
                              mono_qm_atoms_xyz, mono_qm_coords_xyz,
                              comment=xyz_comment_for_monomer_qm)

            # MM Charges XYZ files (NEW LOGIC HERE, filenames and comments UNCHANGED)
            # For {combined_system_term}_mm.xyz
            if actual_gen_aggregate_xyz:
                aggregate_mm_charges_list_for_xyz: List[MMChargeTupleType] = []
                if system_has_mm_solvent and mm_solvent_point_charges_xyz_q:
                    aggregate_mm_charges_list_for_xyz.extend(mm_solvent_point_charges_xyz_q)
                
                if aggregate_mm_charges_list_for_xyz:
                    charges_col = [q_val for x,y,z,q_val in aggregate_mm_charges_list_for_xyz]
                    coords_arr = np.array([(x,y,z) for x,y,z,q_val in aggregate_mm_charges_list_for_xyz])
                    
                    # Original filename and comment from user's script version
                    mm_solvent_agg_comment = f"mm {solvent_name_str} for {base_system_name_for_title} {combined_system_term}"
                    output_path = os.path.join(current_frame_output_dir, f'{combined_system_term}_mm.xyz')
                    fn.write_xyz(output_path, charges_col, coords_arr, comment=mm_solvent_agg_comment)
                    fn.write_to_log(f"Written MM charges XYZ for aggregate to: {output_path} (contains MM solvent charges if any)")
                elif args.mm_solvent : 
                     warn_msg = (f"MM solvent was specified (--mm_solvent) but no point charges were loaded/generated for it. "
                                 f"File '{combined_system_term}_mm.xyz' will not contain MM solvent charges.")
                     print(f"\nWARNING: {warn_msg}") # Ensure newline
                     fn.write_to_log(warn_msg, is_warning=True)

            # For monomer{i+1}_mm.xyz
            if actual_gen_monomer_xyz:
                for i in range(args.aggregate):
                    monomer_name_from_meta = monomers_metadata_list[i].get(fn.JSON_KEY_NAME, f'm{i+1}')
                    
                    # This list will hold all MM charges for this specific monomer's MM XYZ file
                    monomer_combined_mm_charges_for_xyz: List[MMChargeTupleType] = []

                    # 1. Add MM Monomer embedding charges (other monomers)
                    if mm_embedding_charges_for_each_monomer[i]: # Check if list is not empty
                        monomer_combined_mm_charges_for_xyz.extend(mm_embedding_charges_for_each_monomer[i])
                    
                    # 2. Add MM Solvent charges
                    if system_has_mm_solvent and mm_solvent_point_charges_xyz_q:
                        monomer_combined_mm_charges_for_xyz.extend(mm_solvent_point_charges_xyz_q)

                    if monomer_combined_mm_charges_for_xyz: # If there are any charges to write
                        charges_col = [q_val for x,y,z,q_val in monomer_combined_mm_charges_for_xyz]
                        coords_arr = np.array([(x,y,z) for x,y,z,q_val in monomer_combined_mm_charges_for_xyz])
                        # Original filename and comment from user's script version
                        # mm_solvent_mono_comment = f"mm {solvent_name_str} for {monomer_name_from_meta} monomer{i+1}"
                        other_monomer_mm_charges_present = bool(mm_embedding_charges_for_each_monomer[i])
                        if other_monomer_mm_charges_present:
                            mm_solvent_mono_comment = f"mm monomer + mm {solvent_name_str} for {monomer_name_from_meta} monomer{i+1}"
                        else:
                            mm_solvent_mono_comment = f"mm {solvent_name_str} for {monomer_name_from_meta} monomer{i+1}"

                        output_path = os.path.join(current_frame_output_dir, f'monomer{i+1}_mm.xyz')
                        fn.write_xyz(output_path, charges_col, coords_arr, comment=mm_solvent_mono_comment)
                        
                        content_desc = []
                        if mm_embedding_charges_for_each_monomer[i]: content_desc.append("MM monomer embedding charges")
                        if system_has_mm_solvent and mm_solvent_point_charges_xyz_q: content_desc.append("MM solvent charges")
                        
                        if content_desc:
                            fn.write_to_log(f"Written MM charges XYZ for monomer {i+1} to: {output_path} (contains {', '.join(content_desc)})")
                        else: # Should not happen if monomer_combined_mm_charges_for_xyz is True, but as a fallback
                            fn.write_to_log(f"Written MM charges XYZ for monomer {i+1} to: {output_path} (content details unclear but list was non-empty)")

                    # Warning if user specified relevant MM flags, but this specific monomer_mm.xyz ends up empty.
                    elif args.mm_monomer or args.mm_solvent: 
                         # Check if charges were expected but not found for this monomer's context
                         expected_charges_but_empty = False
                         if args.mm_monomer and not mm_embedding_charges_for_each_monomer[i]:
                             expected_charges_but_empty = True
                         if args.mm_solvent and (not system_has_mm_solvent or not mm_solvent_point_charges_xyz_q):
                             # This means solvent was globally specified but not available/empty
                             # And if mm_monomer charges were also empty, then this file is empty.
                             if not mm_embedding_charges_for_each_monomer[i]: # Only trigger if both are missing
                                expected_charges_but_empty = True
                             elif not args.mm_monomer : # if only mm_solvent was specified and it's empty
                                expected_charges_but_empty = True


                         if expected_charges_but_empty:
                             warn_msg_parts = []
                             if args.mm_monomer: warn_msg_parts.append("MM monomer embedding charges were requested/expected but are empty for this context")
                             if args.mm_solvent: warn_msg_parts.append("MM solvent charges were requested/expected but are empty/not loaded")
                             
                             full_warn_msg = (f"For 'monomer{i+1}_mm.xyz': " + " and ".join(warn_msg_parts) +
                                              f". The file 'monomer{i+1}_mm.xyz' will not be written or will be empty (as it has no charges to list).")
                             print(f"\nWARNING: {full_warn_msg}") # Ensure newline
                             fn.write_to_log(full_warn_msg, is_warning=True)
        
        # COM File Generation Block 
        if user_intends_com_output:
            total_sys_charge_from_meta: Union[int, float] = sum(m[fn.JSON_KEY_CHARGE] for m in monomers_metadata_list)
            total_sys_spin_mult_from_meta: int
            if monomers_metadata_list:
                 product_spin = 1
                 for m in monomers_metadata_list:
                     product_spin *= m[fn.JSON_KEY_SPIN_MULT]
                 total_sys_spin_mult_from_meta = product_spin if product_spin > 0 else 1 
            else:
                total_sys_spin_mult_from_meta = 1


            if args.eetg: 
                if actual_gen_aggregate_com: 
                    fn.write_to_log(f"Writing EETG {combined_system_term} .com file...") 
                    eetg_frags_have_qm_solvent = qm_solvent_flags.get('monomer_0_has_added_qm_solvent', False) or \
                                                 (args.aggregate > 1 and qm_solvent_flags.get('monomer_1_has_added_qm_solvent', False))
                    
                    solvent_desc_suffix = fn.get_solvent_descriptor_suffix(eetg_frags_have_qm_solvent, system_has_mm_solvent)
                    eetg_filename_base = f"{combined_system_term}_eetg{solvent_desc_suffix}"
                    eetg_filename = f"{eetg_filename_base}{tag_for_filename}.com"

                    solvent_info_for_eetg_title = ""
                    if eetg_frags_have_qm_solvent and system_has_mm_solvent: solvent_info_for_eetg_title = f" with QM & MM {solvent_name_str}"
                    elif eetg_frags_have_qm_solvent: solvent_info_for_eetg_title = f" with QM {solvent_name_str}"
                    elif system_has_mm_solvent: solvent_info_for_eetg_title = f" with MM {solvent_name_str}"
                    eetg_title = f"{aggregate_title_base}{solvent_info_for_eetg_title} EET Analysis".strip()
                    
                    if args.aggregate == 2 and len(monomer_qm_regions) == 2 and len(monomers_metadata_list) == 2:
                        m1_atoms_eetg, m1_coords_eetg = monomer_qm_regions[0]
                        m2_atoms_eetg, m2_coords_eetg = monomer_qm_regions[1]

                        frag_defs = [
                            (m1_atoms_eetg, m1_coords_eetg, monomers_metadata_list[0][fn.JSON_KEY_CHARGE], monomers_metadata_list[0][fn.JSON_KEY_SPIN_MULT]),
                            (m2_atoms_eetg, m2_coords_eetg, monomers_metadata_list[1][fn.JSON_KEY_CHARGE], monomers_metadata_list[1][fn.JSON_KEY_SPIN_MULT])
                        ]
                        fn.write_com_file(
                            os.path.join(current_frame_output_dir, eetg_filename),
                            gaussian_keywords_for_calcs if gaussian_keywords_for_calcs is not None else [], 
                            eetg_title,
                            total_sys_charge_from_meta, 
                            total_sys_spin_mult_from_meta, 
                            [], np.array([]), 
                            mm_charges_list=mm_solvent_point_charges_xyz_q, 
                            fragment_definitions=frag_defs
                        )
                    else:
                        fn.write_to_log("EETG selected but system is not --aggregate=2 or monomer region data is incomplete. Skipping EETG file.", is_warning=True)

            if actual_gen_monomer_com: 
                fn.write_to_log(f"Writing Monomer .com files (if not exclusively EETG aggregate)...") 
                for i in range(args.aggregate):
                    if i < len(monomer_qm_regions) and i < len(monomers_metadata_list): 
                        mono_meta = monomers_metadata_list[i]
                        mono_qm_atoms, mono_qm_coords = monomer_qm_regions[i]
                        
                        monomer_has_added_qm_solvent = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                        solvent_desc_suffix = fn.get_solvent_descriptor_suffix(monomer_has_added_qm_solvent, system_has_mm_solvent)
                        mono_filename_base = f"monomer{i+1}{solvent_desc_suffix}"
                        mono_filename = f"{mono_filename_base}{tag_for_filename}.com"
                        
                        solvent_info_for_mono_title = ""
                        if monomer_has_added_qm_solvent and system_has_mm_solvent: solvent_info_for_mono_title = f" with QM & MM {solvent_name_str}"
                        elif monomer_has_added_qm_solvent: solvent_info_for_mono_title = f" with QM {solvent_name_str}"
                        elif system_has_mm_solvent: solvent_info_for_mono_title = f" with MM {solvent_name_str}"

                        mono_title = f"{mono_meta.get(fn.JSON_KEY_NAME, f'Monomer {i+1}')} monomer {i+1} {solvent_info_for_mono_title}".strip()

                        all_mm_charges_for_this_monomer_calc: List[MMChargeTupleType] = []
                        if mm_embedding_charges_for_each_monomer[i]: 
                            all_mm_charges_for_this_monomer_calc.extend(mm_embedding_charges_for_each_monomer[i])
                        if mm_solvent_point_charges_xyz_q: 
                            all_mm_charges_for_this_monomer_calc.extend(mm_solvent_point_charges_xyz_q)

                        fn.write_com_file(
                            os.path.join(current_frame_output_dir, mono_filename),
                            gaussian_keywords_for_calcs if gaussian_keywords_for_calcs is not None else [], 
                            mono_title,
                            mono_meta[fn.JSON_KEY_CHARGE], mono_meta[fn.JSON_KEY_SPIN_MULT],
                            mono_qm_atoms, mono_qm_coords, 
                            mm_charges_list=all_mm_charges_for_this_monomer_calc if all_mm_charges_for_this_monomer_calc else None
                        )
                    else:
                        warn_msg = f"Data for monomer {i+1} not available. Skipping its .com file."
                        print(f"\nWARNING: {warn_msg}") # Ensure newline
                        fn.write_to_log(warn_msg, is_warning=True)
            
            if actual_gen_aggregate_com and not args.eetg: 
                fn.write_to_log(f"Writing Aggregate ({combined_system_term}) .com file...") 
                aggregate_has_added_qm_solvent = qm_solvent_flags.get('aggregate_has_added_qm_solvent', False)
                solvent_desc_suffix = fn.get_solvent_descriptor_suffix(aggregate_has_added_qm_solvent, system_has_mm_solvent)
                agg_filename_base = f"{combined_system_term}{solvent_desc_suffix}"
                agg_filename = f"{agg_filename_base}{tag_for_filename}.com"

                solvent_info_for_agg_title = ""
                if aggregate_has_added_qm_solvent and system_has_mm_solvent: solvent_info_for_agg_title = f" with QM & MM {solvent_name_str}"
                elif aggregate_has_added_qm_solvent: solvent_info_for_agg_title = f" with QM {solvent_name_str}"
                elif system_has_mm_solvent: solvent_info_for_agg_title = f" with MM {solvent_name_str}"
                agg_title = f"{aggregate_title_base}{solvent_info_for_agg_title}".strip()
                
                agg_qm_atoms, agg_qm_coords = aggregate_qm_region
                
                fn.write_com_file(
                    os.path.join(current_frame_output_dir, agg_filename),
                    gaussian_keywords_for_calcs if gaussian_keywords_for_calcs is not None else [], 
                    agg_title,
                    total_sys_charge_from_meta, total_sys_spin_mult_from_meta,
                    agg_qm_atoms, agg_qm_coords, 
                    mm_charges_list=mm_solvent_point_charges_xyz_q if mm_solvent_point_charges_xyz_q else None
                )

        # Cleanup temporary XYZ for this frame
        if args.traj_xyz and os.path.exists(temp_xyz_input_filepath) and fn.TEMP_FRAME_XYZ_FILENAME in temp_xyz_input_filepath :
            try:
                os.remove(temp_xyz_input_filepath)
                fn.write_to_log(f"Cleaned up temporary file: {temp_xyz_input_filepath}")
            except OSError as e:
                warn_msg = f"Could not remove temporary file {temp_xyz_input_filepath}: {e}"
                print(f"\nWARNING: {warn_msg}") # Ensure newline
                fn.write_to_log(warn_msg, is_warning=True)
        
        # Update progress indicator for the next iteration or completion
        if args.traj_xyz and total_frames_to_process > 0 and frame_idx == total_frames_to_process -1 :
             sys.stdout.write(f"\rProgress: Frame {frame_idx + 1}/{total_frames_to_process} - Complete. \n")
             sys.stdout.flush()
        elif args.traj_xyz and total_frames_to_process > 0 : # For intermediate frames
            pass # The next loop iteration will print "Processing Frame X / Y" on a new line.
                 # And then overwrite the "Progress: Frame X/Y" line.

    # Final newline if processing a trajectory to move past the progress line.
    if args.traj_xyz and total_frames_to_process > 0:
        sys.stdout.write("\n") 
        sys.stdout.flush()

    final_message = "All processing complete." # Removed leading newline, print() adds one.
    print(final_message)
    fn.write_to_log(final_message)

    if fn.log_file_handle:
        fn.log_file_handle.close()

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e: 
        err_msg_critical = f"CRITICAL FILE ERROR: {e}. Please check file paths and permissions."
        print(f"ERROR: {err_msg_critical}", file=sys.stderr)
        if fn.log_file_handle and not fn.log_file_handle.closed: 
            fn.write_to_log(err_msg_critical, is_error=True)
            import traceback
            fn.write_to_log(traceback.format_exc())
        sys.exit(1)
    except ValueError as e: 
        err_msg_critical = f"CRITICAL VALUE ERROR: {e}. Please check input values and formats."
        print(f"ERROR: {err_msg_critical}", file=sys.stderr)
        if fn.log_file_handle and not fn.log_file_handle.closed:
            fn.write_to_log(err_msg_critical, is_error=True)
            import traceback # Moved import here
            traceback.print_exc(file=fn.log_file_handle) 
        sys.exit(1)
    except Exception as e: 
        err_msg_critical = f"AN UNHANDLED CRITICAL ERROR OCCURRED: {e}"
        print(f"ERROR: {err_msg_critical}", file=sys.stderr)
        if fn.log_file_handle and not fn.log_file_handle.closed: 
            fn.write_to_log(err_msg_critical, is_error=True)
            import traceback # Moved import here
            fn.write_to_log(traceback.format_exc()) 
        sys.exit(1)
    finally:
        if fn.log_file_handle and not fn.log_file_handle.closed:
            fn.log_file_handle.close()
