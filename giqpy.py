#!/usr/bin/env python3
"""Generate Inputs for QM/MM systems for Gaussian or TeraChem."""

# --- Imports ---
import os
import sys
import datetime
from typing import List, Tuple, Optional, Dict, Any

from functions import (
    create_parser,
    write_to_log,
    split_frames,
    write_xyz,
    load_system_info,
    localize_solvent_and_prepare_regions,
    load_monomer_charges_from_file,
    assign_charges_to_solvent_molecules,
    calculate_centroid_distance_between_first_two,
    load_keywords_from_file,
    get_solvent_descriptor_suffix,
    write_com_file,
    AUTO_MM_SOLVENT_TRIGGER,
    TEMP_FRAME_XYZ_FILENAME,
    LOG_FILENAME,
    log_file_handle,
)

# --- Main Routine ---
def main() -> None:
    global log_file_handle

    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Setup log file
    effective_log_filename = args.logfile
    try:
        log_file_handle = open(effective_log_filename, 'w')
        log_file_handle.write(f"GIQPy Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file_handle.write(f"Arguments: {vars(args)}\n\n")
        log_file_handle.flush()
    except IOError as e:
        print(f"CRITICAL ERROR: Could not open log file {effective_log_filename} for writing: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- Setup Log File ---
    # Use the logfile name from args, which has a default.
    # Global LOG_FILENAME constant is not used directly here anymore for open().
    effective_log_filename = args.logfile
    try:
        log_file_handle = open(effective_log_filename, 'w')
        log_file_handle.write(f"GIQPy Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
        log_file_handle.write(f"Arguments: {vars(args)}\n\n")
        log_file_handle.flush()
    except IOError as e:
        print(f"CRITICAL ERROR: Could not open log file {effective_log_filename} for writing: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)


    # --- Argument Validation ---
    user_intends_com_output = args.gauss_files is not None

    if args.nDyes < 1:
        err_msg = "--nDyes must be at least 1."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        parser.error(err_msg)
    if args.eetg:
        if args.nDyes != 2:
            err_msg = "--eetg calculations require --nDyes=2 (a dimer system)."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
        if not user_intends_com_output:
            err_msg = "--eetg calculations require .com file output. Please specify --gauss_files."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
            
    if user_intends_com_output and args.gauss_files == 'monomer' and args.eetg:
        warn_msg = ("Configuration: --gauss_files set to 'monomer' with --eetg. "
                    "EETG is an aggregate/dimer property. An EETG file for the dimer will be generated if --nDyes=2, "
                    "but individual monomer .com files will also be generated as per --gauss_files monomer.")
        print(f"INFO: {warn_msg}") # Changed to INFO as it's a clarification
        write_to_log(warn_msg, is_warning=False) # Not strictly a warning of misconfiguration

    # Gaussian keywords handling
    gaussian_keywords_for_calcs: Optional[List[str]] = None # Variable to hold loaded keywords
    if user_intends_com_output:
        import gen_gauss_files as gf
        if not args.gauss_keywords:
            err_msg = "A Gaussian keywords file must be provided via --gauss_keywords when --gauss_files is specified."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
        
        if not os.path.exists(args.gauss_keywords) or os.path.getsize(args.gauss_keywords) == 0:
            err_msg = f"Gaussian keywords file '{args.gauss_keywords}' not found or is empty."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            # parser.error(err_msg) # This would exit before log flushes sometimes
            raise FileNotFoundError(err_msg) # More standard for file issues
        gaussian_keywords_for_calcs = gf.load_keywords_from_file(args.gauss_keywords)
    elif args.gauss_keywords: # Keywords provided, but .com output not requested
        warn_msg = ("--gauss_keywords was provided, but --gauss_files was not specified. "
                    "The keywords file will be ignored as no .com files are being generated.")
        print(f"WARNING: {warn_msg}")
        write_to_log(warn_msg, is_warning=True)

    # --- End of Argument Validation ---



    base_output_dir: str = os.getcwd() 

    processed_frames_list: List[Tuple[str, str, str]] = split_frames(args.traj, args.nFrames, base_output_dir)
    total_frames_to_process = len(processed_frames_list)
    
    if total_frames_to_process == 0: # This implies an issue with split_frames or input files
        err_msg = f"No frames were found or could be processed from the input XYZ source."
        if args.traj:
             err_msg = f"No frames were found or could be processed from trajectory: {args.traj}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        sys.exit(1)


    monomers_metadata_list: List[Dict[str, Any]]
    solvent_metadata: Dict[str, Any]
    try:
        monomers_metadata_list, solvent_metadata = load_system_info(args.system_info, args.nDyes)
    except Exception as e: # Catch errors from load_system_info (FileNotFound, ValueError)
        print(f"CRITICAL ERROR loading system_info: {e}", file=sys.stderr)
        write_to_log(f"CRITICAL ERROR loading system_info: {e}", is_error=True)
        sys.exit(1)

    solvent_name_str = solvent_metadata.get(JSON_KEY_NAME, "solvent")


    if len(monomers_metadata_list) != args.nDyes:
        # This check might be redundant if load_system_info already validates based on n_dyes count
        msg = (
            f"Number of monomer entries in system_info ({len(monomers_metadata_list)}) "
            f"does not match --nDyes ({args.nDyes})."
        )
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
        # raise ValueError(msg) # Or sys.exit(1)
        sys.exit(1)


    combined_system_term: str = "dimer" if args.nDyes == 2 else "aggregate"
    tag_for_filename = f"_{args.tag}" if args.tag else "" 


    if args.traj and total_frames_to_process > 0 :
        sys.stdout.write(f"\nProcessed Frames: 0 / {total_frames_to_process}\n") # Ensure this prints on a new line
        sys.stdout.flush()

    # --- Determine what types of files to generate based on user's explicit choices ---
    # .com file generation flags
    # If args.gauss_files is None (flag not given), these will be False.
    actual_gen_monomer_com: bool = (user_intends_com_output and args.gauss_files in ['monomer', 'both'])
    actual_gen_aggregate_com: bool = (user_intends_com_output and args.gauss_files in ['dimer', 'both'])

    # .xyz output is always generated
    actual_gen_monomer_xyz: bool = True
    actual_gen_aggregate_xyz: bool = args.nDyes >= 2


    for frame_idx, (frame_id_str, temp_xyz_input_filepath, current_frame_output_dir) in enumerate(processed_frames_list):
        # Progress indicator for trajectory processing
        if args.traj and total_frames_to_process > 0 :
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
        write_to_log(f"\n\n{frame_processing_message}")
        
        # The progress update that gets overwritten
        if args.traj and total_frames_to_process > 0:
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
                localize_solvent_and_prepare_regions(
                    temp_xyz_input_filepath,
                    args.system_info,
                    args.nDyes,
                    args.qmSol_radius
                )
        except Exception as e:
            err_msg = f"Error during solvent localization/region preparation for frame {frame_id_str if frame_id_str else 'single'}: {e}"
            print(f"\nERROR: {err_msg}", file=sys.stderr) # Newline if progress bar was on previous line
            write_to_log(err_msg, is_error=True)
            if args.traj:
                write_to_log(f"Skipping frame {frame_id_str} due to error.", is_warning=True)
                # No need to rewrite progress here, loop will continue or end.
                continue 
            else:
                sys.exit(1) 


        monomer_names_from_meta: List[str] = [m.get(JSON_KEY_NAME, f'm{idx+1}') for idx, m in enumerate(monomers_metadata_list)]
        base_system_name_for_title: str
        if len(set(monomer_names_from_meta)) == 1 and monomers_metadata_list : # check list not empty
            base_system_name_for_title = monomer_names_from_meta[0]
        else: 
            base_system_name_for_title = "_".join(monomer_names_from_meta) if monomer_names_from_meta else "system" # handle empty case
        
        distance_str_for_title: str = ""
        if args.nDyes >= 2:
            dist = calculate_centroid_distance_between_first_two(core_coords_for_dist_calc, n_atoms_per_monomer_for_dist_calc)
            if dist is not None:
                distance_str_for_title = f"(m1-m2 centroid dist: {dist:.2f} A)" # Angstrom symbol
        
        aggregate_title_base: str = f"{base_system_name_for_title} {combined_system_term} {distance_str_for_title}".strip()
        # User requested comments not to be changed, these are the original ones for QM XYZ files
        aggregate_xyz_comment_base: str = f"{base_system_name_for_title} {combined_system_term} {distance_str_for_title}".strip()


        mm_solvent_point_charges_xyz_q: Optional[List[MMChargeTupleType]] = None # Format (x,y,z,q)
        system_has_mm_solvent: bool = False
        if args.mm_solvent == AUTO_MM_SOLVENT_TRIGGER:
            write_to_log("Attempting to auto-detect MM solvent from non-QM solvent molecules...") 
            if non_qm_sol_groups:
                try:
                    mm_solvent_point_charges_xyz_q = assign_charges_to_solvent_molecules(non_qm_sol_groups, solvent_metadata)
                    if mm_solvent_point_charges_xyz_q:
                        write_to_log(f"Generated {len(mm_solvent_point_charges_xyz_q)} MM solvent point charges (x,y,z,q) from auto-detection.") 
                        system_has_mm_solvent = True
                    else:
                        write_to_log("Auto-detection found non-QM solvent groups, but no charges were assigned (check solvent definition or warnings).") 
                except ValueError as e:
                     write_to_log(f"Error assigning charges to auto-detected MM solvent: {e}", is_error=True)
            else:
                write_to_log("No non-QM solvent molecules found to treat as MM solvent via auto-detection.") 
        elif args.mm_solvent is not None: # Path to MM solvent file is provided
            write_to_log(f"Loading MM solvent charges from file: {args.mm_solvent}") 
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
                                    write_to_log(warn_msg, is_warning=True)
                            elif parts: # Line not empty, but not 4 parts
                                warn_msg = f"Malformed MM solvent line in {args.mm_solvent} on line {line_num + 3} (expected 4 values, got {len(parts)}): {line_content.strip()}"
                                print(f"\nWARNING: {warn_msg}") # Ensure newline
                                write_to_log(warn_msg, is_warning=True)
                    else:
                        warn_msg = f"MM solvent file {args.mm_solvent} is too short (expected XYZ-like format with 2 header lines). No charges loaded."
                        print(f"\nWARNING: {warn_msg}") # Ensure newline
                        write_to_log(warn_msg, is_warning=True)

                if loaded_mm_charges_from_file_chg_xyz:
                    # Convert to (x,y,z,q) for internal consistency for Gaussian
                    mm_solvent_point_charges_xyz_q = [(x,y,z,q) for q,x,y,z in loaded_mm_charges_from_file_chg_xyz]
                    system_has_mm_solvent = True
                    write_to_log(f"Loaded {len(mm_solvent_point_charges_xyz_q)} MM solvent point charges (x,y,z,q) from file.") 
                else:
                    # This warning might be redundant if the file was short/malformed and already warned.
                    if os.path.exists(args.mm_solvent): # Only warn if file exists but no charges loaded
                        warn_msg = f"No valid MM solvent charges loaded from file {args.mm_solvent}."
                        print(f"\nWARNING: {warn_msg}") # Ensure newline
                        write_to_log(warn_msg, is_warning=True)
            except FileNotFoundError:
                warn_msg = f"MM solvent file {args.mm_solvent} not found. No MM solvent from file will be used."
                print(f"\nWARNING: {warn_msg}") # Ensure newline
                write_to_log(warn_msg, is_warning=True)
        
        # MM Monomer Embedding Charges (this logic remains the same, result is used later)
        mm_embedding_charges_for_each_monomer: List[List[MMChargeTupleType]] = [[] for _ in range(args.nDyes)] 
        if args.mm_monomer is not None: 
            if args.mm_monomer == ['0']: 
                write_to_log("Applying zero charges for other-monomer embedding as per --mm_monomer 0.")
                current_atom_idx = 0
                core_monomer_coords_list: List[CoordType] = []
                for num_atoms_in_mono in n_atoms_per_monomer_for_dist_calc: 
                    core_monomer_coords_list.append(
                        core_coords_for_dist_calc[current_atom_idx : current_atom_idx + num_atoms_in_mono]
                    )
                    current_atom_idx += num_atoms_in_mono
                
                if args.nDyes == 1:
                    write_to_log(
                        "Note: --mm_monomer 0 specified with --nDyes=1. No other monomers to embed with zero charges.",
                        is_warning=True,
                    )
                else: 
                    for i in range(args.nDyes): 
                        embedding_charges_for_monomer_i: List[MMChargeTupleType] = []
                        for j in range(args.nDyes): 
                            if i == j: continue 
                            other_monomer_coords = core_monomer_coords_list[j]
                            for coord_row in other_monomer_coords: 
                                embedding_charges_for_monomer_i.append((coord_row[0], coord_row[1], coord_row[2], 0.0)) 
                        mm_embedding_charges_for_each_monomer[i] = embedding_charges_for_monomer_i
            elif args.mm_monomer: 
                if len(args.mm_monomer) == args.nDyes :
                    write_to_log(f"Loading MM charges for inter-monomer embedding from {args.nDyes} specified file(s).")
                    all_monomer_explicit_charges_xyz_q = [load_monomer_charges_from_file(f) for f in args.mm_monomer]
                    
                    if args.nDyes == 1:
                         warn_msg = ("--mm_monomer provided with 1 file for --nDyes=1. "
                                     "This charge file will be loaded but not used for inter-monomer embedding as there are no 'other' monomers.")
                         print(f"\nWARNING: {warn_msg}") # Ensure newline
                         write_to_log(warn_msg, is_warning=True)
                    else: 
                        for i in range(args.nDyes): 
                            combined_embedding_charges_for_i = []
                            for j in range(args.nDyes): 
                                if i == j: continue 
                                combined_embedding_charges_for_i.extend(all_monomer_explicit_charges_xyz_q[j])
                            mm_embedding_charges_for_each_monomer[i] = combined_embedding_charges_for_i
                else:
                    warn_msg = (
                        f"--mm_monomer was given {len(args.mm_monomer)} charge file(s), "
                        f"but --nDyes is {args.nDyes}. Expected {args.nDyes} file(s) "
                                f"for this mode of MM monomer embedding. MM monomer embedding charges will be skipped.")
                    print(f"\nWARNING: {warn_msg}") # Ensure newline
                    write_to_log(warn_msg, is_warning=True)


        # --- XYZ File Generation Block ---
        write_to_log("Writing detailed QM region and MM XYZ files (if applicable)...")

        # QM Region XYZ files (filenames and comments are UNCHANGED)
        if actual_gen_aggregate_xyz:
            agg_qm_atoms_xyz, agg_qm_coords_xyz = aggregate_qm_region
            qm_sol_desc_agg = f" + qm {solvent_name_str}" if qm_solvent_flags.get('aggregate_has_added_qm_solvent') else ""
            xyz_comment_for_aggregate_qm = f"{aggregate_xyz_comment_base} qm{qm_sol_desc_agg}"  # Corrected: was 'qm {qm_sol_desc_agg}'
            write_xyz(
                os.path.join(current_frame_output_dir, f'{combined_system_term}_qm.xyz'),
                agg_qm_atoms_xyz,
                agg_qm_coords_xyz,
                comment=xyz_comment_for_aggregate_qm,
            )
            
            if actual_gen_monomer_xyz:
                for i, (mono_qm_atoms_xyz, mono_qm_coords_xyz) in enumerate(monomer_qm_regions):
                    monomer_name_from_meta = monomers_metadata_list[i].get(JSON_KEY_NAME, f'm{i+1}')
                    mono_has_qm_sol = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                    qm_sol_desc_mono = f" + its unique qm {solvent_name_str}" if mono_has_qm_sol else ""
                    xyz_comment_for_monomer_qm = f"{monomer_name_from_meta} monomer{i+1} qm{qm_sol_desc_mono}" # Corrected: was 'qm {qm_sol_desc_mono}'
                    write_xyz(os.path.join(current_frame_output_dir, f'monomer{i+1}_qm.xyz'),
                              mono_qm_atoms_xyz, mono_qm_coords_xyz,
                              comment=xyz_comment_for_monomer_qm)

            # For {combined_system_term}_mm.xyz
            if actual_gen_aggregate_xyz:
                aggregate_mm_charges_list_for_xyz: List[MMChargeTupleType] = []
                if system_has_mm_solvent and mm_solvent_point_charges_xyz_q:
                    aggregate_mm_charges_list_for_xyz.extend(mm_solvent_point_charges_xyz_q)
                
                if aggregate_mm_charges_list_for_xyz:
                    charges_col = [q_val for x,y,z,q_val in aggregate_mm_charges_list_for_xyz]
                    coords_arr = np.array([(x,y,z) for x,y,z,q_val in aggregate_mm_charges_list_for_xyz])
                    
                    mm_solvent_agg_comment = f"mm {solvent_name_str} for {base_system_name_for_title} {combined_system_term}"
                    output_path = os.path.join(current_frame_output_dir, f'{combined_system_term}_mm.xyz')
                    write_xyz(output_path, charges_col, coords_arr, comment=mm_solvent_agg_comment)
                elif args.mm_solvent : 
                     warn_msg = (f"MM solvent was specified (--mm_solvent) but no point charges were loaded/generated for it. "
                                 f"File '{combined_system_term}_mm.xyz' will not contain MM solvent charges.")
                     print(f"\nWARNING: {warn_msg}") # Ensure newline
                     write_to_log(warn_msg, is_warning=True)

            # For monomer{i+1}_mm.xyz
            if actual_gen_monomer_xyz:
                for i in range(args.nDyes):
                    monomer_name_from_meta = monomers_metadata_list[i].get(JSON_KEY_NAME, f'm{i+1}')
                    
                    # This list will hold all MM charges for this specific monomer's MM XYZ file
                    monomer_combined_mm_charges_for_xyz: List[MMChargeTupleType] = []

                    # 1. Add MM Monomer embedding charges (other monomers)
                    if mm_embedding_charges_for_each_monomer[i]: # Check if list is not empty
                        monomer_combined_mm_charges_for_xyz.extend(mm_embedding_charges_for_each_monomer[i])
                    
                    # 2. Add MM Solvent charges
                    if system_has_mm_solvent and mm_solvent_point_charges_xyz_q:
                        monomer_combined_mm_charges_for_xyz.extend(mm_solvent_point_charges_xyz_q)

                        coords_arr = np.array([(x,y,z) for x,y,z,q_val in monomer_combined_mm_charges_for_xyz])
                        # mm_solvent_mono_comment = f"mm {solvent_name_str} for {monomer_name_from_meta} monomer{i+1}"
                        other_monomer_mm_charges_present = bool(mm_embedding_charges_for_each_monomer[i])
                        if other_monomer_mm_charges_present:
                            mm_solvent_mono_comment = f"mm monomer + mm {solvent_name_str} for {monomer_name_from_meta} monomer{i+1}"
                        else:
                            mm_solvent_mono_comment = f"mm {solvent_name_str} for {monomer_name_from_meta} monomer{i+1}"

                        output_path = os.path.join(current_frame_output_dir, f'monomer{i+1}_mm.xyz')
                        write_xyz(output_path, charges_col, coords_arr, comment=mm_solvent_mono_comment)
                        
                        content_desc = []
                        if mm_embedding_charges_for_each_monomer[i]: content_desc.append("MM monomer embedding charges")
                        if system_has_mm_solvent and mm_solvent_point_charges_xyz_q: content_desc.append("MM solvent charges")
                        
                        if content_desc:
                            write_to_log(f"Written MM charges XYZ for monomer {i+1} to: {output_path} (contains {', '.join(content_desc)})")
                        else: # Should not happen if monomer_combined_mm_charges_for_xyz is True, but as a fallback
                            write_to_log(f"Written MM charges XYZ for monomer {i+1} to: {output_path} (content details unclear but list was non-empty)")

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
                             write_to_log(full_warn_msg, is_warning=True)
        
        # COM File Generation Block 
        if user_intends_com_output:
            total_sys_charge_from_meta: Union[int, float] = sum(m[JSON_KEY_CHARGE] for m in monomers_metadata_list)
            total_sys_spin_mult_from_meta: int
            if monomers_metadata_list:
                 product_spin = 1
                 for m in monomers_metadata_list:
                     product_spin *= m[JSON_KEY_SPIN_MULT]
                 total_sys_spin_mult_from_meta = product_spin if product_spin > 0 else 1 
            else:
                total_sys_spin_mult_from_meta = 1


            if args.eetg: 
                if actual_gen_aggregate_com: 
                    write_to_log(f"Writing EETG {combined_system_term} .com file...") 
                    eetg_frags_have_qm_solvent = qm_solvent_flags.get('monomer_0_has_added_qm_solvent', False) or \
                                                 (args.nDyes > 1 and qm_solvent_flags.get('monomer_1_has_added_qm_solvent', False))
                    
                    solvent_desc_suffix = gf.get_solvent_descriptor_suffix(eetg_frags_have_qm_solvent, system_has_mm_solvent)
                    eetg_filename_base = f"{combined_system_term}_eetg{solvent_desc_suffix}"
                    eetg_filename = f"{eetg_filename_base}{tag_for_filename}.com"

                    solvent_info_for_eetg_title = ""
                    if eetg_frags_have_qm_solvent and system_has_mm_solvent: solvent_info_for_eetg_title = f" with QM & MM {solvent_name_str}"
                    elif eetg_frags_have_qm_solvent: solvent_info_for_eetg_title = f" with QM {solvent_name_str}"
                    elif system_has_mm_solvent: solvent_info_for_eetg_title = f" with MM {solvent_name_str}"
                    eetg_title = f"{aggregate_title_base}{solvent_info_for_eetg_title} EET Analysis".strip()
                    
                    if args.nDyes == 2 and len(monomer_qm_regions) == 2 and len(monomers_metadata_list) == 2:
                        m1_atoms_eetg, m1_coords_eetg = monomer_qm_regions[0]
                        m2_atoms_eetg, m2_coords_eetg = monomer_qm_regions[1]

                        frag_defs = [
                            (m1_atoms_eetg, m1_coords_eetg, monomers_metadata_list[0][JSON_KEY_CHARGE], monomers_metadata_list[0][JSON_KEY_SPIN_MULT]),
                            (m2_atoms_eetg, m2_coords_eetg, monomers_metadata_list[1][JSON_KEY_CHARGE], monomers_metadata_list[1][JSON_KEY_SPIN_MULT])
                        ]
                        gf.write_com_file(
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
                        write_to_log("EETG selected but system is not --nDyes=2 or monomer region data is incomplete. Skipping EETG file.", is_warning=True)

            if actual_gen_monomer_com: 
                write_to_log(f"Writing Monomer .com files (if not exclusively EETG aggregate)...") 
                for i in range(args.nDyes):
                    if i < len(monomer_qm_regions) and i < len(monomers_metadata_list): 
                        mono_meta = monomers_metadata_list[i]
                        mono_qm_atoms, mono_qm_coords = monomer_qm_regions[i]
                        
                        monomer_has_added_qm_solvent = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                        solvent_desc_suffix = gf.get_solvent_descriptor_suffix(monomer_has_added_qm_solvent, system_has_mm_solvent)
                        mono_filename_base = f"monomer{i+1}{solvent_desc_suffix}"
                        mono_filename = f"{mono_filename_base}{tag_for_filename}.com"
                        
                        solvent_info_for_mono_title = ""
                        if monomer_has_added_qm_solvent and system_has_mm_solvent: solvent_info_for_mono_title = f" with QM & MM {solvent_name_str}"
                        elif monomer_has_added_qm_solvent: solvent_info_for_mono_title = f" with QM {solvent_name_str}"
                        elif system_has_mm_solvent: solvent_info_for_mono_title = f" with MM {solvent_name_str}"

                        mono_title = f"{mono_meta.get(JSON_KEY_NAME, f'Monomer {i+1}')} monomer {i+1} {solvent_info_for_mono_title}".strip()

                        all_mm_charges_for_this_monomer_calc: List[MMChargeTupleType] = []
                        if mm_embedding_charges_for_each_monomer[i]: 
                            all_mm_charges_for_this_monomer_calc.extend(mm_embedding_charges_for_each_monomer[i])
                        if mm_solvent_point_charges_xyz_q: 
                            all_mm_charges_for_this_monomer_calc.extend(mm_solvent_point_charges_xyz_q)

                        gf.write_com_file(
                            os.path.join(current_frame_output_dir, mono_filename),
                            gaussian_keywords_for_calcs if gaussian_keywords_for_calcs is not None else [], 
                            mono_title,
                            mono_meta[JSON_KEY_CHARGE], mono_meta[JSON_KEY_SPIN_MULT],
                            mono_qm_atoms, mono_qm_coords, 
                            mm_charges_list=all_mm_charges_for_this_monomer_calc if all_mm_charges_for_this_monomer_calc else None
                        )
                    else:
                        warn_msg = f"Data for monomer {i+1} not available. Skipping its .com file."
                        print(f"\nWARNING: {warn_msg}") # Ensure newline
                        write_to_log(warn_msg, is_warning=True)
            
            if actual_gen_aggregate_com and not args.eetg: 
                write_to_log(f"Writing Aggregate ({combined_system_term}) .com file...") 
                aggregate_has_added_qm_solvent = qm_solvent_flags.get('aggregate_has_added_qm_solvent', False)
                solvent_desc_suffix = gf.get_solvent_descriptor_suffix(aggregate_has_added_qm_solvent, system_has_mm_solvent)
                agg_filename_base = f"{combined_system_term}{solvent_desc_suffix}"
                agg_filename = f"{agg_filename_base}{tag_for_filename}.com"

                solvent_info_for_agg_title = ""
                if aggregate_has_added_qm_solvent and system_has_mm_solvent: solvent_info_for_agg_title = f" with QM & MM {solvent_name_str}"
                elif aggregate_has_added_qm_solvent: solvent_info_for_agg_title = f" with QM {solvent_name_str}"
                elif system_has_mm_solvent: solvent_info_for_agg_title = f" with MM {solvent_name_str}"
                agg_title = f"{aggregate_title_base}{solvent_info_for_agg_title}".strip()
                
                agg_qm_atoms, agg_qm_coords = aggregate_qm_region
                
                gf.write_com_file(
                    os.path.join(current_frame_output_dir, agg_filename),
                    gaussian_keywords_for_calcs if gaussian_keywords_for_calcs is not None else [], 
                    agg_title,
                    total_sys_charge_from_meta, total_sys_spin_mult_from_meta,
                    agg_qm_atoms, agg_qm_coords, 
                    mm_charges_list=mm_solvent_point_charges_xyz_q if mm_solvent_point_charges_xyz_q else None
                )

        # Cleanup temporary XYZ for this frame
        if args.traj and os.path.exists(temp_xyz_input_filepath) and TEMP_FRAME_XYZ_FILENAME in temp_xyz_input_filepath :
            try:
                os.remove(temp_xyz_input_filepath)
                write_to_log(f"Cleaned up temporary file: {temp_xyz_input_filepath}")
            except OSError as e:
                warn_msg = f"Could not remove temporary file {temp_xyz_input_filepath}: {e}"
                print(f"\nWARNING: {warn_msg}") # Ensure newline
                write_to_log(warn_msg, is_warning=True)
        
        # Update progress indicator for the next iteration or completion
        if args.traj and total_frames_to_process > 0 and frame_idx == total_frames_to_process -1 :
             sys.stdout.write(f"\rProgress: Frame {frame_idx + 1}/{total_frames_to_process} - Complete. \n")
             sys.stdout.flush()
        elif args.traj and total_frames_to_process > 0 : # For intermediate frames
            pass # The next loop iteration will print "Processing Frame X / Y" on a new line.
                 # And then overwrite the "Progress: Frame X/Y" line.

    # Final newline if processing a trajectory to move past the progress line.
    if args.traj and total_frames_to_process > 0:
        sys.stdout.write("\n") 
        sys.stdout.flush()

    final_message = "All processing complete." # Removed leading newline, print() adds one.
    print(final_message)
    write_to_log(final_message)

    if log_file_handle:
        log_file_handle.close()
if __name__ == '__main__':
    main()
