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
import argparse
import os
import json
import numpy as np
import re
import shutil 
import sys 
import datetime 
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union

# --- Constants ---
AUTO_MM_SOLVENT_TRIGGER: str = "auto_detect_mm_solvent_from_input_xyz"
TEMP_FRAME_XYZ_FILENAME: str = "_current_frame_data.xyz"
LOG_FILENAME: str = "giqpy_run.log" # Modified default log filename in main

# --- JSON Keys Constants ---
JSON_KEY_NAME: str = "name"
JSON_KEY_NATOMS: str = "nAtoms"
JSON_KEY_CHARGE: str = "charge"
JSON_KEY_SPIN_MULT: str = "spin_mult"
JSON_KEY_MOL_FORMULA: str = "mol_formula"
JSON_KEY_CHARGES_ARRAY: str = "charges" 
JSON_KEY_ELEMENT: str = "element"       

# --- Global Log File Object ---
log_file_handle: Optional[Any] = None

# --- Type Aliases ---
CoordType = np.ndarray 
AtomListType = List[str]
ChargeListType = List[float] 
SolventGroupType = Tuple[AtomListType, CoordType]
MMChargeTupleType = Tuple[float, float, float, float] 

# --- Helper for logging to file ---
def write_to_log(message: str, is_error: bool = False, is_warning: bool = False) -> None:
    """Writes a message to the global log file if it's open."""
    if log_file_handle:
        prefix = ""
        if is_error:
            prefix = "ERROR: "
        elif is_warning:
            prefix = "WARNING: "
        try:
            log_file_handle.write(prefix + message + "\n")
            log_file_handle.flush() 
        except IOError:
            print(f"FALLBACK CONSOLE (log write error): {prefix}{message}", file=sys.stderr)


# --- Functions ---

def split_frames(
    single_xyz_path: Optional[str], 
    traj_xyz_path: Optional[str], 
    num_frames_to_extract: Optional[int], 
    base_output_dir_for_traj: str
) -> List[Tuple[Optional[str], str, str]]:
    """
    Processes single or trajectory XYZ files.
    For trajectories, creates frame-specific directories and temporary XYZ files within them.
    Returns a list of (frame_id_str, path_to_xyz_for_frame, output_dir_for_frame) tuples.
    """
    processed_frames_info: List[Tuple[Optional[str], str, str]] = []

    if single_xyz_path:
        # Ensure base output directory exists even for single_xyz if it's not current dir
        if base_output_dir_for_traj != "." and not os.path.exists(base_output_dir_for_traj):
            try:
                os.makedirs(base_output_dir_for_traj, exist_ok=True)
            except OSError as e:
                err_msg = f"Could not create output directory {base_output_dir_for_traj}: {e}"
                print(f"ERROR: {err_msg}", file=sys.stderr)
                if log_file_handle: write_to_log(err_msg, is_error=True)
                raise
        processed_frames_info.append((None, single_xyz_path, base_output_dir_for_traj))
        return processed_frames_info

    if not traj_xyz_path: 
        # This case should ideally not be reached if logic in main is correct
        # (i.e., one of single_xyz_path or traj_xyz_path must be provided).
        print("ERROR: Trajectory path is None, but single_xyz_path was also None.", file=sys.stderr)
        if log_file_handle: write_to_log("Trajectory path is None, but single_xyz_path was also None.", is_error=True)
        return [] # Or raise an error

    try:
        with open(traj_xyz_path, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"ERROR: Trajectory file not found: {traj_xyz_path}", file=sys.stderr)
        if log_file_handle: write_to_log(f"Trajectory file not found: {traj_xyz_path}", is_error=True)
        raise
        
    if not lines:
        print(f"ERROR: Empty trajectory: {traj_xyz_path}", file=sys.stderr)
        if log_file_handle: write_to_log(f"Empty trajectory: {traj_xyz_path}", is_error=True)
        raise ValueError(f"Empty trajectory: {traj_xyz_path}")
    
    try:
        total_atoms_per_frame = int(lines[0].strip())
    except ValueError:
        print(f"ERROR: First line of trajectory {traj_xyz_path} must be the number of atoms.", file=sys.stderr)
        if log_file_handle: write_to_log(f"First line of trajectory {traj_xyz_path} must be the number of atoms.", is_error=True)
        raise
        
    block_size = total_atoms_per_frame + 2 
    if block_size <= 2: # Number of atoms must be positive.
        print(f"ERROR: Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.", file=sys.stderr)
        if log_file_handle: write_to_log(f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.", is_error=True)
        raise ValueError(f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.")

    max_possible_frames = len(lines) // block_size
    
    actual_frames_to_process = num_frames_to_extract if num_frames_to_extract and num_frames_to_extract <= max_possible_frames else max_possible_frames
    if num_frames_to_extract and num_frames_to_extract > max_possible_frames:
        warn_msg = f"Requested {num_frames_to_extract} frames, but trajectory only contains {max_possible_frames}. Processing all available frames."
        print(f"WARNING: {warn_msg}")
        if log_file_handle: write_to_log(warn_msg, is_warning=True)


    for i in range(actual_frames_to_process):
        frame_id_str = str(i + 1)
        output_dir_for_frame = os.path.join(base_output_dir_for_traj, frame_id_str)
        try:
            os.makedirs(output_dir_for_frame, exist_ok=True)
        except OSError as e:
            err_msg = f"Could not create frame-specific output directory {output_dir_for_frame}: {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            if log_file_handle: write_to_log(err_msg, is_error=True)
            raise

        temp_xyz_path_for_frame = os.path.join(output_dir_for_frame, TEMP_FRAME_XYZ_FILENAME)
        
        start_line_idx = i * block_size
        frame_block_lines = lines[start_line_idx : start_line_idx + block_size]
        
        try:
            with open(temp_xyz_path_for_frame, 'w') as f_temp_xyz:
                f_temp_xyz.write("\n".join(frame_block_lines) + "\n")
        except IOError as e:
            err_msg = f"Could not write temporary frame file {temp_xyz_path_for_frame}: {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            if log_file_handle: write_to_log(err_msg, is_error=True)
            raise
            
        processed_frames_info.append((frame_id_str, temp_xyz_path_for_frame, output_dir_for_frame))
    
    return processed_frames_info

# ... (rest of the functions like load_system_info, parse_formula, etc., are unchanged) ...
def load_system_info(system_info_path: str, aggregate: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load monomer and solvent metadata from a single JSON array file.
    Performs validation on the structure and content of the JSON.
    """
    if log_file_handle: write_to_log(f"Loading system info from: {system_info_path}")
    try:
        with open(system_info_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        err_msg = f"System info JSON file not found: {system_info_path}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        if log_file_handle: write_to_log(err_msg, is_error=True)
        raise
    except json.JSONDecodeError as e:
        err_msg = f"Invalid JSON format in system info file: {system_info_path}. Error: {e}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        if log_file_handle: write_to_log(err_msg, is_error=True)
        raise ValueError(err_msg)

    if not isinstance(data, list) or len(data) < aggregate + 1:
        msg = f"system_info must be a JSON array with at least {aggregate + 1} entries ({aggregate} monomers + 1 solvent)."
        print(f"ERROR: {msg}", file=sys.stderr)
        if log_file_handle: write_to_log(msg, is_error=True)
        raise ValueError(msg)
    
    monomers_data: List[Dict[str, Any]] = data[:aggregate]
    solvent_data: Dict[str, Any] = data[aggregate] 

    # Validate monomers
    for i, monomer in enumerate(monomers_data):
        monomer_id_for_log = monomer.get(JSON_KEY_NAME, f"monomer_at_index_{i}")
        if not isinstance(monomer, dict):
            raise ValueError(f"Monomer entry at index {i} is not a valid JSON object.")
        
        required_keys_monomer: Dict[str, Any] = { 
            JSON_KEY_NAME: str, 
            JSON_KEY_NATOMS: int, 
            JSON_KEY_CHARGE: (int, float), 
            JSON_KEY_SPIN_MULT: int
        }
        for key, expected_type in required_keys_monomer.items():
            if key not in monomer:
                raise ValueError(f"Monomer '{monomer_id_for_log}' is missing required key: '{key}'.")
            if not isinstance(monomer[key], expected_type): # type: ignore
                raise ValueError(f"Monomer '{monomer_id_for_log}' key '{key}' has incorrect type. Expected {expected_type}, got {type(monomer[key])}.")
        
        if monomer[JSON_KEY_NATOMS] <= 0:
            raise ValueError(f"Monomer '{monomer_id_for_log}' key '{JSON_KEY_NATOMS}' must be a positive integer.")
        if monomer[JSON_KEY_SPIN_MULT] < 1:
            raise ValueError(f"Monomer '{monomer_id_for_log}' key '{JSON_KEY_SPIN_MULT}' must be a positive integer.")
        if JSON_KEY_MOL_FORMULA not in monomer: 
            warn_msg = f"Monomer '{monomer_id_for_log}' is missing optional key: '{JSON_KEY_MOL_FORMULA}'."
            print(f"WARNING: {warn_msg}")
            if log_file_handle: write_to_log(warn_msg, is_warning=True)


    # Validate solvent
    if not isinstance(solvent_data, dict):
        raise ValueError("Solvent entry in system_info is not a valid JSON object.")
    
    required_keys_solvent: Dict[str, Any] = { 
        JSON_KEY_MOL_FORMULA: str,
        JSON_KEY_NATOMS: int,
        JSON_KEY_CHARGES_ARRAY: list
    }
    solvent_id_for_log = solvent_data.get(JSON_KEY_NAME, "solvent") 
    solvent_data[JSON_KEY_NAME] = solvent_id_for_log # Ensure name key exists for logging

    for key, expected_type in required_keys_solvent.items():
        if key not in solvent_data:
            raise ValueError(f"Solvent entry '{solvent_id_for_log}' is missing required key: '{key}'.")
        if not isinstance(solvent_data[key], expected_type): # type: ignore
            raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{key}' has incorrect type. Expected {expected_type}, got {type(solvent_data[key])}.")

    if solvent_data[JSON_KEY_NATOMS] <= 0:
        raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{JSON_KEY_NATOMS}' must be a positive integer.")
    if not solvent_data[JSON_KEY_CHARGES_ARRAY]: # Must have at least one charge entry if nAtoms > 0
        if solvent_data[JSON_KEY_NATOMS] > 0 : # Only raise if atoms are expected
             raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{JSON_KEY_CHARGES_ARRAY}' must be a non-empty list when nAtoms > 0.")
    if len(solvent_data[JSON_KEY_CHARGES_ARRAY]) != solvent_data[JSON_KEY_NATOMS]:
        raise ValueError(f"Solvent entry '{solvent_id_for_log}': Number of entries in '{JSON_KEY_CHARGES_ARRAY}' ({len(solvent_data[JSON_KEY_CHARGES_ARRAY])}) "
                         f"does not match '{JSON_KEY_NATOMS}' ({solvent_data[JSON_KEY_NATOMS]}).")

    for charge_entry_idx, charge_entry in enumerate(solvent_data[JSON_KEY_CHARGES_ARRAY]):
        if not isinstance(charge_entry, dict):
            raise ValueError(f"Solvent '{solvent_id_for_log}', charge entry at index {charge_entry_idx} in '{JSON_KEY_CHARGES_ARRAY}' is not a valid JSON object.")
        if JSON_KEY_ELEMENT not in charge_entry or not isinstance(charge_entry[JSON_KEY_ELEMENT], str):
            raise ValueError(f"Solvent '{solvent_id_for_log}', charge entry at index {charge_entry_idx}: missing or invalid '{JSON_KEY_ELEMENT}'.")
        if JSON_KEY_CHARGE not in charge_entry or not isinstance(charge_entry[JSON_KEY_CHARGE], (int, float)):
            raise ValueError(f"Solvent '{solvent_id_for_log}', charge entry at index {charge_entry_idx}: missing or invalid '{JSON_KEY_CHARGE}'.")
    
    if log_file_handle: write_to_log("System info loaded and validated successfully.")
    return monomers_data, solvent_data


def parse_formula(fmt: str) -> Dict[str, int]:
    """
    Parse molecular formula into element counts. e.g., "H2O" -> {'H':2, 'O':1}.
    """
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', fmt)
    counts: Dict[str, int] = {}
    for elem, num_str in tokens:
        counts[elem] = int(num_str) if num_str else 1
    return counts


def group_qm_molecules(atoms: AtomListType, coords: CoordType, fmt_counts: Dict[str, int]) -> List[SolventGroupType]:
    """
    Group solvent atoms into molecules by known formula counts.
    Returns list of (atom_symbols_for_molecule, coords_array_for_molecule).
    """
    size = sum(fmt_counts.values()) 
    if not atoms: 
        return []
    if size == 0 : # Avoid division by zero if formula is empty (e.g. "None" or "")
        msg = f"Cannot group solvent molecules: formula '{fmt_counts}' results in zero atoms per molecule."
        print(f"ERROR: {msg}", file=sys.stderr)
        if log_file_handle: write_to_log(msg, is_error=True)
        raise ValueError(msg)
    if len(atoms) % size != 0:
        msg = f"Total solvent atom count ({len(atoms)}) is not divisible by formula size ({size}). Ensure correct solvent definition and XYZ content."
        print(f"ERROR: {msg}", file=sys.stderr)
        if log_file_handle: write_to_log(msg, is_error=True)
        raise ValueError(msg)
    
    groups: List[SolventGroupType] = []
    for i in range(0, len(atoms), size):
        grp_atoms: AtomListType = atoms[i:i+size]
        grp_coords: CoordType = coords[i:i+size]
        groups.append((grp_atoms, grp_coords))
    return groups


def is_within_radius(core_coords_list: List[CoordType], mol_coords_array: CoordType, radius: float) -> bool:
    """
    Check if any atom in mol_coords_array lies within radius of any core_coords_array in core_coords_list.
    """
    if radius < 0: return False # Negative radius means no selection
    for core_coord_set in core_coords_list:
        if core_coord_set.size == 0 or mol_coords_array.size == 0:
            continue
        # Efficient distance calculation:
        # Expand dims for broadcasting: core (N,1,3), mol (1,M,3) -> diff (N,M,3)
        diff = core_coord_set[:, np.newaxis, :] - mol_coords_array[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2)) # Norm calculation
        dists_sq = np.sum(diff**2, axis=2) # Compare squared distances to avoid sqrt
        if np.any(dists_sq < radius**2):
            return True
    return False


def flatten_groups(groups_of_molecules: List[SolventGroupType]) -> Tuple[AtomListType, CoordType]:
    """
    Flatten grouped molecules to a single atom list & a single coord array.
    """
    flat_atoms: AtomListType = []
    flat_coords_list: List[CoordType] = []
    for atoms_in_mol, coords_in_mol in groups_of_molecules:
        flat_atoms.extend(atoms_in_mol)
        if coords_in_mol.size > 0: 
            flat_coords_list.append(coords_in_mol)
    
    if not flat_coords_list: 
        return flat_atoms, np.array([]) # Return empty array with proper shape for vstack if flat_atoms is also empty
    return flat_atoms, np.vstack(flat_coords_list)


def write_xyz(path: str, atoms: Union[AtomListType, ChargeListType], coords: CoordType, comment: str = "") -> None:
    """
    Export atoms and coords to an XYZ file.
    The 'atoms' list can contain strings (element symbols) or floats (charges for MM solvent XYZ).
    """
    coords_array = np.array(coords)
    if coords_array.ndim == 1 and coords_array.shape[0] == 3 and len(atoms) == 1: 
        coords_array = coords_array.reshape(1, 3)
    elif coords_array.ndim == 1 and len(atoms) > 1 and coords_array.size != 3 * len(atoms) : 
        # If it's a flat 1D array for multiple atoms, it should be N*3
        raise ValueError(f"Coordinates array for multiple atoms has unexpected 1D shape: {coords_array.shape}, expected ({len(atoms)*3},)")
    elif coords_array.ndim == 1 and len(atoms) > 1 and coords_array.size == 3 * len(atoms) :
        coords_array = coords_array.reshape(len(atoms), 3) # Reshape if it's a flat list of coords
    elif coords_array.size == 0 and len(atoms) > 0: # Check if atoms exist but coords are genuinely empty
        if not (len(atoms) == 1 and atoms[0] == ""): # Allow a single empty atom string with no coords (though unusual)
             raise ValueError("Cannot write XYZ: Atoms provided but coordinates are empty.")
    elif coords_array.ndim == 2 and coords_array.shape[0] != len(atoms):
         raise ValueError(f"Mismatch between number of atoms ({len(atoms)}) and number of coordinate rows ({coords_array.shape[0]}).")
    elif coords_array.ndim > 2 :
        raise ValueError(f"Coordinates array has too many dimensions: {coords_array.ndim}")


    try:
        with open(path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for atom_symbol_or_charge, xyz_row in zip(atoms, coords_array):
                if isinstance(atom_symbol_or_charge, (float, np.floating)): # Include numpy floats
                    atom_col_str = f"{atom_symbol_or_charge:<12.5f}" 
                else:
                    atom_col_str = f"{str(atom_symbol_or_charge):<3}" 
                f.write(f"{atom_col_str} {xyz_row[0]:12.5f} {xyz_row[1]:12.5f} {xyz_row[2]:12.5f}\n")
    except IOError as e:
        err_msg = f"Could not write XYZ file {path}: {e}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        if log_file_handle: write_to_log(err_msg, is_error=True)
        raise

def localize_solvent_and_prepare_regions(
    main_xyz_input_filepath: str, 
    system_info_path: str, 
    aggregate: int, 
    qm_radius: float,
    qm_aggregate_coords_override: Optional[CoordType] = None 
) -> Tuple[List[Tuple[AtomListType, CoordType]], Tuple[AtomListType, CoordType], List[SolventGroupType], Dict[str, bool], CoordType, List[int]]:
    """
    Processes input XYZ to define QM regions for monomers and aggregate.
    If qm_aggregate_coords_override is provided, these coordinates are used for the core.
    Solvent is always taken from main_xyz_input_filepath.
    """
    monomers_meta, solvent_meta = load_system_info(system_info_path, aggregate)
    n_atoms_per_monomer_list: List[int] = [m[JSON_KEY_NATOMS] for m in monomers_meta]
    total_core_atom_count_from_json = sum(n_atoms_per_monomer_list)
    
    loaded_main_coords_list: List[List[float]] = []
    loaded_main_atoms_list: AtomListType = []
    try:
        with open(main_xyz_input_filepath, 'r') as f_xyz:
            lines = f_xyz.readlines()
    except FileNotFoundError:
        err_msg = f"Input XYZ file not found: {main_xyz_input_filepath}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        if log_file_handle: write_to_log(err_msg, is_error=True)
        raise

    if len(lines) < 2: raise ValueError(f"XYZ file {main_xyz_input_filepath} is too short.")
    try: num_atoms_in_main_file = int(lines[0].strip())
    except ValueError: raise ValueError(f"First line of XYZ {main_xyz_input_filepath} must be atom count.")
    if len(lines) < num_atoms_in_main_file + 2: raise ValueError(f"XYZ {main_xyz_input_filepath} has fewer lines ({len(lines)}) than expected based on atom count ({num_atoms_in_main_file + 2}).")
    
    for ln_idx, ln_content in enumerate(lines[2 : num_atoms_in_main_file + 2]):
        parts = ln_content.split()
        if len(parts) >= 4:
            loaded_main_atoms_list.append(parts[0])
            try: loaded_main_coords_list.append([float(p) for p in parts[1:4]])
            except ValueError: raise ValueError(f"Malformed coord in {main_xyz_input_filepath} line {ln_idx+3}: {ln_content.strip()}")
        else: raise ValueError(f"Malformed line in {main_xyz_input_filepath} line {ln_idx+3}: {ln_content.strip()}")

    main_atoms_all_array = np.array(loaded_main_atoms_list, dtype=str)
    main_coords_all_array = np.array(loaded_main_coords_list)

    core_atoms_all_list: AtomListType
    core_coords_all_for_dist_calc: CoordType # These are the ones used for distance calculations
    core_coords_to_use_for_qm: CoordType # These are the ones that go into the QM region output (can be overridden)

    solvent_atoms_from_xyz_list: AtomListType
    solvent_coords_from_xyz_array: CoordType

    if qm_aggregate_coords_override is not None:
        if qm_aggregate_coords_override.shape[0] != total_core_atom_count_from_json:
            msg = (f"Atom count in --qm_aggregate_xyz ({qm_aggregate_coords_override.shape[0]}) "
                   f"does not match total core atoms from system_info ({total_core_atom_count_from_json}).")
            print(f"ERROR: {msg}", file=sys.stderr)
            if log_file_handle: write_to_log(msg, is_error=True)
            raise ValueError(msg)
        
        # Atom types for the core are still taken from the main XYZ file
        core_atoms_all_list = main_atoms_all_array[:total_core_atom_count_from_json].tolist() 
        core_coords_to_use_for_qm = qm_aggregate_coords_override # Use override for output
        # For distance calculations (e.g. solvent shell), it's often better to use the *actual* positions
        # of the core from the main XYZ, unless the override is meant to also redefine the solvent shell center.
        # Assuming the override is just for the QM part, and solvent shell is relative to original core.
        # If qm_aggregate_coords_override should ALSO define the center for solvent selection, then use it here too.
        # For now, let's assume qm_radius is relative to the *original* core positions for consistency.
        # If it should be relative to the *new* core positions, then:
        # core_coords_all_for_dist_calc = qm_aggregate_coords_override
        core_coords_all_for_dist_calc = main_coords_all_array[:total_core_atom_count_from_json] # Sticking to original for solvent finding
        
        if num_atoms_in_main_file < total_core_atom_count_from_json:
            msg = (f"Main input XYZ file '{main_xyz_input_filepath}' has fewer atoms ({num_atoms_in_main_file}) "
                   f"than expected for the core ({total_core_atom_count_from_json}) when using --qm_aggregate_xyz. "
                   f"It must still contain at least the core atoms for atom typing and solvent indexing.")
            print(f"ERROR: {msg}", file=sys.stderr)
            if log_file_handle: write_to_log(msg, is_error=True)
            raise ValueError(msg)
            
        solvent_atoms_from_xyz_list = main_atoms_all_array[total_core_atom_count_from_json:].tolist()
        solvent_coords_from_xyz_array = main_coords_all_array[total_core_atom_count_from_json:]
        if log_file_handle: write_to_log(f"Using core coordinates for QM region from --qm_aggregate_xyz. Atom types for core and all solvent taken from {main_xyz_input_filepath}. Solvent localization relative to core from {main_xyz_input_filepath}.")

    else: # No override
        core_atoms_all_list = main_atoms_all_array[:total_core_atom_count_from_json].tolist()
        core_coords_all_for_dist_calc = main_coords_all_array[:total_core_atom_count_from_json]
        core_coords_to_use_for_qm = core_coords_all_for_dist_calc # They are the same
        solvent_atoms_from_xyz_list = main_atoms_all_array[total_core_atom_count_from_json:].tolist()
        solvent_coords_from_xyz_array = main_coords_all_array[total_core_atom_count_from_json:]

    solvent_formula_counts = parse_formula(solvent_meta[JSON_KEY_MOL_FORMULA])
    all_solvent_groups_list: List[SolventGroupType] = group_qm_molecules(solvent_atoms_from_xyz_list, solvent_coords_from_xyz_array, solvent_formula_counts)
    
    # Use core_coords_all_for_dist_calc for finding solvent around the aggregate
    aggregate_qm_solvent_groups: List[SolventGroupType] = [g for g in all_solvent_groups_list if is_within_radius([core_coords_all_for_dist_calc], g[1], qm_radius)]
    aggregate_qm_solvent_atoms_list, aggregate_qm_solvent_coords_array = flatten_groups(aggregate_qm_solvent_groups)
    
    aggregate_qm_group_ids = set(map(id, aggregate_qm_solvent_groups)) # Efficiently track selected groups
    non_qm_solvent_groups: List[SolventGroupType] = [g for g in all_solvent_groups_list if id(g) not in aggregate_qm_group_ids]

    monomer_localized_qm_solvent_groups_list: List[List[SolventGroupType]] = [] 
    current_core_idx_dist = 0 # For iterating through core_coords_all_for_dist_calc
    for num_atoms_mono in n_atoms_per_monomer_list:
        # Use coords for distance calculation here
        monomer_core_coords_for_dist = core_coords_all_for_dist_calc[current_core_idx_dist : current_core_idx_dist + num_atoms_mono]
        localized_groups = [g for g in all_solvent_groups_list if is_within_radius([monomer_core_coords_for_dist], g[1], qm_radius)]
        monomer_localized_qm_solvent_groups_list.append(localized_groups)
        current_core_idx_dist += num_atoms_mono
        
    seen_qm_solvent_ids_for_monomers: set = set() # To ensure solvent molecules are not double-counted in monomer QM regions
    monomer_unique_qm_solvent_data_list: List[Tuple[AtomListType, CoordType]] = [] 
    qm_solvent_flags: Dict[str, bool] = {} # To track if QM solvent was actually added

    for i in range(len(monomers_meta)):
        unique_groups_for_this_mono = []
        for group in monomer_localized_qm_solvent_groups_list[i]:
            if id(group) not in seen_qm_solvent_ids_for_monomers:
                unique_groups_for_this_mono.append(group)
                seen_qm_solvent_ids_for_monomers.add(id(group)) # Mark as seen for subsequent monomers
        added_qm_atoms, added_qm_coords = flatten_groups(unique_groups_for_this_mono)
        monomer_unique_qm_solvent_data_list.append((added_qm_atoms, added_qm_coords))
        qm_solvent_flags[f'monomer_{i}_has_added_qm_solvent'] = bool(added_qm_atoms) # True if list is not empty

    qm_solvent_flags['aggregate_has_added_qm_solvent'] = bool(aggregate_qm_solvent_atoms_list)

    # Prepare final QM regions using core_coords_to_use_for_qm
    aggregate_final_qm_atoms = core_atoms_all_list + aggregate_qm_solvent_atoms_list
    aggregate_final_qm_coords = np.vstack((core_coords_to_use_for_qm, aggregate_qm_solvent_coords_array)) if aggregate_qm_solvent_coords_array.size > 0 else core_coords_to_use_for_qm
    aggregate_qm_region_for_gauss: Tuple[AtomListType, CoordType] = (aggregate_final_qm_atoms, aggregate_final_qm_coords)

    monomer_qm_regions_for_gauss: List[Tuple[AtomListType, CoordType]] = []
    current_core_idx_qm = 0 # For iterating through core_coords_to_use_for_qm
    for i in range(len(monomers_meta)):
        num_atoms_in_this_mono = n_atoms_per_monomer_list[i]
        mono_core_atoms = core_atoms_all_list[current_core_idx_qm : current_core_idx_qm + num_atoms_in_this_mono]
        mono_core_coords_for_qm = core_coords_to_use_for_qm[current_core_idx_qm : current_core_idx_qm + num_atoms_in_this_mono]
        
        added_qm_atoms, added_qm_coords = monomer_unique_qm_solvent_data_list[i] # These are already from all_solvent_groups_list (original coords)
        
        final_mono_atoms = mono_core_atoms + added_qm_atoms
        final_mono_coords = np.vstack((mono_core_coords_for_qm, added_qm_coords)) if added_qm_coords.size > 0 else mono_core_coords_for_qm
        monomer_qm_regions_for_gauss.append((final_mono_atoms, final_mono_coords))
        current_core_idx_qm += num_atoms_in_this_mono
            
    # Return core_coords_all_for_dist_calc because this is what's used for centroid distance calculations
    return (monomer_qm_regions_for_gauss, aggregate_qm_region_for_gauss, 
            non_qm_solvent_groups, qm_solvent_flags, 
            core_coords_all_for_dist_calc, n_atoms_per_monomer_list)


def load_monomer_charges_from_file(charge_file_path: str) -> List[MMChargeTupleType]:
    """
    Load MM charges from a file for monomer embedding. Format: charge x y z.
    Returns list of (x, y, z, charge) tuples for Gaussian compatibility.
    """
    charges_data: List[MMChargeTupleType] = []
    try:
        with open(charge_file_path, 'r') as f:
            for line_num, line_content in enumerate(f):
                parts = line_content.strip().split()
                if len(parts) == 4:
                    try:
                        # File format is charge x y z
                        charge_val = float(parts[0])
                        x_val, y_val, z_val = float(parts[1]), float(parts[2]), float(parts[3])
                        charges_data.append((x_val, y_val, z_val, charge_val)) # Store as (x,y,z,q)
                    except ValueError:
                        warn_msg = f"Non-numeric value in charge file {charge_file_path} on line {line_num + 1}: {line_content.strip()}"
                        print(f"WARNING: {warn_msg}")
                        if log_file_handle: write_to_log(warn_msg, is_warning=True)
                elif parts: # If line is not empty and not 4 parts
                    warn_msg = f"Malformed line in charge file {charge_file_path} on line {line_num + 1} (expected 4 values, got {len(parts)}): {line_content.strip()}"
                    print(f"WARNING: {warn_msg}")
                    if log_file_handle: write_to_log(warn_msg, is_warning=True)
    except FileNotFoundError:
        # This should ideally be caught earlier or handled by parser.error if file is mandatory
        warn_msg = f"MM Monomer charge file not found: {charge_file_path}. Returning empty charges for this file."
        print(f"WARNING: {warn_msg}")
        if log_file_handle: write_to_log(warn_msg, is_warning=True)
    return charges_data

def assign_charges_to_solvent_molecules(solvent_groups: List[SolventGroupType], solvent_metadata_entry: Dict[str, Any]) -> List[MMChargeTupleType]:
    """
    Assigns charges to solvent molecules based on the solvent_metadata_entry.
    Returns list of (x, y, z, charge) tuples for Gaussian compatibility.
    """
    mm_solvent_charges_list: List[MMChargeTupleType] = []
    defined_charges_per_mol: List[Dict[str, Any]] = solvent_metadata_entry[JSON_KEY_CHARGES_ARRAY]
    num_atoms_per_solvent_mol_defined: int = solvent_metadata_entry[JSON_KEY_NATOMS]

    if len(defined_charges_per_mol) != num_atoms_per_solvent_mol_defined:
        msg = (f"Mismatch in solvent definition: '{JSON_KEY_NATOMS}' ({num_atoms_per_solvent_mol_defined}) "
               f"does not match entries in '{JSON_KEY_CHARGES_ARRAY}' ({len(defined_charges_per_mol)}) "
               f"for solvent '{solvent_metadata_entry.get(JSON_KEY_NAME, 'Unnamed Solvent')}'.")
        print(f"ERROR: {msg}", file=sys.stderr)
        if log_file_handle: write_to_log(msg, is_error=True)
        raise ValueError(msg)

    for mol_idx, (atom_symbols_in_molecule, coords_for_molecule) in enumerate(solvent_groups):
        if len(atom_symbols_in_molecule) != num_atoms_per_solvent_mol_defined:
            warn_msg = (f"Solvent molecule {mol_idx+1} (auto-detected for MM) has {len(atom_symbols_in_molecule)} atoms, "
                        f"but system_info defines {num_atoms_per_solvent_mol_defined} for solvent "
                        f"'{solvent_metadata_entry.get(JSON_KEY_NAME, 'Unnamed Solvent')}'. Skipping this molecule for MM charges.")
            print(f"WARNING: {warn_msg}")
            if log_file_handle: write_to_log(warn_msg, is_warning=True)
            continue

        for atom_idx_in_mol in range(len(atom_symbols_in_molecule)):
            atom_symbol_from_xyz = atom_symbols_in_molecule[atom_idx_in_mol]
            atom_coord = coords_for_molecule[atom_idx_in_mol]
            
            # Get charge based on position in the defined solvent molecule template
            charge_info_for_this_atom_pos = defined_charges_per_mol[atom_idx_in_mol]
            charge_value: float = charge_info_for_this_atom_pos[JSON_KEY_CHARGE]
            defined_element_for_pos: str = charge_info_for_this_atom_pos[JSON_KEY_ELEMENT]

            if atom_symbol_from_xyz != defined_element_for_pos:
                warn_msg = (f"Element mismatch for atom {atom_idx_in_mol+1} in auto-detected MM solvent molecule {mol_idx+1}. "
                            f"XYZ has '{atom_symbol_from_xyz}', system_info expects '{defined_element_for_pos}' at this position. "
                            f"Using charge defined for '{defined_element_for_pos}' ({charge_value}).")
                print(f"WARNING: {warn_msg}")
                if log_file_handle: write_to_log(warn_msg, is_warning=True)
            
            mm_solvent_charges_list.append((atom_coord[0], atom_coord[1], atom_coord[2], charge_value)) # (x,y,z,q)
            
    return mm_solvent_charges_list


def get_solvent_descriptor_suffix(entity_has_added_qm_solvent: bool, system_has_mm_solvent: bool) -> str:
    """Determines the solvent descriptor part of the .com filename (without .com extension)."""
    if entity_has_added_qm_solvent and system_has_mm_solvent:
        return "_qmsol_mmsol" # More descriptive
    elif entity_has_added_qm_solvent:
        return "_qmsol"
    elif system_has_mm_solvent:
        return "_mmsol"
    else:
        return "" 


def write_com_file( 
    path: str, keywords: List[str], title: str, charge: Union[int, float], spin: int,
    atoms_list: AtomListType, coords_array: CoordType, 
    mm_charges_list: Optional[List[MMChargeTupleType]] = None, 
    fragment_definitions: Optional[List[Tuple[AtomListType, CoordType, int, int]]] = None
) -> None:   
    """
    Writes a Gaussian .com input file.
    mm_charges_list format: (x, y, z, q).
    fragment_definitions: [(atoms1, coords1, chg1, spin1), (atoms2, coords2, chg2, spin2)]
    """
    final_keywords = list(keywords) # Make a copy
    if mm_charges_list:
        # Check if '# charge' or 'charge' by itself is in keywords for external charges
        charge_keyword_present = any(re.fullmatch(r"#.*charge", kw.lower().strip()) or kw.lower().strip() == "charge" for kw in final_keywords)
        if not charge_keyword_present:
            # Find first line starting with # to insert 'charge' keyword, or append if none.
            # This ensures 'charge' is part of the route section.
            inserted = False
            for i, kw_line in enumerate(final_keywords):
                if kw_line.strip().startswith("#"):
                    final_keywords.insert(i + 1, "charge") # Add 'charge' on the next line
                    inserted = True
                    break
            if not inserted: # No line starting with #, so just append.
                 final_keywords.append("charge")


    try:
        with open(path, 'w') as f:
            # Ensure %chk is first, then keywords
            f.write(f"%chk={os.path.splitext(os.path.basename(path))[0]}.chk\n")
            for kw_line in final_keywords:
                f.write(kw_line + "\n")
            f.write("\n" + title + "\n\n") # Ensure blank line after keywords
            
            if fragment_definitions: 
                # Overall charge/spin, then per-fragment charge/spin, all on one line
                f.write(f"{charge} {spin}") # Overall
                for _, _, chg_frag, spin_frag in fragment_definitions:
                    f.write(f" {chg_frag} {spin_frag}") # Fragment specific
                f.write("\n") # End of charge/spin line
                
                for frag_idx, (frag_atoms, frag_coords, _, _) in enumerate(fragment_definitions):
                    for atom_sym, xyz_coords in zip(frag_atoms, frag_coords):
                        f.write(f" {atom_sym}(Fragment={frag_idx+1}) {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")
            else: 
                f.write(f"{charge} {spin}\n")
                for atom_sym, xyz_coords in zip(atoms_list, coords_array):
                    f.write(f" {atom_sym} {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")
            
            if mm_charges_list: 
                f.write("\n") # Blank line before MM charges
                for x_mm, y_mm, z_mm, q_mm in mm_charges_list: 
                    f.write(f" {x_mm:14.5f} {y_mm:12.5f} {z_mm:12.5f} {q_mm:12.5f}\n") # x,y,z,q format
            
            f.write("\n") # Ensure a final blank line
    except IOError as e:
        err_msg = f"Could not write .com file {path}: {e}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        if log_file_handle: write_to_log(err_msg, is_error=True)
        raise


def load_keywords_from_file(path: str) -> List[str]:
    """
    Load Gaussian route keywords from a text file.
    """
    try:
        with open(path) as f:
            # Strip whitespace from each line and filter out empty lines
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        err_msg = f"Keywords file not found: {path}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        if log_file_handle: write_to_log(err_msg, is_error=True)
        raise

def calculate_centroid_distance_between_first_two(core_coords_all_array: CoordType, n_atoms_per_monomer_list: List[int]) -> Optional[float]:
    """
    Calculates the distance between the centroids of the first two monomers' core regions.
    Returns the distance as a float, or None if not applicable or if a monomer has no atoms/coords.
    """
    if len(n_atoms_per_monomer_list) < 2:
        return None 

    n_atoms_m1 = n_atoms_per_monomer_list[0]
    if n_atoms_m1 == 0: 
        if log_file_handle: write_to_log("Cannot calculate centroid distance: Monomer 1 has 0 atoms defined in system_info.")
        return None 
    # Check if core_coords_all_array actually has enough coordinates for M1
    if core_coords_all_array.shape[0] < n_atoms_m1:
        if log_file_handle: write_to_log(f"Cannot calculate centroid distance for M1: Core coords array has {core_coords_all_array.shape[0]} atoms, but M1 expects {n_atoms_m1}.")
        return None
    coords_m1 = core_coords_all_array[0:n_atoms_m1]
    if coords_m1.shape[0] == 0: # Should be caught by n_atoms_m1 == 0, but as a safeguard for array slicing
        if log_file_handle: write_to_log("Cannot calculate centroid distance: Monomer 1 coordinate slice is empty.")
        return None 
    centroid_m1 = np.mean(coords_m1, axis=0)

    n_atoms_m2 = n_atoms_per_monomer_list[1]
    if n_atoms_m2 == 0: 
        if log_file_handle: write_to_log("Cannot calculate centroid distance: Monomer 2 has 0 atoms defined in system_info.")
        return None 
    start_idx_m2 = n_atoms_m1
    end_idx_m2 = n_atoms_m1 + n_atoms_m2
    # Check if core_coords_all_array actually has enough coordinates for M2
    if core_coords_all_array.shape[0] < end_idx_m2:
        if log_file_handle: write_to_log(f"Cannot calculate centroid distance for M2: Core coords array has {core_coords_all_array.shape[0]} atoms, but M1+M2 expect {end_idx_m2}.")
        return None
    coords_m2 = core_coords_all_array[start_idx_m2:end_idx_m2]
    if coords_m2.shape[0] == 0: # Safeguard
        if log_file_handle: write_to_log("Cannot calculate centroid distance: Monomer 2 coordinate slice is empty.")
        return None 
    centroid_m2 = np.mean(coords_m2, axis=0)
    
    distance: float = np.linalg.norm(centroid_m1 - centroid_m2)
    return distance

def main() -> None:
    global log_file_handle
    # Default log filename moved here for clarity, can be overridden by args
    current_default_log_filename = "giqpy_run.log"


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
    parser.add_argument('--mm_solvent', type=str, nargs='?', const=AUTO_MM_SOLVENT_TRIGGER, default=None,
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
        log_file_handle = open(effective_log_filename, 'w')
        log_file_handle.write(f"GIQPy Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
        log_file_handle.write(f"Arguments: {vars(args)}\n\n")
        log_file_handle.flush()
    except IOError as e:
        print(f"CRITICAL ERROR: Could not open log file {effective_log_filename} for writing: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)


    # --- Argument Validation ---
    user_intends_com_output = args.output_com is not None
    user_intends_xyz_output = args.output_xyz is not None and args.output_xyz != 'none'

    if not user_intends_com_output and not user_intends_xyz_output:
        err_msg = "No output format selected. You must specify --output_com and/or --output_xyz."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        parser.error(err_msg)

    if args.traj_xyz and args.frames is None:
        err_msg = "--frames is required when --traj_xyz is used."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        parser.error(err_msg) 
    if args.aggregate < 1:
        err_msg = "--aggregate must be at least 1."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        parser.error(err_msg)
    if args.eetg:
        if args.aggregate != 2:
            err_msg = "--eetg calculations require --aggregate=2 (a dimer system)."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
        if not user_intends_com_output:
            err_msg = "--eetg calculations require .com file output. Please specify --output_com."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
            
    if user_intends_com_output and args.output_com == 'monomer' and args.eetg:
        warn_msg = ("Configuration: --output_com set to 'monomer' with --eetg. "
                    "EETG is an aggregate/dimer property. An EETG file for the dimer will be generated if --aggregate=2, "
                    "but individual monomer .com files will also be generated as per --output_com monomer.")
        print(f"INFO: {warn_msg}") # Changed to INFO as it's a clarification
        write_to_log(warn_msg, is_warning=False) # Not strictly a warning of misconfiguration

    # Gaussian keywords handling
    gaussian_keywords_for_calcs: Optional[List[str]] = None # Variable to hold loaded keywords
    if user_intends_com_output:
        if not args.gauss_keywords:
            err_msg = "A Gaussian keywords file must be provided via --gauss_keywords when --output_com is specified."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            parser.error(err_msg)
        
        if not os.path.exists(args.gauss_keywords) or os.path.getsize(args.gauss_keywords) == 0:
            err_msg = f"Gaussian keywords file '{args.gauss_keywords}' not found or is empty."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            # parser.error(err_msg) # This would exit before log flushes sometimes
            raise FileNotFoundError(err_msg) # More standard for file issues
        gaussian_keywords_for_calcs = load_keywords_from_file(args.gauss_keywords)
    elif args.gauss_keywords: # Keywords provided, but .com output not requested
        warn_msg = ("--gauss_keywords was provided, but --output_com was not specified. "
                    "The keywords file will be ignored as no .com files are being generated.")
        print(f"WARNING: {warn_msg}")
        write_to_log(warn_msg, is_warning=True)

    # --- End of Argument Validation ---

    # Load qm_aggregate_xyz if provided
    qm_aggregate_coords_override: Optional[CoordType] = None
    if args.qm_aggregate_xyz:
        write_to_log(f"Loading user-defined aggregate coordinates from: {args.qm_aggregate_xyz}")
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
            write_to_log(f"Successfully loaded {qm_aggregate_coords_override.shape[0]} aggregate coordinates from {args.qm_aggregate_xyz}.")
        except Exception as e: # Catch generic exceptions during file load/parse
            err_msg = f"Failed to load or parse --qm_aggregate_xyz file '{args.qm_aggregate_xyz}': {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            sys.exit(1) # Critical error, cannot proceed


    base_output_dir: str = os.getcwd() 

    processed_frames_list: List[Tuple[Optional[str], str, str]] = split_frames(args.single_xyz, args.traj_xyz, args.frames, base_output_dir)
    total_frames_to_process = len(processed_frames_list)
    
    if total_frames_to_process == 0: # This implies an issue with split_frames or input files
        err_msg = f"No frames were found or could be processed from the input XYZ source."
        if args.traj_xyz:
             err_msg = f"No frames were found or could be processed from trajectory: {args.traj_xyz}"
        elif args.single_xyz:
             err_msg = f"Could not process single XYZ file: {args.single_xyz}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        sys.exit(1)


    monomers_metadata_list: List[Dict[str, Any]]
    solvent_metadata: Dict[str, Any]
    try:
        monomers_metadata_list, solvent_metadata = load_system_info(args.system_info, args.aggregate)
    except Exception as e: # Catch errors from load_system_info (FileNotFound, ValueError)
        print(f"CRITICAL ERROR loading system_info: {e}", file=sys.stderr)
        write_to_log(f"CRITICAL ERROR loading system_info: {e}", is_error=True)
        sys.exit(1)

    solvent_name_str = solvent_metadata.get(JSON_KEY_NAME, "solvent")


    if len(monomers_metadata_list) != args.aggregate:
        # This check might be redundant if load_system_info already validates based on aggregate count
        msg = (f"Number of monomer entries in system_info ({len(monomers_metadata_list)}) "
               f"does not match --aggregate ({args.aggregate}).")
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
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
        write_to_log(f"\n\n{frame_processing_message}")
        
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
                localize_solvent_and_prepare_regions(
                    temp_xyz_input_filepath,
                    args.system_info,
                    args.aggregate,
                    args.qm_solvent, # qm_solvent radius
                    qm_aggregate_coords_override=qm_aggregate_coords_override 
                )
        except Exception as e:
            err_msg = f"Error during solvent localization/region preparation for frame {frame_id_str if frame_id_str else 'single'}: {e}"
            print(f"\nERROR: {err_msg}", file=sys.stderr) # Newline if progress bar was on previous line
            write_to_log(err_msg, is_error=True)
            if args.traj_xyz:
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
        if args.aggregate >= 2:
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
        mm_embedding_charges_for_each_monomer: List[List[MMChargeTupleType]] = [[] for _ in range(args.aggregate)] 
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
                
                if args.aggregate == 1:
                    write_to_log("Note: --mm_monomer 0 specified with --aggregate=1. No other monomers to embed with zero charges.", is_warning=True)
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
                    write_to_log(f"Loading MM charges for inter-monomer embedding from {args.aggregate} specified file(s).")
                    all_monomer_explicit_charges_xyz_q = [load_monomer_charges_from_file(f) for f in args.mm_monomer]
                    
                    if args.aggregate == 1:
                         warn_msg = ("--mm_monomer provided with 1 file for --aggregate=1. "
                                     "This charge file will be loaded but not used for inter-monomer embedding as there are no 'other' monomers.")
                         print(f"\nWARNING: {warn_msg}") # Ensure newline
                         write_to_log(warn_msg, is_warning=True)
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
                    write_to_log(warn_msg, is_warning=True)


        # --- XYZ File Generation Block ---
        if user_intends_xyz_output:
            write_to_log("Writing detailed QM region and MM XYZ files (if applicable)...") 
            
            # QM Region XYZ files (filenames and comments are UNCHANGED)
            if actual_gen_aggregate_xyz:
                agg_qm_atoms_xyz, agg_qm_coords_xyz = aggregate_qm_region
                qm_sol_desc_agg = f" + qm {solvent_name_str}" if qm_solvent_flags.get('aggregate_has_added_qm_solvent') else ""
                # Original comment from user's script version
                xyz_comment_for_aggregate_qm = f"{aggregate_xyz_comment_base} qm{qm_sol_desc_agg}" # Corrected: was 'qm {qm_sol_desc_agg}'
                write_xyz(os.path.join(current_frame_output_dir, f'{combined_system_term}_qm.xyz'), 
                          agg_qm_atoms_xyz, agg_qm_coords_xyz, 
                          comment=xyz_comment_for_aggregate_qm)
            
            if actual_gen_monomer_xyz:
                for i, (mono_qm_atoms_xyz, mono_qm_coords_xyz) in enumerate(monomer_qm_regions):
                    monomer_name_from_meta = monomers_metadata_list[i].get(JSON_KEY_NAME, f'm{i+1}')
                    mono_has_qm_sol = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                    qm_sol_desc_mono = f" + its unique qm {solvent_name_str}" if mono_has_qm_sol else ""
                    # Original comment from user's script version
                    xyz_comment_for_monomer_qm = f"{monomer_name_from_meta} monomer{i+1} qm{qm_sol_desc_mono}" # Corrected: was 'qm {qm_sol_desc_mono}'
                    write_xyz(os.path.join(current_frame_output_dir, f'monomer{i+1}_qm.xyz'),
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
                    write_xyz(output_path, charges_col, coords_arr, comment=mm_solvent_agg_comment)
                    write_to_log(f"Written MM charges XYZ for aggregate to: {output_path} (contains MM solvent charges if any)")
                elif args.mm_solvent : 
                     warn_msg = (f"MM solvent was specified (--mm_solvent) but no point charges were loaded/generated for it. "
                                 f"File '{combined_system_term}_mm.xyz' will not contain MM solvent charges.")
                     print(f"\nWARNING: {warn_msg}") # Ensure newline
                     write_to_log(warn_msg, is_warning=True)

            # For monomer{i+1}_mm.xyz
            if actual_gen_monomer_xyz:
                for i in range(args.aggregate):
                    monomer_name_from_meta = monomers_metadata_list[i].get(JSON_KEY_NAME, f'm{i+1}')
                    
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
        
        # COM File Generation Block (This logic is mostly unchanged from your provided script)
        if user_intends_com_output:
            # ... (all the COM file generation logic remains as is) ...
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
                                                 (args.aggregate > 1 and qm_solvent_flags.get('monomer_1_has_added_qm_solvent', False))
                    
                    solvent_desc_suffix = get_solvent_descriptor_suffix(eetg_frags_have_qm_solvent, system_has_mm_solvent)
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
                            (m1_atoms_eetg, m1_coords_eetg, monomers_metadata_list[0][JSON_KEY_CHARGE], monomers_metadata_list[0][JSON_KEY_SPIN_MULT]),
                            (m2_atoms_eetg, m2_coords_eetg, monomers_metadata_list[1][JSON_KEY_CHARGE], monomers_metadata_list[1][JSON_KEY_SPIN_MULT])
                        ]
                        write_com_file(
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
                        write_to_log("EETG selected but system is not --aggregate=2 or monomer region data is incomplete. Skipping EETG file.", is_warning=True)

            if actual_gen_monomer_com: 
                write_to_log(f"Writing Monomer .com files (if not exclusively EETG aggregate)...") 
                for i in range(args.aggregate):
                    if i < len(monomer_qm_regions) and i < len(monomers_metadata_list): 
                        mono_meta = monomers_metadata_list[i]
                        mono_qm_atoms, mono_qm_coords = monomer_qm_regions[i]
                        
                        monomer_has_added_qm_solvent = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                        solvent_desc_suffix = get_solvent_descriptor_suffix(monomer_has_added_qm_solvent, system_has_mm_solvent)
                        mono_filename_base = f"monomer{i+1}{solvent_desc_suffix}"
                        mono_filename = f"{mono_filename_base}{tag_for_filename}.com"
                        
                        solvent_info_for_mono_title = ""
                        if monomer_has_added_qm_solvent and system_has_mm_solvent: solvent_info_for_mono_title = f" with QM & MM {solvent_name_str}"
                        elif monomer_has_added_qm_solvent: solvent_info_for_mono_title = f" with QM {solvent_name_str}"
                        elif system_has_mm_solvent: solvent_info_for_mono_title = f" with MM {solvent_name_str}"
                        
                        mono_title = f"{mono_meta.get(JSON_KEY_NAME, f'Monomer {i+1}')}{solvent_info_for_mono_title}".strip()
                        
                        all_mm_charges_for_this_monomer_calc: List[MMChargeTupleType] = []
                        if mm_embedding_charges_for_each_monomer[i]: 
                            all_mm_charges_for_this_monomer_calc.extend(mm_embedding_charges_for_each_monomer[i])
                        if mm_solvent_point_charges_xyz_q: 
                            all_mm_charges_for_this_monomer_calc.extend(mm_solvent_point_charges_xyz_q)

                        write_com_file(
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
                solvent_desc_suffix = get_solvent_descriptor_suffix(aggregate_has_added_qm_solvent, system_has_mm_solvent)
                agg_filename_base = f"{combined_system_term}{solvent_desc_suffix}"
                agg_filename = f"{agg_filename_base}{tag_for_filename}.com"

                solvent_info_for_agg_title = ""
                if aggregate_has_added_qm_solvent and system_has_mm_solvent: solvent_info_for_agg_title = f" with QM & MM {solvent_name_str}"
                elif aggregate_has_added_qm_solvent: solvent_info_for_agg_title = f" with QM {solvent_name_str}"
                elif system_has_mm_solvent: solvent_info_for_agg_title = f" with MM {solvent_name_str}"
                agg_title = f"{aggregate_title_base}{solvent_info_for_agg_title}".strip()
                
                agg_qm_atoms, agg_qm_coords = aggregate_qm_region
                
                write_com_file(
                    os.path.join(current_frame_output_dir, agg_filename),
                    gaussian_keywords_for_calcs if gaussian_keywords_for_calcs is not None else [], 
                    agg_title,
                    total_sys_charge_from_meta, total_sys_spin_mult_from_meta,
                    agg_qm_atoms, agg_qm_coords, 
                    mm_charges_list=mm_solvent_point_charges_xyz_q if mm_solvent_point_charges_xyz_q else None
                )

        # Cleanup temporary XYZ for this frame
        if args.traj_xyz and os.path.exists(temp_xyz_input_filepath) and TEMP_FRAME_XYZ_FILENAME in temp_xyz_input_filepath :
            try:
                os.remove(temp_xyz_input_filepath)
                write_to_log(f"Cleaned up temporary file: {temp_xyz_input_filepath}")
            except OSError as e:
                warn_msg = f"Could not remove temporary file {temp_xyz_input_filepath}: {e}"
                print(f"\nWARNING: {warn_msg}") # Ensure newline
                write_to_log(warn_msg, is_warning=True)
        
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
    write_to_log(final_message)

    if log_file_handle:
        log_file_handle.close()

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e: 
        err_msg_critical = f"CRITICAL FILE ERROR: {e}. Please check file paths and permissions."
        print(f"ERROR: {err_msg_critical}", file=sys.stderr)
        if log_file_handle and not log_file_handle.closed: 
            write_to_log(err_msg_critical, is_error=True)
            import traceback
            write_to_log(traceback.format_exc())
        sys.exit(1)
    except ValueError as e: 
        err_msg_critical = f"CRITICAL VALUE ERROR: {e}. Please check input values and formats."
        print(f"ERROR: {err_msg_critical}", file=sys.stderr)
        if log_file_handle and not log_file_handle.closed:
            write_to_log(err_msg_critical, is_error=True)
            import traceback # Moved import here
            traceback.print_exc(file=log_file_handle) 
        sys.exit(1)
    except Exception as e: 
        err_msg_critical = f"AN UNHANDLED CRITICAL ERROR OCCURRED: {e}"
        print(f"ERROR: {err_msg_critical}", file=sys.stderr)
        if log_file_handle and not log_file_handle.closed: 
            write_to_log(err_msg_critical, is_error=True)
            import traceback # Moved import here
            write_to_log(traceback.format_exc()) 
        sys.exit(1)
    finally:
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.close()