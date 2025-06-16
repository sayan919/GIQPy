"""Utility functions for GIQPy."""

import os
import json
import numpy as np
import re
import sys
from typing import List, Tuple, Optional, Dict, Any, Union

# --- Constants ---
AUTO_MM_SOLVENT_TRIGGER: str = "auto_detect_mm_solvent_from_input_xyz"
TEMP_FRAME_XYZ_FILENAME: str = "_current_frame_data.xyz"
LOG_FILENAME: str = "giqpy_run.log"

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
        return "_qm_mm" # More descriptive
    elif entity_has_added_qm_solvent:
        return "_qm"
    elif system_has_mm_solvent:
        return "_mm"
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
                    final_keywords.insert(i + 1, "# charge") # Add 'charge' on the next line
                    inserted = True
                    break
            if not inserted: # No line starting with #, so just append.
                 final_keywords.append("# charge")


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

