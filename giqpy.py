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
    
    --output_com (Optional) : monomer/dimer/both; (default: both).
    OR,
    --output_xyz (Optional) : monomer/dimer/both/none; (default: both).

    --gauss_keywords (Required with --output_com) : Gaussian keywords (must be used with --output_com).
    
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
        if base_output_dir_for_traj != "." and not os.path.exists(base_output_dir_for_traj):
            os.makedirs(base_output_dir_for_traj, exist_ok=True)
        processed_frames_info.append((None, single_xyz_path, base_output_dir_for_traj))
        return processed_frames_info

    if not traj_xyz_path: 
        print("ERROR: Trajectory path is None, but single_xyz_path was also None.", file=sys.stderr)
        write_to_log("Trajectory path is None, but single_xyz_path was also None.", is_error=True)
        return []

    try:
        with open(traj_xyz_path, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"ERROR: Trajectory file not found: {traj_xyz_path}", file=sys.stderr)
        write_to_log(f"Trajectory file not found: {traj_xyz_path}", is_error=True)
        raise
        
    if not lines:
        print(f"ERROR: Empty trajectory: {traj_xyz_path}", file=sys.stderr)
        write_to_log(f"Empty trajectory: {traj_xyz_path}", is_error=True)
        raise ValueError(f"Empty trajectory: {traj_xyz_path}")
    
    try:
        total_atoms_per_frame = int(lines[0].strip())
    except ValueError:
        print(f"ERROR: First line of trajectory {traj_xyz_path} must be the number of atoms.", file=sys.stderr)
        write_to_log(f"First line of trajectory {traj_xyz_path} must be the number of atoms.", is_error=True)
        raise
        
    block_size = total_atoms_per_frame + 2 
    if block_size <= 2:
        print(f"ERROR: Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.", file=sys.stderr)
        write_to_log(f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.", is_error=True)
        raise ValueError(f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.")

    max_possible_frames = len(lines) // block_size
    
    actual_frames_to_process = num_frames_to_extract if num_frames_to_extract and num_frames_to_extract <= max_possible_frames else max_possible_frames
    if num_frames_to_extract and num_frames_to_extract > max_possible_frames:
        warn_msg = f"Requested {num_frames_to_extract} frames, but trajectory only contains {max_possible_frames}. Processing all available frames."
        print(f"WARNING: {warn_msg}")
        write_to_log(warn_msg, is_warning=True)


    for i in range(actual_frames_to_process):
        frame_id_str = str(i + 1)
        output_dir_for_frame = os.path.join(base_output_dir_for_traj, frame_id_str)
        os.makedirs(output_dir_for_frame, exist_ok=True)
        
        temp_xyz_path_for_frame = os.path.join(output_dir_for_frame, TEMP_FRAME_XYZ_FILENAME)
        
        start_line_idx = i * block_size
        frame_block_lines = lines[start_line_idx : start_line_idx + block_size]
        
        try:
            with open(temp_xyz_path_for_frame, 'w') as f_temp_xyz:
                f_temp_xyz.write("\n".join(frame_block_lines) + "\n")
        except IOError as e:
            err_msg = f"Could not write temporary frame file {temp_xyz_path_for_frame}: {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            raise
            
        processed_frames_info.append((frame_id_str, temp_xyz_path_for_frame, output_dir_for_frame))
    
    return processed_frames_info


def load_system_info(system_info_path: str, aggregate: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load monomer and solvent metadata from a single JSON array file.
    Performs validation on the structure and content of the JSON.
    """
    write_to_log(f"Loading system info from: {system_info_path}") 
    try:
        with open(system_info_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        err_msg = f"System info JSON file not found: {system_info_path}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        raise
    except json.JSONDecodeError as e:
        err_msg = f"Invalid JSON format in system info file: {system_info_path}. Error: {e}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        raise ValueError(err_msg)

    if not isinstance(data, list) or len(data) < aggregate + 1:
        msg = f"system_info must be a JSON array with at least {aggregate + 1} entries ({aggregate} monomers + 1 solvent)."
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
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
            write_to_log(warn_msg, is_warning=True)


    # Validate solvent
    if not isinstance(solvent_data, dict):
        raise ValueError("Solvent entry in system_info is not a valid JSON object.")
    
    required_keys_solvent: Dict[str, Any] = { 
        JSON_KEY_MOL_FORMULA: str,
        JSON_KEY_NATOMS: int,
        JSON_KEY_CHARGES_ARRAY: list
    }
    solvent_id_for_log = solvent_data.get(JSON_KEY_NAME, "solvent") 
    solvent_data[JSON_KEY_NAME] = solvent_id_for_log 

    for key, expected_type in required_keys_solvent.items():
        if key not in solvent_data:
            raise ValueError(f"Solvent entry '{solvent_id_for_log}' is missing required key: '{key}'.")
        if not isinstance(solvent_data[key], expected_type): # type: ignore
            raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{key}' has incorrect type. Expected {expected_type}, got {type(solvent_data[key])}.")

    if solvent_data[JSON_KEY_NATOMS] <= 0:
        raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{JSON_KEY_NATOMS}' must be a positive integer.")
    if not solvent_data[JSON_KEY_CHARGES_ARRAY]:
        raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{JSON_KEY_CHARGES_ARRAY}' must be a non-empty list.")
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
    
    write_to_log("System info loaded and validated successfully.") 
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
    if len(atoms) % size != 0:
        msg = f"Total solvent atom count ({len(atoms)}) is not divisible by formula size ({size}). Ensure correct solvent definition and XYZ content."
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
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
    for core_coord_set in core_coords_list:
        if core_coord_set.size == 0 or mol_coords_array.size == 0:
            continue
        dists = np.linalg.norm(core_coord_set[:, None, :] - mol_coords_array[None, :, :], axis=2)
        if np.any(dists < radius):
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
        return flat_atoms, np.array([]) 
    return flat_atoms, np.vstack(flat_coords_list)


def write_xyz(path: str, atoms: Union[AtomListType, ChargeListType], coords: CoordType, comment: str = "") -> None:
    """
    Export atoms and coords to an XYZ file.
    The 'atoms' list can contain strings (element symbols) or floats (charges for MM solvent XYZ).
    """
    coords_array = np.array(coords)
    if coords_array.ndim == 1 and coords_array.shape[0] == 3 and len(atoms) == 1: 
        coords_array = coords_array.reshape(1, 3)
    elif coords_array.ndim == 1 and len(atoms) > 1 : 
        raise ValueError(f"Coordinates array for multiple atoms has unexpected shape: {coords_array.shape}")
    elif coords_array.size == 0 and len(atoms) > 0:
        raise ValueError("Cannot write XYZ: Atoms provided but coordinates are empty.")
    elif coords_array.ndim == 2 and coords_array.shape[0] != len(atoms):
         raise ValueError(f"Mismatch between number of atoms ({len(atoms)}) and number of coordinate rows ({coords_array.shape[0]}).")

    try:
        with open(path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for atom_symbol_or_charge, xyz_row in zip(atoms, coords_array):
                if isinstance(atom_symbol_or_charge, float):
                    atom_col_str = f"{atom_symbol_or_charge:<12.5f}" 
                else:
                    atom_col_str = f"{str(atom_symbol_or_charge):<3}" 
                f.write(f"{atom_col_str} {xyz_row[0]:12.5f} {xyz_row[1]:12.5f} {xyz_row[2]:12.5f}\n")
    except IOError as e:
        err_msg = f"Could not write XYZ file {path}: {e}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
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
        write_to_log(err_msg, is_error=True)
        raise

    if len(lines) < 2: raise ValueError(f"XYZ file {main_xyz_input_filepath} is too short.")
    try: num_atoms_in_main_file = int(lines[0].strip())
    except ValueError: raise ValueError(f"First line of XYZ {main_xyz_input_filepath} must be atom count.")
    if len(lines) < num_atoms_in_main_file + 2: raise ValueError(f"XYZ {main_xyz_input_filepath} has fewer lines than atom count.")
    
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
    core_coords_all_for_dist_calc: CoordType
    solvent_atoms_from_xyz_list: AtomListType
    solvent_coords_from_xyz_array: CoordType

    if qm_aggregate_coords_override is not None:
        if qm_aggregate_coords_override.shape[0] != total_core_atom_count_from_json:
            msg = (f"Atom count in --qm_aggregate_xyz ({qm_aggregate_coords_override.shape[0]}) "
                   f"does not match total core atoms from system_info ({total_core_atom_count_from_json}).")
            print(f"ERROR: {msg}", file=sys.stderr)
            write_to_log(msg, is_error=True)
            raise ValueError(msg)
        
        core_atoms_all_list = main_atoms_all_array[:total_core_atom_count_from_json].tolist() 
        core_coords_all_for_dist_calc = qm_aggregate_coords_override
        
        if num_atoms_in_main_file < total_core_atom_count_from_json:
            msg = (f"Main input XYZ file '{main_xyz_input_filepath}' has fewer atoms ({num_atoms_in_main_file}) "
                   f"than expected for the core ({total_core_atom_count_from_json}) when using --qm_aggregate_xyz. "
                   f"It must still contain at least the core atoms for solvent indexing.")
            print(f"ERROR: {msg}", file=sys.stderr)
            write_to_log(msg, is_error=True)
            raise ValueError(msg)
            
        solvent_atoms_from_xyz_list = main_atoms_all_array[total_core_atom_count_from_json:].tolist()
        solvent_coords_from_xyz_array = main_coords_all_array[total_core_atom_count_from_json:]
        write_to_log(f"Using core coordinates from --qm_aggregate_xyz. Atom types for core and all solvent taken from {main_xyz_input_filepath}.")

    else:
        core_atoms_all_list = main_atoms_all_array[:total_core_atom_count_from_json].tolist()
        core_coords_all_for_dist_calc = main_coords_all_array[:total_core_atom_count_from_json]
        solvent_atoms_from_xyz_list = main_atoms_all_array[total_core_atom_count_from_json:].tolist()
        solvent_coords_from_xyz_array = main_coords_all_array[total_core_atom_count_from_json:]

    solvent_formula_counts = parse_formula(solvent_meta[JSON_KEY_MOL_FORMULA])
    all_solvent_groups_list: List[SolventGroupType] = group_qm_molecules(solvent_atoms_from_xyz_list, solvent_coords_from_xyz_array, solvent_formula_counts)
    
    aggregate_qm_solvent_groups: List[SolventGroupType] = [g for g in all_solvent_groups_list if is_within_radius([core_coords_all_for_dist_calc], g[1], qm_radius)]
    aggregate_qm_solvent_atoms_list, aggregate_qm_solvent_coords_array = flatten_groups(aggregate_qm_solvent_groups)
    
    aggregate_qm_group_ids = set(map(id, aggregate_qm_solvent_groups))
    non_qm_solvent_groups: List[SolventGroupType] = [g for g in all_solvent_groups_list if id(g) not in aggregate_qm_group_ids]

    monomer_localized_qm_solvent_groups_list: List[List[SolventGroupType]] = [] 
    current_core_idx = 0
    for num_atoms_mono in n_atoms_per_monomer_list:
        monomer_core_coords = core_coords_all_for_dist_calc[current_core_idx : current_core_idx + num_atoms_mono]
        localized_groups = [g for g in all_solvent_groups_list if is_within_radius([monomer_core_coords], g[1], qm_radius)]
        monomer_localized_qm_solvent_groups_list.append(localized_groups)
        current_core_idx += num_atoms_mono
        
    seen_qm_solvent_ids_for_monomers: set = set()
    monomer_unique_qm_solvent_data_list: List[Tuple[AtomListType, CoordType]] = [] 
    qm_solvent_flags: Dict[str, bool] = {}

    for i in range(len(monomers_meta)):
        unique_groups_for_this_mono = []
        for group in monomer_localized_qm_solvent_groups_list[i]:
            if id(group) not in seen_qm_solvent_ids_for_monomers:
                unique_groups_for_this_mono.append(group)
                seen_qm_solvent_ids_for_monomers.add(id(group))
        added_qm_atoms, added_qm_coords = flatten_groups(unique_groups_for_this_mono)
        monomer_unique_qm_solvent_data_list.append((added_qm_atoms, added_qm_coords))
        qm_solvent_flags[f'monomer_{i}_has_added_qm_solvent'] = bool(added_qm_atoms)

    qm_solvent_flags['aggregate_has_added_qm_solvent'] = bool(aggregate_qm_solvent_atoms_list)

    aggregate_final_qm_atoms = core_atoms_all_list + aggregate_qm_solvent_atoms_list
    aggregate_final_qm_coords = np.vstack((core_coords_all_for_dist_calc, aggregate_qm_solvent_coords_array)) if aggregate_qm_solvent_coords_array.size > 0 else core_coords_all_for_dist_calc
    aggregate_qm_region_for_gauss: Tuple[AtomListType, CoordType] = (aggregate_final_qm_atoms, aggregate_final_qm_coords)

    monomer_qm_regions_for_gauss: List[Tuple[AtomListType, CoordType]] = []
    current_core_idx = 0
    for i in range(len(monomers_meta)):
        mono_core_atoms = core_atoms_all_list[current_core_idx : current_core_idx + n_atoms_per_monomer_list[i]]
        mono_core_coords = core_coords_all_for_dist_calc[current_core_idx : current_core_idx + n_atoms_per_monomer_list[i]]
        
        added_qm_atoms, added_qm_coords = monomer_unique_qm_solvent_data_list[i]
        
        final_mono_atoms = mono_core_atoms + added_qm_atoms
        final_mono_coords = np.vstack((mono_core_coords, added_qm_coords)) if added_qm_coords.size > 0 else mono_core_coords
        monomer_qm_regions_for_gauss.append((final_mono_atoms, final_mono_coords))
        current_core_idx += n_atoms_per_monomer_list[i]
            
    return (monomer_qm_regions_for_gauss, aggregate_qm_region_for_gauss, 
            non_qm_solvent_groups, qm_solvent_flags, 
            core_coords_all_for_dist_calc, n_atoms_per_monomer_list)


def load_monomer_charges_from_file(charge_file_path: str) -> List[MMChargeTupleType]:
    """
    Load MM charges from a file for monomer embedding. Format: charge x y z.
    Returns list of (x, y, z, charge) tuples.
    """
    charges_data: List[MMChargeTupleType] = []
    try:
        with open(charge_file_path, 'r') as f:
            for line_num, line_content in enumerate(f):
                parts = line_content.strip().split()
                if len(parts) == 4:
                    try:
                        charge = float(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        charges_data.append((x, y, z, charge)) 
                    except ValueError:
                        warn_msg = f"Non-numeric value in charge file {charge_file_path} on line {line_num + 1}: {line_content.strip()}"
                        print(f"WARNING: {warn_msg}")
                        write_to_log(warn_msg, is_warning=True)
                elif parts: 
                    warn_msg = f"Malformed line in charge file {charge_file_path} on line {line_num + 1}: {line_content.strip()}"
                    print(f"WARNING: {warn_msg}")
                    write_to_log(warn_msg, is_warning=True)
    except FileNotFoundError:
        warn_msg = f"MM Monomer charge file not found: {charge_file_path}. Returning empty charges."
        print(f"WARNING: {warn_msg}")
        write_to_log(warn_msg, is_warning=True)
    return charges_data

def assign_charges_to_solvent_molecules(solvent_groups: List[SolventGroupType], solvent_metadata_entry: Dict[str, Any]) -> List[MMChargeTupleType]:
    """
    Assigns charges to solvent molecules based on the solvent_metadata_entry.
    Returns list of (charge, x, y, z) tuples.
    """
    mm_solvent_charges_list: List[MMChargeTupleType] = []
    defined_charges_per_mol: List[Dict[str, Any]] = solvent_metadata_entry[JSON_KEY_CHARGES_ARRAY]
    num_atoms_per_solvent_mol_defined: int = solvent_metadata_entry[JSON_KEY_NATOMS]

    if len(defined_charges_per_mol) != num_atoms_per_solvent_mol_defined:
        msg = (f"Mismatch in solvent definition: '{JSON_KEY_NATOMS}' ({num_atoms_per_solvent_mol_defined}) "
               f"does not match entries in '{JSON_KEY_CHARGES_ARRAY}' ({len(defined_charges_per_mol)}) "
               f"for solvent '{solvent_metadata_entry.get(JSON_KEY_NAME, 'Unnamed Solvent')}'.")
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
        raise ValueError(msg)

    for mol_idx, (atom_symbols_in_molecule, coords_for_molecule) in enumerate(solvent_groups):
        if len(atom_symbols_in_molecule) != num_atoms_per_solvent_mol_defined:
            warn_msg = (f"Solvent molecule {mol_idx+1} (auto-detected for MM) has {len(atom_symbols_in_molecule)} atoms, "
                        f"but system_info defines {num_atoms_per_solvent_mol_defined} for solvent "
                        f"'{solvent_metadata_entry.get(JSON_KEY_NAME, 'Unnamed Solvent')}'. Skipping this molecule.")
            print(f"WARNING: {warn_msg}")
            write_to_log(warn_msg, is_warning=True)
            continue

        for atom_idx_in_mol in range(len(atom_symbols_in_molecule)):
            atom_symbol = atom_symbols_in_molecule[atom_idx_in_mol]
            atom_coord = coords_for_molecule[atom_idx_in_mol]
            
            charge_info_for_this_atom_pos = defined_charges_per_mol[atom_idx_in_mol]
            charge_value: float = charge_info_for_this_atom_pos[JSON_KEY_CHARGE]
            defined_element_for_pos: str = charge_info_for_this_atom_pos[JSON_KEY_ELEMENT]

            if atom_symbol != defined_element_for_pos:
                warn_msg = (f"Element mismatch for atom {atom_idx_in_mol+1} in auto-detected MM solvent molecule {mol_idx+1}. "
                            f"XYZ has '{atom_symbol}', system_info expects '{defined_element_for_pos}' at this position. "
                            f"Using charge for '{defined_element_for_pos}' ({charge_value}).")
                print(f"WARNING: {warn_msg}")
                write_to_log(warn_msg, is_warning=True)
            
            mm_solvent_charges_list.append((charge_value, atom_coord[0], atom_coord[1], atom_coord[2]))
            
    return mm_solvent_charges_list


def get_solvent_descriptor_suffix(entity_has_added_qm_solvent: bool, system_has_mm_solvent: bool) -> str:
    """Determines the solvent descriptor part of the .com filename (without .com extension)."""
    if entity_has_added_qm_solvent and system_has_mm_solvent:
        return "_qm_mm"
    elif entity_has_added_qm_solvent:
        return "_qm"
    elif system_has_mm_solvent:
        return "_mm"
    else:
        return "" 


def write_com_file( # Renamed from write_vee_com and write_eetg_com to be more general
    path: str, keywords: List[str], title: str, charge: Union[int, float], spin: int,
    atoms_list: AtomListType, coords_array: CoordType, 
    mm_charges_list: Optional[List[MMChargeTupleType]] = None, # Combined MM charges (monomer or solvent)
    fragment_definitions: Optional[List[Tuple[AtomListType, CoordType, int, int]]] = None # For EETG: [(atoms1, coords1, chg1, spin1), (atoms2, coords2, chg2, spin2)]
) -> None:   
    """
    Writes a Gaussian .com input file.
    Can handle standard calculations and EETG (if fragment_definitions is provided).
    mm_charges_list are (x,y,z,q) for Gaussian's charge keyword.
    """
    final_keywords = list(keywords) 
    if mm_charges_list:
        charge_keyword_present = any("# charge" in kw.lower() for kw in final_keywords)
        if not charge_keyword_present:
            final_keywords.append("# charge ! MM charges present for QM/MM embedding")

    try:
        with open(path, 'w') as f:
            f.write(f"%chk={os.path.splitext(os.path.basename(path))[0]}.chk\n")
            for kw_line in final_keywords:
                f.write(kw_line + "\n")
            f.write("\n" + title + "\n\n")
            
            if fragment_definitions: # EETG case
                # Overall charge/spin, then per-fragment charge/spin
                f.write(f"{charge} {spin} ")
                for _, _, chg_frag, spin_frag in fragment_definitions:
                    f.write(f"{chg_frag} {spin_frag} ")
                f.write("\n")
                
                for frag_idx, (frag_atoms, frag_coords, _, _) in enumerate(fragment_definitions):
                    for atom_sym, xyz_coords in zip(frag_atoms, frag_coords):
                        f.write(f" {atom_sym}(Fragment={frag_idx+1}) {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")
            else: # Standard (VEE-like) case
                f.write(f"{charge} {spin}\n")
                for atom_sym, xyz_coords in zip(atoms_list, coords_array):
                    f.write(f" {atom_sym} {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")
            
            if mm_charges_list: 
                f.write("\n") 
                for x_mm, y_mm, z_mm, q_mm in mm_charges_list: 
                    f.write(f" {x_mm:14.5f} {y_mm:12.5f} {z_mm:12.5f} {q_mm:12.5f}\n")
            
            f.write("\n") 
    except IOError as e:
        err_msg = f"Could not write .com file {path}: {e}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        raise


def load_keywords_from_file(path: str) -> List[str]:
    """
    Load Gaussian route keywords from a text file.
    """
    try:
        with open(path) as f:
            return [line.rstrip() for line in f if line.strip()]
    except FileNotFoundError:
        err_msg = f"Keywords file not found: {path}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        raise

def calculate_centroid_distance_between_first_two(core_coords_all_array: CoordType, n_atoms_per_monomer_list: List[int]) -> Optional[float]:
    """
    Calculates the distance between the centroids of the first two monomers' core regions.
    Returns the distance as a float, or None if not applicable.
    """
    if len(n_atoms_per_monomer_list) < 2:
        return None 

    n_atoms_m1 = n_atoms_per_monomer_list[0]
    if n_atoms_m1 == 0: 
        write_to_log("Cannot calculate centroid distance: Monomer 1 has 0 atoms.")
        return None 
    coords_m1 = core_coords_all_array[0:n_atoms_m1]
    if coords_m1.shape[0] == 0: 
        write_to_log("Cannot calculate centroid distance: Monomer 1 coordinate array is empty.")
        return None 
    centroid_m1 = np.mean(coords_m1, axis=0)

    n_atoms_m2 = n_atoms_per_monomer_list[1]
    if n_atoms_m2 == 0: 
        write_to_log("Cannot calculate centroid distance: Monomer 2 has 0 atoms.")
        return None 
    start_idx_m2 = n_atoms_m1
    end_idx_m2 = n_atoms_m1 + n_atoms_m2
    coords_m2 = core_coords_all_array[start_idx_m2:end_idx_m2]
    if coords_m2.shape[0] == 0: 
        write_to_log("Cannot calculate centroid distance: Monomer 2 coordinate array is empty.")
        return None 
    centroid_m2 = np.mean(coords_m2, axis=0)
    
    distance: float = np.linalg.norm(centroid_m1 - centroid_m2)
    return distance

def main() -> None:
    global log_file_handle 

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description="Generate Gaussian .com and .xyz files from single or trajectory XYZ with QM/MM options.",
        epilog="""
-------------------------------------------------------------------------------
Core Flags:
  --single_xyz <file.xyz> or --traj_xyz <file.xyz> :
                                  Input coordinate file(s).
                                  gen_qm_mm_files.py uses these for the entire system (monomers + solvent).
  --frames <N>                    : Number of frames if --traj_xyz is used.
  --aggregate <M>               : Number of core monomer units (e.g., 2 for a dimer).
  --system_info <file.json>     : Single JSON defining all monomers and solvent properties.
                                  This is gen_qm_mm_files.py's comprehensive way to get monomer details
                                  (name, nAtoms, charge, spin_mult, mol_formula) and
                                  solvent details (name, mol_formula, nAtoms, charges per atom).
  --gauss_keywords <file.txt>   : Text file with Gaussian route section keywords.
                                  Each line is a keyword. Conditionally required if generating .com files.
                                  Note: PCM solvent environment details, like from an older `--env` flag,
                                  should be included directly within these Gaussian keywords
                                  (e.g., SCRF=(PCM,Solvent=Water)).
  --qm_aggregate_xyz <file.xyz> : Optional: Use coordinates from this file for the core aggregate,
                                  overriding those from the main input XYZ. Atom order must match.
  --qm_solvent <radius>         : Defines QM solvent shell by radius (Å) around core atoms.
                                  This provides explicit QM solvent inclusion.
  --mm_solvent [file.xyz]       : Optional. Defines MM solvent.
                                  - If path provided: XYZ-like file (charge x y z per line, skips 2 headers).
                                  - If flag used alone: Auto-detects non-QM solvent from input XYZ
                                    and uses charges from system_info.json.
  --mm_monomer [0|file1 ...]    : Optional. Defines MM charges for embedding from other monomers.
                                  - If '0': Embeds other monomers with zero charges at their atomic positions.
                                  - If file(s) provided: File(s) with "charge x y z" for MM charges.
  --eetg                        : Flag to generate only EETG input for dimers (requires --aggregate=2).
  --output_com <choice>         : Control .com file generation (monomer, dimer, both; default: both).
  --output_xyz [choice]         : Control detailed .xyz file generation (monomer, dimer, both, none; 
                                  If flag is present without value, defaults to "both". If flag absent, no detailed XYZs are generated.)
  --tag <TAG_STRING>            : Optional custom tag for generated .com filenames (e.g., ..._TAG.com).
  --logfile <filename>          : Specify log file name (default: run.log).
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
    parser.add_argument('--gauss_keywords', type=str, default=None, 
                        help='File with Gaussian keywords (if generating .com files).')
    parser.add_argument('--qm_aggregate_xyz', type=str, default=None,
                        help='Optional: XYZ file for core aggregate coordinates, overriding main input for core.')
    parser.add_argument('--qm_solvent', type=float, default=5.0,
                        help='Radius (Å) for QM solvent selection (default: 5.0).')
    parser.add_argument('--mm_monomer', type=str, nargs='*',
                        help="MM embedding: '0' for zero charges, or charge file(s).")
    parser.add_argument('--mm_solvent', type=str, nargs='?', const=AUTO_MM_SOLVENT_TRIGGER, default=None,
                        help='MM solvent: XYZ-like charge file, or flag alone for auto-detect.')
    parser.add_argument('--eetg', action='store_true',
                        help='Generate EETG input for dimer (requires --aggregate 2).')
    parser.add_argument('--output_com', choices=['monomer', 'dimer', 'both'], default='both',
                        help='.com files to generate: monomer, dimer, or both (default: both).')
    parser.add_argument('--output_xyz', nargs='?', choices=['monomer', 'dimer', 'both'], const='both', default=None, 
                        help="Detailed XYZs to generate. If flag given, defaults to 'both'. If absent, none.")
    parser.add_argument('--tag', type=str, default="",
                        help='Optional custom tag for .com filenames (e.g., monomer1_qm_mm_TAG.com).')
    
    args = parser.parse_args()

    # --- Setup Log File ---
    try:
        log_file_handle = open(LOG_FILENAME, 'w')
        log_file_handle.write(f"GIQPy Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
        log_file_handle.write(f"Arguments: {vars(args)}\n\n")
        log_file_handle.flush()
    except IOError as e:
        print(f"ERROR: Could not open log file {LOG_FILENAME} for writing: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)


    # --- Initial Argument Checks ---
    if not args.output_com and not args.output_xyz: 
        err_msg = "No output requested. Please specify --output_com and/or --output_xyz."
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
    if args.eetg and args.aggregate != 2:
        err_msg = "--eetg calculations require --aggregate=2 (a dimer system)."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        parser.error(err_msg)
    if args.output_com == 'monomer' and args.eetg:
        warn_msg = "Configuration: --output_com monomer is selected with --eetg. EETG is an aggregate/dimer property and will not be generated for the combined system."
        print(f"WARNING: {warn_msg}")
        write_to_log(warn_msg, is_warning=True)
    
    gaussian_keywords: Optional[List[str]] = None
    if args.gauss_keywords:
        if not os.path.exists(args.gauss_keywords) or os.path.getsize(args.gauss_keywords) == 0:
            err_msg = f"Gaussian keywords file '{args.gauss_keywords}' not found or is empty."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            raise FileNotFoundError(err_msg)
        gaussian_keywords = load_keywords_from_file(args.gauss_keywords)
    elif args.output_com in ['monomer', 'dimer', 'both']: 
        err_msg = "--gauss_keywords must be provided when generating .com files (i.e., when --output_com is 'monomer', 'dimer', or 'both')."
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        parser.error(err_msg)

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
            for ln_idx, ln_content in enumerate(agg_lines[2 : num_atoms_in_agg_file + 2]):
                parts = ln_content.split()
                if len(parts) >= 4:
                    try: temp_agg_coords_list.append([float(p) for p in parts[1:4]])
                    except ValueError: raise ValueError(f"Malformed coord in {args.qm_aggregate_xyz} line {ln_idx+3}")
                else: raise ValueError(f"Malformed line in {args.qm_aggregate_xyz} line {ln_idx+3}")
            qm_aggregate_coords_override = np.array(temp_agg_coords_list)
            write_to_log(f"Successfully loaded {qm_aggregate_coords_override.shape[0]} aggregate coordinates from {args.qm_aggregate_xyz}.")
        except Exception as e:
            err_msg = f"Failed to load or parse --qm_aggregate_xyz file '{args.qm_aggregate_xyz}': {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            write_to_log(err_msg, is_error=True)
            sys.exit(1)


    base_output_dir: str = os.getcwd() 

    processed_frames_list: List[Tuple[Optional[str], str, str]] = split_frames(args.single_xyz, args.traj_xyz, args.frames, base_output_dir)
    total_frames_to_process = len(processed_frames_list)
    
    if total_frames_to_process == 0 and args.traj_xyz : 
        err_msg = f"No frames were found or could be processed from trajectory: {args.traj_xyz}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        sys.exit(1)
    elif total_frames_to_process == 0 and args.single_xyz:
         err_msg = f"Could not process single XYZ file: {args.single_xyz}"
         print(f"ERROR: {err_msg}", file=sys.stderr)
         write_to_log(err_msg, is_error=True)
         sys.exit(1)


    monomers_metadata_list: List[Dict[str, Any]]
    solvent_metadata: Dict[str, Any]
    monomers_metadata_list, solvent_metadata = load_system_info(args.system_info, args.aggregate)
    solvent_name_str = solvent_metadata.get(JSON_KEY_NAME, "solvent")


    if len(monomers_metadata_list) != args.aggregate:
        msg = (f"Number of monomer entries in system_info ({len(monomers_metadata_list)}) "
               f"does not match --aggregate ({args.aggregate}).")
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
        raise ValueError(msg)

    combined_system_term: str = "dimer" if args.aggregate == 2 else "aggregate"
    tag_for_filename = f"_{args.tag}" if args.tag else "" 


    if args.traj_xyz and total_frames_to_process > 0 :
        sys.stdout.write(f"Processed Frames: 0 / {total_frames_to_process}")
        sys.stdout.flush()


    for frame_idx, (frame_id_str, temp_xyz_input_filepath, current_frame_output_dir) in enumerate(processed_frames_list):
        if args.traj_xyz and total_frames_to_process > 0:
            sys.stdout.write('\r' + ' ' * 50 + '\r') 
            sys.stdout.flush()

        logged_output_dir_name = frame_id_str if frame_id_str else "." 
        
        frame_processing_message = f"--- Processing Frame {frame_idx + 1} / {total_frames_to_process} ({'ID: '+frame_id_str if frame_id_str else 'single file'}) -> Output Dir: '{logged_output_dir_name}' ---"
        print(frame_processing_message) 
        write_to_log(f"\n\n{frame_processing_message}") 


        monomer_qm_regions: List[Tuple[AtomListType, CoordType]]
        aggregate_qm_region: Tuple[AtomListType, CoordType]
        non_qm_sol_groups: List[SolventGroupType]
        qm_solvent_flags: Dict[str, bool]
        core_coords_for_dist_calc: CoordType
        n_atoms_per_monomer_for_dist_calc: List[int]

        monomer_qm_regions, aggregate_qm_region, non_qm_sol_groups, qm_solvent_flags, \
        core_coords_for_dist_calc, n_atoms_per_monomer_for_dist_calc = \
            localize_solvent_and_prepare_regions(
                temp_xyz_input_filepath,
                args.system_info,
                args.aggregate,
                args.qm_solvent,
                qm_aggregate_coords_override=qm_aggregate_coords_override 
            )

        monomer_names_from_meta: List[str] = [m.get(JSON_KEY_NAME, f'm{idx+1}') for idx, m in enumerate(monomers_metadata_list)]
        base_system_name_for_title: str
        if len(set(monomer_names_from_meta)) == 1: 
            base_system_name_for_title = monomer_names_from_meta[0]
        else: 
            base_system_name_for_title = "_".join(monomer_names_from_meta)
        
        distance_str_for_title: str = ""
        if args.aggregate >= 2:
            dist = calculate_centroid_distance_between_first_two(core_coords_for_dist_calc, n_atoms_per_monomer_for_dist_calc)
            if dist is not None:
                distance_str_for_title = f"(m1-m2 centroid distance: {dist:.2f} Å)"
        
        aggregate_title_base: str = f"{base_system_name_for_title} {combined_system_term} {distance_str_for_title}".strip()
        aggregate_xyz_comment_base: str = f"{base_system_name_for_title} {combined_system_term} {distance_str_for_title}".strip()

        mm_solvent_point_charges: Optional[List[MMChargeTupleType]] = None
        system_has_mm_solvent: bool = False
        if args.mm_solvent == AUTO_MM_SOLVENT_TRIGGER:
            write_to_log("Attempting to auto-detect MM solvent from non-QM solvent molecules...") 
            if non_qm_sol_groups:
                mm_solvent_point_charges = assign_charges_to_solvent_molecules(non_qm_sol_groups, solvent_metadata)
                if mm_solvent_point_charges:
                    write_to_log(f"Generated {len(mm_solvent_point_charges)} MM solvent point charges from auto-detection.") 
                    system_has_mm_solvent = True
                else:
                    write_to_log("Auto-detection found non-QM solvent groups, but no charges were assigned (check solvent definition or warnings).") 
            else:
                write_to_log("No non-QM solvent molecules found to treat as MM solvent.") 
        elif args.mm_solvent is not None: 
            write_to_log(f"Loading MM solvent charges from file: {args.mm_solvent}") 
            loaded_mm_charges_from_file: List[MMChargeTupleType] = []
            try:
                with open(args.mm_solvent, 'r') as f_mm_sol_file:
                    lines = f_mm_sol_file.readlines()
                    if len(lines) >=2: 
                        for line_num, line_content in enumerate(lines[2:]): 
                            parts = line_content.strip().split()
                            if len(parts) == 4: 
                                try:
                                    loaded_mm_charges_from_file.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
                                except ValueError:
                                    warn_msg = f"Non-numeric value in MM solvent file {args.mm_solvent} on line {line_num + 3}: {line_content.strip()}"
                                    print(f"WARNING: {warn_msg}")
                                    write_to_log(warn_msg, is_warning=True)
                            elif parts:
                                warn_msg = f"Malformed MM solvent line in {args.mm_solvent} on line {line_num + 3}: {line_content.strip()}"
                                print(f"WARNING: {warn_msg}")
                                write_to_log(warn_msg, is_warning=True)
                    else:
                        warn_msg = f"MM solvent file {args.mm_solvent} is too short, expected XYZ-like format with 2 header lines."
                        print(f"WARNING: {warn_msg}")
                        write_to_log(warn_msg, is_warning=True)

                if loaded_mm_charges_from_file:
                    mm_solvent_point_charges = loaded_mm_charges_from_file
                    system_has_mm_solvent = True
                    write_to_log(f"Loaded {len(mm_solvent_point_charges)} MM solvent point charges from file.") 
                else:
                    warn_msg = f"No valid MM solvent charges loaded from file {args.mm_solvent}."
                    print(f"WARNING: {warn_msg}")
                    write_to_log(warn_msg, is_warning=True)
            except FileNotFoundError:
                warn_msg = f"MM solvent file {args.mm_solvent} not found. No MM solvent will be used."
                print(f"WARNING: {warn_msg}")
                write_to_log(warn_msg, is_warning=True)
        
        if args.output_xyz: 
            generate_monomer_xyz = args.output_xyz in ['monomer', 'both']
            generate_aggregate_xyz = args.output_xyz in ['dimer', 'both'] 

            write_to_log("Writing QM region and MM solvent XYZ files...") 
            if generate_aggregate_xyz:
                agg_qm_atoms_xyz, agg_qm_coords_xyz = aggregate_qm_region
                qm_sol_desc_agg = f" + qm {solvent_name_str}" if qm_solvent_flags.get('aggregate_has_added_qm_solvent') else ""
                xyz_comment_for_aggregate_qm = f"{aggregate_xyz_comment_base}{qm_sol_desc_agg}"
                write_xyz(os.path.join(current_frame_output_dir, f'{combined_system_term}_qm.xyz'), 
                          agg_qm_atoms_xyz, agg_qm_coords_xyz, 
                          comment=xyz_comment_for_aggregate_qm)
            
            if generate_monomer_xyz:
                for i, (mono_qm_atoms_xyz, mono_qm_coords_xyz) in enumerate(monomer_qm_regions):
                    monomer_name_from_meta = monomers_metadata_list[i].get(JSON_KEY_NAME, f'm{i+1}')
                    mono_has_qm_sol = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                    qm_sol_desc_mono = f" + its unique qm {solvent_name_str}" if mono_has_qm_sol else ""
                    xyz_comment_for_monomer_qm = f"{monomer_name_from_meta} monomer{i+1}{qm_sol_desc_mono}"
                    write_xyz(os.path.join(current_frame_output_dir, f'monomer{i+1}_qm.xyz'),
                              mono_qm_atoms_xyz, mono_qm_coords_xyz,
                              comment=xyz_comment_for_monomer_qm)
            
            if system_has_mm_solvent and mm_solvent_point_charges:
                mm_solvent_charges_as_first_col: ChargeListType = [charge_val for charge_val, x,y,z in mm_solvent_point_charges]
                mm_solvent_coords_for_xyz: CoordType = np.array([(x,y,z) for q,x,y,z in mm_solvent_point_charges])
                
                if generate_aggregate_xyz:
                    mm_solvent_agg_comment = f"mm {solvent_name_str} for {base_system_name_for_title} {combined_system_term}"
                    write_xyz(os.path.join(current_frame_output_dir, f'{combined_system_term}_mm_solvent.xyz'), 
                              mm_solvent_charges_as_first_col, mm_solvent_coords_for_xyz, 
                              comment=mm_solvent_agg_comment)
                if generate_monomer_xyz:
                    for i in range(args.aggregate):
                         monomer_name_from_meta = monomers_metadata_list[i].get(JSON_KEY_NAME, f'm{i+1}')
                         mm_solvent_mono_comment = f"mm {solvent_name_str} for {monomer_name_from_meta} monomer{i+1}"
                         write_xyz(os.path.join(current_frame_output_dir, f'monomer{i+1}_mm_solvent.xyz'),
                                   mm_solvent_charges_as_first_col, mm_solvent_coords_for_xyz,
                                   comment=mm_solvent_mono_comment)
            elif args.mm_solvent and args.output_xyz: 
                warn_msg = "MM solvent was specified but no point charges were loaded/generated; cannot write MM solvent XYZ files."
                print(f"WARNING: {warn_msg}")
                write_to_log(warn_msg, is_warning=True)

        mm_embedding_for_monomer: List[List[MMChargeTupleType]] = [[] for _ in range(args.aggregate)] 
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
                            if i == j:
                                continue
                            other_monomer_coords = core_monomer_coords_list[j]
                            for coord_row in other_monomer_coords:
                                embedding_charges_for_monomer_i.append((coord_row[0], coord_row[1], coord_row[2], 0.0))
                        mm_embedding_for_monomer[i] = embedding_charges_for_monomer_i

            elif len(args.mm_monomer) == args.aggregate and args.aggregate > 0: 
                all_monomer_explicit_charges = [load_monomer_charges_from_file(f) for f in args.mm_monomer]
                if args.aggregate == 1:
                     warn_msg = "--mm_monomer provided with file(s) for --aggregate=1. This charge file will not be used for inter-monomer embedding."
                     print(f"WARNING: {warn_msg}")
                     write_to_log(warn_msg, is_warning=True)
                elif args.aggregate == 2: 
                    mm_embedding_for_monomer[0] = all_monomer_explicit_charges[1] 
                    mm_embedding_for_monomer[1] = all_monomer_explicit_charges[0] 
                else: 
                    for i in range(args.aggregate):
                        combined_embedding_charges = []
                        for j in range(args.aggregate):
                            if i == j: continue
                            combined_embedding_charges.extend(all_monomer_explicit_charges[j])
                        mm_embedding_for_monomer[i] = combined_embedding_charges
            elif len(args.mm_monomer) == 0: 
                 write_to_log("Note: --mm_monomer was specified without files (empty list). No inter-monomer MM charges will be applied.") 
            else: 
                warn_msg = (f"--mm_monomer was given with {len(args.mm_monomer)} file(s) "
                            f"(and not as '0'). Expected 0, 1 (for '0' option), or {args.aggregate} files. "
                            f"MM monomer embedding might be skipped or incorrect.")
                print(f"WARNING: {warn_msg}")
                write_to_log(warn_msg, is_warning=True)


        total_sys_charge: Union[int, float] = sum(m[JSON_KEY_CHARGE] for m in monomers_metadata_list)
        total_sys_spin: int = int(np.prod([m[JSON_KEY_SPIN_MULT] for m in monomers_metadata_list])) if monomers_metadata_list else 1
        
        generate_monomer_com_files: bool = args.output_com in ['monomer', 'both']
        generate_aggregate_com_files: bool = args.output_com in ['dimer', 'both'] 

        if args.eetg: 
            if generate_aggregate_com_files: 
                write_to_log(f"Writing EETG {combined_system_term} .com file...") 
                eetg_has_qm_solvent_in_fragments = qm_solvent_flags.get('monomer_0_has_added_qm_solvent', False) or \
                                                 qm_solvent_flags.get('monomer_1_has_added_qm_solvent', False)
                solvent_desc_suffix = get_solvent_descriptor_suffix(eetg_has_qm_solvent_in_fragments, system_has_mm_solvent)
                
                eetg_filename_base = f"{combined_system_term}_eetg{solvent_desc_suffix}"
                eetg_filename = f"{eetg_filename_base}{tag_for_filename}.com"

                solvent_info_for_eetg_title = f" in {solvent_name_str}" if (eetg_has_qm_solvent_in_fragments or system_has_mm_solvent) else ""
                eetg_title = f"{aggregate_title_base}{solvent_info_for_eetg_title} EET Analysis".strip()
                
                m1_atoms_eetg, m1_coords_eetg = monomer_qm_regions[0]
                m2_atoms_eetg, m2_coords_eetg = monomer_qm_regions[1]

                write_com_file( # Using the unified writer
                    os.path.join(current_frame_output_dir, eetg_filename),
                    gaussian_keywords if gaussian_keywords is not None else [], 
                    eetg_title,
                    total_sys_charge, total_sys_spin,
                    [], np.array([]), # No single QM region for EETG wrapper
                    mm_charges_list=mm_solvent_point_charges, # System-wide MM solvent
                    fragment_definitions=[
                        (m1_atoms_eetg, m1_coords_eetg, monomers_metadata_list[0][JSON_KEY_CHARGE], monomers_metadata_list[0][JSON_KEY_SPIN_MULT]),
                        (m2_atoms_eetg, m2_coords_eetg, monomers_metadata_list[1][JSON_KEY_CHARGE], monomers_metadata_list[1][JSON_KEY_SPIN_MULT])
                    ]
                )
            elif not generate_aggregate_com_files and args.eetg : 
                 write_to_log(f"Skipping EETG {combined_system_term}: --output_com is set to 'monomer'.") 
        
        if generate_monomer_com_files:
            write_to_log(f"Writing VEE monomer .com files...") 
            for i in range(args.aggregate):
                if i < len(monomer_qm_regions): 
                    mono_meta = monomers_metadata_list[i]
                    mono_qm_atoms, mono_qm_coords = monomer_qm_regions[i]
                    
                    monomer_has_added_qm_solvent = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
                    solvent_desc_suffix = get_solvent_descriptor_suffix(monomer_has_added_qm_solvent, system_has_mm_solvent)
                    mono_vee_filename_base = f"monomer{i+1}{solvent_desc_suffix}"
                    mono_vee_filename = f"{mono_vee_filename_base}{tag_for_filename}.com"
                    
                    solvent_info_for_mono_title = f" in {solvent_name_str}" if (monomer_has_added_qm_solvent or system_has_mm_solvent) else ""
                    mono_vee_title = f"{mono_meta.get(JSON_KEY_NAME)} monomer{i+1}{solvent_info_for_mono_title}".strip() # Generalized from VEE
                    
                    # Combine mm_embedding for this monomer and system-wide mm_solvent
                    all_mm_for_monomer = []
                    if mm_embedding_for_monomer[i]:
                        all_mm_for_monomer.extend(mm_embedding_for_monomer[i])
                    if mm_solvent_point_charges: # mm_solvent_point_charges are (q,x,y,z)
                        # Convert to (x,y,z,q) for consistency if needed, or handle in write_com_file
                        # For now, assume write_com_file handles the (q,x,y,z) from mm_solvent_point_charges correctly
                        # and (x,y,z,q) from mm_embedding_for_monomer.
                        # Let's standardize to (x,y,z,q) for the combined list.
                        for q_s, x_s, y_s, z_s in mm_solvent_point_charges:
                            all_mm_for_monomer.append((x_s, y_s, z_s, q_s))


                    write_com_file(
                        os.path.join(current_frame_output_dir, mono_vee_filename),
                        gaussian_keywords if gaussian_keywords is not None else [], 
                        mono_vee_title,
                        mono_meta[JSON_KEY_CHARGE], mono_meta[JSON_KEY_SPIN_MULT],
                        mono_qm_atoms, mono_qm_coords, 
                        mm_charges_list=all_mm_for_monomer if all_mm_for_monomer else None
                    )
                else:
                    warn_msg = f"Data for monomer {i+1} not available in monomer_qm_regions. Skipping its VEE file."
                    print(f"WARNING: {warn_msg}")
                    write_to_log(warn_msg, is_warning=True)
        
        if generate_aggregate_com_files and not args.eetg:
            write_to_log(f"Writing VEE {combined_system_term} .com file...") 
            aggregate_has_added_qm_solvent = qm_solvent_flags.get('aggregate_has_added_qm_solvent', False)
            solvent_desc_suffix = get_solvent_descriptor_suffix(aggregate_has_added_qm_solvent, system_has_mm_solvent)
            agg_vee_filename_base = f"{combined_system_term}{solvent_desc_suffix}"
            agg_vee_filename = f"{agg_vee_filename_base}{tag_for_filename}.com"

            solvent_info_for_agg_title = f" in {solvent_name_str}" if (aggregate_has_added_qm_solvent or system_has_mm_solvent) else ""
            agg_vee_title = f"{aggregate_title_base}{solvent_info_for_agg_title}".strip() # Generalized from VEE
            
            agg_qm_atoms, agg_qm_coords = aggregate_qm_region
            
            # For aggregate, only system-wide MM solvent applies, not inter-monomer embedding
            mm_charges_for_aggregate = []
            if mm_solvent_point_charges:
                 for q_s, x_s, y_s, z_s in mm_solvent_point_charges:
                    mm_charges_for_aggregate.append((x_s, y_s, z_s, q_s))


            write_com_file(
                os.path.join(current_frame_output_dir, agg_vee_filename),
                gaussian_keywords if gaussian_keywords is not None else [], 
                agg_vee_title,
                total_sys_charge, total_sys_spin,
                agg_qm_atoms, agg_qm_coords, 
                mm_charges_list=mm_charges_for_aggregate if mm_charges_for_aggregate else None
            )

        if args.traj_xyz and os.path.exists(temp_xyz_input_filepath) and TEMP_FRAME_XYZ_FILENAME in temp_xyz_input_filepath :
            try:
                os.remove(temp_xyz_input_filepath)
                write_to_log(f"Cleaned up temporary file: {temp_xyz_input_filepath}")
            except OSError as e:
                warn_msg = f"Could not remove temporary file {temp_xyz_input_filepath}: {e}"
                print(f"WARNING: {warn_msg}")
                write_to_log(warn_msg, is_warning=True)
        
        if args.traj_xyz and total_frames_to_process > 0:
            sys.stdout.write(f"\rProcessed Frames: {frame_idx + 1} / {total_frames_to_process}")
            sys.stdout.flush()

    if args.traj_xyz and total_frames_to_process > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    final_message = "\nAll processing complete."
    print(final_message)
    write_to_log(final_message)

    if log_file_handle:
        log_file_handle.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        err_msg_critical = f"An unhandled error occurred: {e}"
        print(f"ERROR: {err_msg_critical}", file=sys.stderr)
        if log_file_handle: 
            write_to_log(err_msg_critical, is_error=True)
            import traceback
            write_to_log(traceback.format_exc())
            log_file_handle.close()
        sys.exit(1)
    finally:
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.close()
