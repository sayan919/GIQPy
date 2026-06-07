"""Shared utility functions for GIQPy (giqpy.py and xyz_to_gaussian.py)."""

import os
import json
import numpy as np
import re
import sys
from typing import List, Tuple, Optional, Dict, Any, Union

# --- Constants ---
AUTO_MM_SOLVENT_TRIGGER: str = "auto_detect_mm_solvent_from_input_xyz"
TEMP_FRAME_XYZ_FILENAME: str = "_current_frame_data.xyz"

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
MMChargeTupleType = Tuple[float, float, float, float]  # (x, y, z, charge)


# --- Helper for logging to file ---
def write_to_log(message: str, is_error: bool = False, is_warning: bool = False) -> None:
    """Writes a message to the global log file if it's open."""
    if log_file_handle:
        prefix = "ERROR: " if is_error else ("WARNING: " if is_warning else "")
        try:
            log_file_handle.write(prefix + message + "\n")
            log_file_handle.flush()
        except IOError:
            print(f"FALLBACK CONSOLE (log write error): {prefix}{message}", file=sys.stderr)


# --- Frame handling ---
def split_frames(
    traj_xyz_path: str,
    num_frames_to_extract: Optional[int],
    base_output_dir: str,
) -> List[Tuple[str, str, str]]:
    """
    Split a (possibly multi-frame) trajectory XYZ into per-frame directories.

    For a single-frame input simply use --nFrames 1.
    Returns a list of (frame_id_str, path_to_xyz_for_frame, output_dir_for_frame) tuples.
    """
    try:
        with open(traj_xyz_path, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        err = f"Trajectory file not found: {traj_xyz_path}"
        print(f"ERROR: {err}", file=sys.stderr)
        write_to_log(err, is_error=True)
        raise

    if not lines:
        err = f"Empty trajectory: {traj_xyz_path}"
        print(f"ERROR: {err}", file=sys.stderr)
        write_to_log(err, is_error=True)
        raise ValueError(err)

    try:
        total_atoms_per_frame = int(lines[0].strip())
    except ValueError:
        err = f"First line of trajectory {traj_xyz_path} must be the number of atoms."
        print(f"ERROR: {err}", file=sys.stderr)
        write_to_log(err, is_error=True)
        raise

    block_size = total_atoms_per_frame + 2
    if block_size <= 2:
        err = f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}."
        print(f"ERROR: {err}", file=sys.stderr)
        write_to_log(err, is_error=True)
        raise ValueError(err)

    max_possible_frames = len(lines) // block_size
    if num_frames_to_extract and num_frames_to_extract <= max_possible_frames:
        frames_to_process = num_frames_to_extract
    else:
        frames_to_process = max_possible_frames
        if num_frames_to_extract and num_frames_to_extract > max_possible_frames:
            warn = (f"Requested {num_frames_to_extract} frames, but trajectory only contains "
                    f"{max_possible_frames}. Processing all available frames.")
            print(f"WARNING: {warn}")
            write_to_log(warn, is_warning=True)

    processed_frames_info: List[Tuple[str, str, str]] = []
    for i in range(frames_to_process):
        frame_id_str = str(i + 1)
        output_dir_for_frame = os.path.join(base_output_dir, frame_id_str)
        try:
            os.makedirs(output_dir_for_frame, exist_ok=True)
        except OSError as e:
            err = f"Could not create frame-specific output directory {output_dir_for_frame}: {e}"
            print(f"ERROR: {err}", file=sys.stderr)
            write_to_log(err, is_error=True)
            raise

        temp_xyz_path_for_frame = os.path.join(output_dir_for_frame, TEMP_FRAME_XYZ_FILENAME)
        start = i * block_size
        frame_block_lines = lines[start: start + block_size]
        try:
            with open(temp_xyz_path_for_frame, 'w') as f_temp:
                f_temp.write("\n".join(frame_block_lines) + "\n")
        except IOError as e:
            err = f"Could not write temporary frame file {temp_xyz_path_for_frame}: {e}"
            print(f"ERROR: {err}", file=sys.stderr)
            write_to_log(err, is_error=True)
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
        monomer_id_for_log = monomer.get(JSON_KEY_NAME, f"monomer_at_index_{i}") if isinstance(monomer, dict) else f"monomer_at_index_{i}"
        if not isinstance(monomer, dict):
            raise ValueError(f"Monomer entry at index {i} is not a valid JSON object.")

        required_keys_monomer: Dict[str, Any] = {
            JSON_KEY_NAME: str,
            JSON_KEY_NATOMS: int,
            JSON_KEY_CHARGE: (int, float),
            JSON_KEY_SPIN_MULT: int,
        }
        for key, expected_type in required_keys_monomer.items():
            if key not in monomer:
                raise ValueError(f"Monomer '{monomer_id_for_log}' is missing required key: '{key}'.")
            if not isinstance(monomer[key], expected_type):  # type: ignore
                raise ValueError(f"Monomer '{monomer_id_for_log}' key '{key}' has incorrect type. "
                                 f"Expected {expected_type}, got {type(monomer[key])}.")

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
        JSON_KEY_CHARGES_ARRAY: list,
    }
    solvent_id_for_log = solvent_data.get(JSON_KEY_NAME, "solvent")
    solvent_data[JSON_KEY_NAME] = solvent_id_for_log  # Ensure name key exists for logging

    for key, expected_type in required_keys_solvent.items():
        if key not in solvent_data:
            raise ValueError(f"Solvent entry '{solvent_id_for_log}' is missing required key: '{key}'.")
        if not isinstance(solvent_data[key], expected_type):  # type: ignore
            raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{key}' has incorrect type. "
                             f"Expected {expected_type}, got {type(solvent_data[key])}.")

    if solvent_data[JSON_KEY_NATOMS] <= 0:
        raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{JSON_KEY_NATOMS}' must be a positive integer.")
    if not solvent_data[JSON_KEY_CHARGES_ARRAY] and solvent_data[JSON_KEY_NATOMS] > 0:
        raise ValueError(f"Solvent entry '{solvent_id_for_log}' key '{JSON_KEY_CHARGES_ARRAY}' must be a non-empty list when nAtoms > 0.")
    if len(solvent_data[JSON_KEY_CHARGES_ARRAY]) != solvent_data[JSON_KEY_NATOMS]:
        raise ValueError(f"Solvent entry '{solvent_id_for_log}': Number of entries in '{JSON_KEY_CHARGES_ARRAY}' "
                         f"({len(solvent_data[JSON_KEY_CHARGES_ARRAY])}) does not match '{JSON_KEY_NATOMS}' "
                         f"({solvent_data[JSON_KEY_NATOMS]}).")

    for idx, charge_entry in enumerate(solvent_data[JSON_KEY_CHARGES_ARRAY]):
        if not isinstance(charge_entry, dict):
            raise ValueError(f"Solvent '{solvent_id_for_log}', charge entry at index {idx} in '{JSON_KEY_CHARGES_ARRAY}' is not a valid JSON object.")
        if JSON_KEY_ELEMENT not in charge_entry or not isinstance(charge_entry[JSON_KEY_ELEMENT], str):
            raise ValueError(f"Solvent '{solvent_id_for_log}', charge entry at index {idx}: missing or invalid '{JSON_KEY_ELEMENT}'.")
        if JSON_KEY_CHARGE not in charge_entry or not isinstance(charge_entry[JSON_KEY_CHARGE], (int, float)):
            raise ValueError(f"Solvent '{solvent_id_for_log}', charge entry at index {idx}: missing or invalid '{JSON_KEY_CHARGE}'.")

    write_to_log("System info loaded and validated successfully.")
    return monomers_data, solvent_data


def parse_formula(fmt: str) -> Dict[str, int]:
    """Parse molecular formula into element counts. e.g., "H2O" -> {'H':2, 'O':1}."""
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', fmt)
    counts: Dict[str, int] = {}
    for elem, num_str in tokens:
        if elem:
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
    if size == 0:
        msg = f"Cannot group solvent molecules: formula '{fmt_counts}' results in zero atoms per molecule."
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
        raise ValueError(msg)
    if len(atoms) % size != 0:
        msg = (f"Total solvent atom count ({len(atoms)}) is not divisible by formula size ({size}). "
               f"Ensure correct solvent definition and XYZ content.")
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
        raise ValueError(msg)

    groups: List[SolventGroupType] = []
    for i in range(0, len(atoms), size):
        groups.append((atoms[i:i + size], coords[i:i + size]))
    return groups


def min_atom_distance(core_coords: CoordType, mol_coords: CoordType) -> float:
    """Minimum atom-atom distance between a core coordinate set and a molecule's coordinates."""
    if core_coords.size == 0 or mol_coords.size == 0:
        return float('inf')
    diff = core_coords[:, np.newaxis, :] - mol_coords[np.newaxis, :, :]
    return float(np.sqrt(np.min(np.sum(diff ** 2, axis=2))))


def flatten_groups(groups_of_molecules: List[SolventGroupType]) -> Tuple[AtomListType, CoordType]:
    """Flatten grouped molecules into a single atom list and a single coord array."""
    flat_atoms: AtomListType = []
    flat_coords_list: List[CoordType] = []
    for atoms_in_mol, coords_in_mol in groups_of_molecules:
        flat_atoms.extend(atoms_in_mol)
        if coords_in_mol.size > 0:
            flat_coords_list.append(coords_in_mol)
    if not flat_coords_list:
        return flat_atoms, np.empty((0, 3))
    return flat_atoms, np.vstack(flat_coords_list)


def read_xyz(path: str) -> Tuple[AtomListType, CoordType, str]:
    """
    Read a standard XYZ file (atom-count line, comment line, then 'element x y z' rows).
    Returns (atoms, coords (N,3) array, comment).
    """
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        err = f"XYZ file not found: {path}"
        print(f"ERROR: {err}", file=sys.stderr)
        write_to_log(err, is_error=True)
        raise

    if len(lines) < 2:
        raise ValueError(f"XYZ file {path} is too short.")
    try:
        n_atoms = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError(f"First line of XYZ {path} must be an atom count.")
    if len(lines) < n_atoms + 2:
        raise ValueError(f"XYZ {path} has fewer lines ({len(lines)}) than expected for {n_atoms} atoms.")

    comment = lines[1].rstrip("\n")
    atoms: AtomListType = []
    coords: List[List[float]] = []
    for i, ln in enumerate(lines[2:n_atoms + 2]):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed line in {path} line {i + 3}: {ln.strip()}")
        atoms.append(parts[0])
        try:
            coords.append([float(p) for p in parts[1:4]])
        except ValueError:
            raise ValueError(f"Malformed coordinate in {path} line {i + 3}: {ln.strip()}")
    return atoms, np.array(coords), comment


def write_xyz(path: str, atoms: Union[AtomListType, ChargeListType], coords: CoordType, comment: str = "") -> None:
    """
    Export atoms and coords to an XYZ file.
    The 'atoms' list can contain strings (element symbols) or floats (charges for MM XYZ).
    """
    coords_array = np.asarray(coords, dtype=float)
    if coords_array.size == 0:
        coords_array = coords_array.reshape(0, 3)
    elif coords_array.ndim == 1:
        coords_array = coords_array.reshape(-1, 3)
    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise ValueError(f"Coordinates for {path} must be shape (N, 3); got {coords_array.shape}.")
    if coords_array.shape[0] != len(atoms):
        raise ValueError(f"Mismatch between number of atoms ({len(atoms)}) and coordinate rows ({coords_array.shape[0]}) for {path}.")

    try:
        with open(path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for atom_symbol_or_charge, xyz_row in zip(atoms, coords_array):
                if isinstance(atom_symbol_or_charge, (float, np.floating)):
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
    monomers_meta: List[Dict[str, Any]],
    solvent_meta: Dict[str, Any],
    qm_radius: float,
) -> Tuple[List[SolventGroupType], SolventGroupType, List[SolventGroupType], Dict[str, bool], CoordType, List[int]]:
    """
    Read the input XYZ and partition QM solvent into per-monomer and aggregate regions.

    Each solvent molecule within ``qm_radius`` of the core is assigned to the SINGLE nearest
    monomer (by minimum atom-atom distance). This guarantees that no solvent atom is ever shared
    between two monomers' QM regions, while the aggregate QM region is the union of all selected
    solvent molecules.

    Returns:
        monomer_qm_regions    : list of (atoms, coords) per monomer (core + its unique QM solvent)
        aggregate_qm_region   : (atoms, coords) for the full aggregate (core + all QM solvent)
        non_qm_solvent_groups : solvent molecules outside the QM shell
        qm_solvent_flags      : booleans 'monomer_i_has_added_qm_solvent' / 'aggregate_has_added_qm_solvent'
        core_coords           : (Ncore, 3) core coordinates (used for centroid distances)
        n_atoms_per_monomer   : per-monomer core atom counts
    """
    n_atoms_per_monomer_list: List[int] = [m[JSON_KEY_NATOMS] for m in monomers_meta]
    total_core_atom_count = sum(n_atoms_per_monomer_list)

    atoms_all, coords_all, _ = read_xyz(main_xyz_input_filepath)
    if len(atoms_all) < total_core_atom_count:
        msg = (f"Input XYZ '{main_xyz_input_filepath}' has fewer atoms ({len(atoms_all)}) "
               f"than the core requires ({total_core_atom_count}).")
        print(f"ERROR: {msg}", file=sys.stderr)
        write_to_log(msg, is_error=True)
        raise ValueError(msg)

    core_atoms = atoms_all[:total_core_atom_count]
    core_coords = coords_all[:total_core_atom_count]
    solvent_atoms = atoms_all[total_core_atom_count:]
    solvent_coords = coords_all[total_core_atom_count:]

    # Group solvent atoms into whole molecules.
    solvent_formula_counts = parse_formula(solvent_meta[JSON_KEY_MOL_FORMULA])
    all_solvent_groups = group_qm_molecules(solvent_atoms, solvent_coords, solvent_formula_counts)

    # Per-monomer core coordinate slices.
    monomer_core_coords: List[CoordType] = []
    idx = 0
    for n in n_atoms_per_monomer_list:
        monomer_core_coords.append(core_coords[idx:idx + n])
        idx += n

    # Single pass: assign each solvent molecule to its nearest monomer (if within radius).
    monomer_groups: List[List[SolventGroupType]] = [[] for _ in n_atoms_per_monomer_list]
    aggregate_groups: List[SolventGroupType] = []
    non_qm_solvent_groups: List[SolventGroupType] = []
    for group in all_solvent_groups:
        g_coords = group[1]
        dmins = [min_atom_distance(mc, g_coords) for mc in monomer_core_coords]
        nearest = int(np.argmin(dmins))
        if qm_radius >= 0 and dmins[nearest] < qm_radius:
            monomer_groups[nearest].append(group)
            aggregate_groups.append(group)
        else:
            non_qm_solvent_groups.append(group)

    # Aggregate QM region.
    agg_sol_atoms, agg_sol_coords = flatten_groups(aggregate_groups)
    aggregate_qm_atoms = core_atoms + agg_sol_atoms
    aggregate_qm_coords = np.vstack((core_coords, agg_sol_coords)) if agg_sol_coords.size > 0 else core_coords
    aggregate_qm_region: SolventGroupType = (aggregate_qm_atoms, aggregate_qm_coords)

    # Per-monomer QM regions and flags.
    qm_solvent_flags: Dict[str, bool] = {}
    monomer_qm_regions: List[SolventGroupType] = []
    idx = 0
    for i, n in enumerate(n_atoms_per_monomer_list):
        mono_core_atoms = core_atoms[idx:idx + n]
        mono_core_coords = core_coords[idx:idx + n]
        idx += n
        add_atoms, add_coords = flatten_groups(monomer_groups[i])
        final_atoms = mono_core_atoms + add_atoms
        final_coords = np.vstack((mono_core_coords, add_coords)) if add_coords.size > 0 else mono_core_coords
        monomer_qm_regions.append((final_atoms, final_coords))
        qm_solvent_flags[f'monomer_{i}_has_added_qm_solvent'] = bool(add_atoms)
    qm_solvent_flags['aggregate_has_added_qm_solvent'] = bool(agg_sol_atoms)

    return (monomer_qm_regions, aggregate_qm_region, non_qm_solvent_groups,
            qm_solvent_flags, core_coords, n_atoms_per_monomer_list)


def read_charge_file(path: str, skip_header: int = 0) -> List[MMChargeTupleType]:
    """
    Read a 'charge x y z' point-charge file. Skips ``skip_header`` leading lines
    (use 2 for XYZ-like MM files, 0 for bare charge lists).
    Returns a list of (x, y, z, charge) tuples for Gaussian compatibility.
    """
    charges_data: List[MMChargeTupleType] = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        warn_msg = f"Charge file not found: {path}. Returning empty charge list."
        print(f"WARNING: {warn_msg}")
        write_to_log(warn_msg, is_warning=True)
        return charges_data

    for offset, line_content in enumerate(lines[skip_header:]):
        line_num = offset + skip_header + 1
        parts = line_content.split()
        if len(parts) == 4:
            try:
                q, x, y, z = (float(p) for p in parts)  # file format is: charge x y z
                charges_data.append((x, y, z, q))
            except ValueError:
                warn_msg = f"Non-numeric value in charge file {path} on line {line_num}: {line_content.strip()}"
                print(f"WARNING: {warn_msg}")
                write_to_log(warn_msg, is_warning=True)
        elif parts:
            warn_msg = (f"Malformed line in charge file {path} on line {line_num} "
                        f"(expected 4 values, got {len(parts)}): {line_content.strip()}")
            print(f"WARNING: {warn_msg}")
            write_to_log(warn_msg, is_warning=True)
    return charges_data


def load_monomer_charges_from_file(charge_file_path: str) -> List[MMChargeTupleType]:
    """Load MM monomer-embedding charges (format: 'charge x y z', no header)."""
    return read_charge_file(charge_file_path, skip_header=0)


def assign_charges_to_solvent_molecules(solvent_groups: List[SolventGroupType], solvent_metadata_entry: Dict[str, Any]) -> List[MMChargeTupleType]:
    """
    Assign charges to solvent molecules based on the solvent metadata template.
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
        write_to_log(msg, is_error=True)
        raise ValueError(msg)

    for mol_idx, (atom_symbols_in_molecule, coords_for_molecule) in enumerate(solvent_groups):
        if len(atom_symbols_in_molecule) != num_atoms_per_solvent_mol_defined:
            warn_msg = (f"Solvent molecule {mol_idx + 1} (auto-detected for MM) has {len(atom_symbols_in_molecule)} atoms, "
                        f"but system_info defines {num_atoms_per_solvent_mol_defined} for solvent "
                        f"'{solvent_metadata_entry.get(JSON_KEY_NAME, 'Unnamed Solvent')}'. Skipping this molecule for MM charges.")
            print(f"WARNING: {warn_msg}")
            write_to_log(warn_msg, is_warning=True)
            continue

        for atom_idx_in_mol in range(len(atom_symbols_in_molecule)):
            atom_symbol_from_xyz = atom_symbols_in_molecule[atom_idx_in_mol]
            atom_coord = coords_for_molecule[atom_idx_in_mol]

            charge_info_for_this_atom_pos = defined_charges_per_mol[atom_idx_in_mol]
            charge_value: float = charge_info_for_this_atom_pos[JSON_KEY_CHARGE]
            defined_element_for_pos: str = charge_info_for_this_atom_pos[JSON_KEY_ELEMENT]

            if atom_symbol_from_xyz != defined_element_for_pos:
                warn_msg = (f"Element mismatch for atom {atom_idx_in_mol + 1} in auto-detected MM solvent molecule {mol_idx + 1}. "
                            f"XYZ has '{atom_symbol_from_xyz}', system_info expects '{defined_element_for_pos}' at this position. "
                            f"Using charge defined for '{defined_element_for_pos}' ({charge_value}).")
                print(f"WARNING: {warn_msg}")
                write_to_log(warn_msg, is_warning=True)

            mm_solvent_charges_list.append((atom_coord[0], atom_coord[1], atom_coord[2], charge_value))

    return mm_solvent_charges_list


# --- Gaussian .com helpers (used by xyz_to_gaussian.py) ---
def get_solvent_descriptor_suffix(entity_has_added_qm_solvent: bool, system_has_mm_solvent: bool) -> str:
    """Solvent descriptor part of a .com filename (without extension)."""
    if entity_has_added_qm_solvent and system_has_mm_solvent:
        return "_qm_mm"
    if entity_has_added_qm_solvent:
        return "_qm"
    if system_has_mm_solvent:
        return "_mm"
    return ""


def get_solvent_title_fragment(entity_has_added_qm_solvent: bool, system_has_mm_solvent: bool, solvent_name: str) -> str:
    """Human-readable ' with QM/MM <solvent>' fragment for a .com title."""
    if entity_has_added_qm_solvent and system_has_mm_solvent:
        return f" with QM & MM {solvent_name}"
    if entity_has_added_qm_solvent:
        return f" with QM {solvent_name}"
    if system_has_mm_solvent:
        return f" with MM {solvent_name}"
    return ""


def write_com_file(
    path: str, keywords: List[str], title: str, charge: Union[int, float], spin: int,
    atoms_list: AtomListType, coords_array: CoordType,
    mm_charges_list: Optional[List[MMChargeTupleType]] = None,
    fragment_definitions: Optional[List[Tuple[AtomListType, CoordType, int, int]]] = None,
) -> None:
    """
    Write a Gaussian .com input file.
    mm_charges_list format: (x, y, z, q).
    fragment_definitions: [(atoms1, coords1, chg1, spin1), (atoms2, coords2, chg2, spin2)]
    """
    final_keywords = list(keywords)
    if mm_charges_list:
        charge_keyword_present = any(
            re.fullmatch(r"#.*charge", kw.lower().strip()) or kw.lower().strip() == "charge"
            for kw in final_keywords
        )
        if not charge_keyword_present:
            inserted = False
            for i, kw_line in enumerate(final_keywords):
                if kw_line.strip().startswith("#"):
                    final_keywords.insert(i + 1, "# charge")
                    inserted = True
                    break
            if not inserted:
                final_keywords.append("# charge")

    try:
        with open(path, 'w') as f:
            f.write(f"%chk={os.path.splitext(os.path.basename(path))[0]}.chk\n")
            for kw_line in final_keywords:
                f.write(kw_line + "\n")
            f.write("\n" + title + "\n\n")

            if fragment_definitions:
                f.write(f"{charge} {spin}")
                for _, _, chg_frag, spin_frag in fragment_definitions:
                    f.write(f" {chg_frag} {spin_frag}")
                f.write("\n")
                for frag_idx, (frag_atoms, frag_coords, _, _) in enumerate(fragment_definitions):
                    for atom_sym, xyz_coords in zip(frag_atoms, frag_coords):
                        f.write(f" {atom_sym}(Fragment={frag_idx + 1}) {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")
            else:
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
    """Load Gaussian route keywords from a text file (one per line, blanks dropped)."""
    try:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        err_msg = f"Keywords file not found: {path}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        write_to_log(err_msg, is_error=True)
        raise


def calculate_centroid_distance_between_first_two(core_coords_all_array: CoordType, n_atoms_per_monomer_list: List[int]) -> Optional[float]:
    """
    Distance between the centroids of the first two monomers' core regions, or None if not applicable.
    """
    if len(n_atoms_per_monomer_list) < 2:
        return None

    n_atoms_m1 = n_atoms_per_monomer_list[0]
    n_atoms_m2 = n_atoms_per_monomer_list[1]
    if n_atoms_m1 == 0 or n_atoms_m2 == 0:
        write_to_log("Cannot calculate centroid distance: a monomer has 0 atoms defined in system_info.")
        return None
    if core_coords_all_array.shape[0] < n_atoms_m1 + n_atoms_m2:
        write_to_log(f"Cannot calculate centroid distance: core coords array has {core_coords_all_array.shape[0]} atoms, "
                     f"but M1+M2 expect {n_atoms_m1 + n_atoms_m2}.")
        return None

    centroid_m1 = np.mean(core_coords_all_array[0:n_atoms_m1], axis=0)
    centroid_m2 = np.mean(core_coords_all_array[n_atoms_m1:n_atoms_m1 + n_atoms_m2], axis=0)
    return float(np.linalg.norm(centroid_m1 - centroid_m2))
