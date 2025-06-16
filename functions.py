# Consolidated helper functions for giqpy
import argparse
import os
import json
import numpy as np
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union

# --- Constants ---
AUTO_MM_SOLVENT_TRIGGER = "auto_detect_mm_solvent_from_input_xyz"
TEMP_FRAME_XYZ_FILENAME = "_current_frame_data.xyz"
LOG_FILENAME = "giqpy_run.log"

# --- JSON keys ---
JSON_KEY_NAME = "name"
JSON_KEY_NATOMS = "nAtoms"
JSON_KEY_CHARGE = "charge"
JSON_KEY_SPIN_MULT = "spin_mult"
JSON_KEY_MOL_FORMULA = "mol_formula"
JSON_KEY_CHARGES_ARRAY = "charges"
JSON_KEY_ELEMENT = "element"

# --- Global log file handle ---
log_file_handle: Optional[Any] = None

# --- Dataclasses ---
@dataclass
class SolventCharge:
    element: str
    charge: float

@dataclass
class MonomerMeta:
    name: str
    nAtoms: int
    charge: float
    spin_mult: int
    mol_formula: str = ""

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class SolventMeta:
    name: str
    mol_formula: str
    nAtoms: int
    charges: List[SolventCharge]

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

# --- Type aliases ---
CoordType = np.ndarray
AtomListType = List[str]
ChargeListType = List[float]
SolventGroupType = Tuple[AtomListType, CoordType]
MMChargeTupleType = Tuple[float, float, float, float]

# --- Logging helper ---
def write_to_log(message: str, is_error: bool = False, is_warning: bool = False) -> None:
    """Write a message to the global log file if it is open."""
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

# --- CLI parser ---
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate Gaussian .com and .xyz files from trajectory XYZ with QM/MM options.",
    )
    parser.add_argument('--traj', required=True, help='Multi-frame trajectory XYZ file. Use --nFrames 1 for a single frame.')
    parser.add_argument('--nFrames', type=int, help='Number of frames to process (default: all).')
    parser.add_argument('--nDyes', type=int, required=True, help='Number of core monomer units.')
    parser.add_argument('--system_info', required=True, help='JSON file with monomer and solvent metadata.')
    parser.add_argument('--gauss_files', nargs='?', choices=['monomer', 'dimer', 'both'], const='both', default=None,
                        help='Generate Gaussian .com input files. If flag present without value, defaults to "both".')
    parser.add_argument('--gauss_keywords', help='File with Gaussian keywords (required with --gauss_files).')
    parser.add_argument('--qmSol_radius', type=float, default=5.0, help='Radius (Å) for QM solvent selection. Negative value disables QM solvent.')
    parser.add_argument('--mm_monomer', nargs='*', help='MM charges for embedding from other monomers.')
    parser.add_argument('--mm_solvent', nargs='?', const=AUTO_MM_SOLVENT_TRIGGER, default=None,
                        help='Include MM solvent. Provide file path or use flag alone for auto-detection.')
    parser.add_argument('--eetg', action='store_true', help='Generate only EETG input for dimers (requires --nDyes=2).')
    parser.add_argument('--tag', default='', help='Optional custom tag for generated filenames.')
    parser.add_argument('--logfile', default=LOG_FILENAME, help='Specify log file name.')
    return parser

# --- Trajectory utilities ---
def split_frames(traj_xyz_path: str, num_frames_to_extract: Optional[int], base_output_dir_for_traj: str) -> List[Tuple[str, str, str]]:
    processed_frames_info: List[Tuple[str, str, str]] = []
    try:
        with open(traj_xyz_path, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        write_to_log(f"Trajectory file not found: {traj_xyz_path}", is_error=True)
        raise
    if not lines:
        write_to_log(f"Empty trajectory: {traj_xyz_path}", is_error=True)
        raise ValueError(f"Empty trajectory: {traj_xyz_path}")
    try:
        total_atoms_per_frame = int(lines[0].strip())
    except ValueError:
        write_to_log(f"First line of trajectory {traj_xyz_path} must be the number of atoms.", is_error=True)
        raise
    block_size = total_atoms_per_frame + 2
    if block_size <= 2:
        write_to_log(f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.", is_error=True)
        raise ValueError(f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.")
    max_possible_frames = len(lines) // block_size
    actual_frames_to_process = num_frames_to_extract if num_frames_to_extract and num_frames_to_extract <= max_possible_frames else max_possible_frames
    if num_frames_to_extract and num_frames_to_extract > max_possible_frames:
        warn_msg = f"Requested {num_frames_to_extract} frames, but trajectory only contains {max_possible_frames}. Processing all available frames."
        write_to_log(warn_msg, is_warning=True)
    for i in range(actual_frames_to_process):
        frame_id_str = str(i + 1)
        output_dir_for_frame = os.path.join(base_output_dir_for_traj, frame_id_str)
        os.makedirs(output_dir_for_frame, exist_ok=True)
        temp_xyz_path_for_frame = os.path.join(output_dir_for_frame, TEMP_FRAME_XYZ_FILENAME)
        start_line_idx = i * block_size
        frame_block_lines = lines[start_line_idx : start_line_idx + block_size]
        with open(temp_xyz_path_for_frame, 'w') as f_temp_xyz:
            f_temp_xyz.write("\n".join(frame_block_lines) + "\n")
        processed_frames_info.append((frame_id_str, temp_xyz_path_for_frame, output_dir_for_frame))
    return processed_frames_info

def write_xyz(path: str, atoms: Union[AtomListType, ChargeListType], coords: CoordType, comment: str = "") -> None:
    coords_array = np.array(coords)
    if coords_array.ndim == 1 and len(atoms) > 1:
        coords_array = coords_array.reshape(len(atoms), 3)
    with open(path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom_sym, xyz_row in zip(atoms, coords_array):
            atom_col = f"{atom_sym:<12.5f}" if isinstance(atom_sym, (float, np.floating)) else f"{str(atom_sym):<3}"
            f.write(f"{atom_col} {xyz_row[0]:12.5f} {xyz_row[1]:12.5f} {xyz_row[2]:12.5f}\n")

# --- System metadata utilities ---
def load_system_info(system_info_path: str, n_dyes: int) -> Tuple[List[MonomerMeta], SolventMeta]:
    write_to_log(f"Loading system info from: {system_info_path}")
    with open(system_info_path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) < n_dyes + 1:
        raise ValueError(f"system_info must be a JSON array with at least {n_dyes + 1} entries ({n_dyes} monomers + 1 solvent).")
    monomer_objs: List[MonomerMeta] = []
    for mono in data[:n_dyes]:
        monomer_objs.append(MonomerMeta(name=mono[JSON_KEY_NAME], nAtoms=mono[JSON_KEY_NATOMS], charge=mono[JSON_KEY_CHARGE], spin_mult=mono[JSON_KEY_SPIN_MULT], mol_formula=mono.get(JSON_KEY_MOL_FORMULA, "")))
    sol = data[n_dyes]
    charges = [SolventCharge(element=c[JSON_KEY_ELEMENT], charge=c[JSON_KEY_CHARGE]) for c in sol[JSON_KEY_CHARGES_ARRAY]]
    solvent_obj = SolventMeta(name=sol.get(JSON_KEY_NAME, "solvent"), mol_formula=sol[JSON_KEY_MOL_FORMULA], nAtoms=sol[JSON_KEY_NATOMS], charges=charges)
    write_to_log("System info loaded and validated successfully.")
    return monomer_objs, solvent_obj

def parse_formula(fmt: str) -> Dict[str, int]:
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', fmt)
    counts: Dict[str, int] = {}
    for elem, num_str in tokens:
        counts[elem] = int(num_str) if num_str else 1
    return counts

def group_qm_molecules(atoms: AtomListType, coords: CoordType, fmt_counts: Dict[str, int]) -> List[SolventGroupType]:
    size = sum(fmt_counts.values())
    if size == 0:
        raise ValueError(f"Cannot group solvent molecules: formula '{fmt_counts}' results in zero atoms per molecule.")
    if len(atoms) % size != 0:
        raise ValueError(f"Total solvent atom count ({len(atoms)}) is not divisible by formula size ({size}). Ensure correct solvent definition.")
    groups: List[SolventGroupType] = []
    for i in range(0, len(atoms), size):
        groups.append((atoms[i:i+size], coords[i:i+size]))
    return groups

def is_within_radius(core_coords_list: List[CoordType], mol_coords_array: CoordType, radius: float) -> bool:
    if radius < 0:
        return False
    for core_coord_set in core_coords_list:
        diff = core_coord_set[:, np.newaxis, :] - mol_coords_array[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        if np.any(dists_sq < radius**2):
            return True
    return False

def flatten_groups(groups_of_molecules: List[SolventGroupType]) -> Tuple[AtomListType, CoordType]:
    flat_atoms: AtomListType = []
    flat_coords_list: List[CoordType] = []
    for atoms_in_mol, coords_in_mol in groups_of_molecules:
        flat_atoms.extend(atoms_in_mol)
        if coords_in_mol.size > 0:
            flat_coords_list.append(coords_in_mol)
    if not flat_coords_list:
        return flat_atoms, np.array([])
    return flat_atoms, np.vstack(flat_coords_list)

def localize_solvent_and_prepare_regions(
    main_xyz_input_filepath: str,
    system_info_path: str,
    n_dyes: int,
    qm_radius: float,
) -> Tuple[List[Tuple[AtomListType, CoordType]], Tuple[AtomListType, CoordType], List[SolventGroupType], Dict[str, bool], CoordType, List[int]]:
    """Localize QM solvent and prepare QM/MM regions."""
    monomers_meta, solvent_meta = load_system_info(system_info_path, n_dyes)
    n_atoms_per_monomer_list: List[int] = [m[JSON_KEY_NATOMS] for m in monomers_meta]
    total_core_atom_count_from_json = sum(n_atoms_per_monomer_list)

    loaded_main_coords_list: List[List[float]] = []
    loaded_main_atoms_list: AtomListType = []
    with open(main_xyz_input_filepath, 'r') as f_xyz:
        lines = f_xyz.readlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file {main_xyz_input_filepath} is too short.")
    num_atoms_in_main_file = int(lines[0].strip())
    if len(lines) < num_atoms_in_main_file + 2:
        raise ValueError(f"XYZ {main_xyz_input_filepath} has fewer lines ({len(lines)}) than expected based on atom count ({num_atoms_in_main_file + 2}).")
    for ln_idx, ln_content in enumerate(lines[2 : num_atoms_in_main_file + 2]):
        parts = ln_content.split()
        if len(parts) >= 4:
            loaded_main_atoms_list.append(parts[0])
            loaded_main_coords_list.append([float(p) for p in parts[1:4]])
        else:
            raise ValueError(f"Malformed line in {main_xyz_input_filepath} line {ln_idx+3}: {ln_content.strip()}")

    main_atoms_all_array = np.array(loaded_main_atoms_list, dtype=str)
    main_coords_all_array = np.array(loaded_main_coords_list)

    core_atoms_all_list = main_atoms_all_array[:total_core_atom_count_from_json].tolist()
    core_coords_all_for_dist_calc = main_coords_all_array[:total_core_atom_count_from_json]
    core_coords_to_use_for_qm = core_coords_all_for_dist_calc
    solvent_atoms_from_xyz_list = main_atoms_all_array[total_core_atom_count_from_json:].tolist()
    solvent_coords_from_xyz_array = main_coords_all_array[total_core_atom_count_from_json:]

    solvent_formula_counts = parse_formula(solvent_meta[JSON_KEY_MOL_FORMULA])
    all_solvent_groups_list: List[SolventGroupType] = group_qm_molecules(solvent_atoms_from_xyz_list, solvent_coords_from_xyz_array, solvent_formula_counts)

    aggregate_qm_solvent_groups: List[SolventGroupType] = [g for g in all_solvent_groups_list if is_within_radius([core_coords_all_for_dist_calc], g[1], qm_radius)]
    aggregate_qm_solvent_atoms_list, aggregate_qm_solvent_coords_array = flatten_groups(aggregate_qm_solvent_groups)
    aggregate_qm_group_ids = set(map(id, aggregate_qm_solvent_groups))
    non_qm_solvent_groups: List[SolventGroupType] = [g for g in all_solvent_groups_list if id(g) not in aggregate_qm_group_ids]

    monomer_localized_qm_solvent_groups_list: List[List[SolventGroupType]] = []
    current_core_idx_dist = 0
    for num_atoms_mono in n_atoms_per_monomer_list:
        monomer_core_coords_for_dist = core_coords_all_for_dist_calc[current_core_idx_dist : current_core_idx_dist + num_atoms_mono]
        localized_groups = [g for g in all_solvent_groups_list if is_within_radius([monomer_core_coords_for_dist], g[1], qm_radius)]
        monomer_localized_qm_solvent_groups_list.append(localized_groups)
        current_core_idx_dist += num_atoms_mono

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
    aggregate_final_qm_coords = np.vstack((core_coords_to_use_for_qm, aggregate_qm_solvent_coords_array)) if aggregate_qm_solvent_coords_array.size > 0 else core_coords_to_use_for_qm
    aggregate_qm_region_for_gauss: Tuple[AtomListType, CoordType] = (aggregate_final_qm_atoms, aggregate_final_qm_coords)

    monomer_qm_regions_for_gauss: List[Tuple[AtomListType, CoordType]] = []
    current_core_idx_qm = 0
    for i in range(len(monomers_meta)):
        num_atoms_in_this_mono = n_atoms_per_monomer_list[i]
        mono_core_atoms = core_atoms_all_list[current_core_idx_qm : current_core_idx_qm + num_atoms_in_this_mono]
        mono_core_coords_for_qm = core_coords_to_use_for_qm[current_core_idx_qm : current_core_idx_qm + num_atoms_in_this_mono]
        added_qm_atoms, added_qm_coords = monomer_unique_qm_solvent_data_list[i]
        final_mono_atoms = mono_core_atoms + added_qm_atoms
        final_mono_coords = np.vstack((mono_core_coords_for_qm, added_qm_coords)) if added_qm_coords.size > 0 else mono_core_coords_for_qm
        monomer_qm_regions_for_gauss.append((final_mono_atoms, final_mono_coords))
        current_core_idx_qm += num_atoms_in_this_mono

    return (
        monomer_qm_regions_for_gauss,
        aggregate_qm_region_for_gauss,
        non_qm_solvent_groups,
        qm_solvent_flags,
        core_coords_all_for_dist_calc,
        n_atoms_per_monomer_list,
    )

def load_monomer_charges_from_file(charge_file_path: str) -> List[MMChargeTupleType]:
    charges_data: List[MMChargeTupleType] = []
    try:
        with open(charge_file_path, 'r') as f:
            for line_num, line_content in enumerate(f):
                parts = line_content.strip().split()
                if len(parts) == 4:
                    try:
                        charge_val = float(parts[0])
                        x_val, y_val, z_val = float(parts[1]), float(parts[2]), float(parts[3])
                        charges_data.append((x_val, y_val, z_val, charge_val))
                    except ValueError:
                        warn_msg = f"Non-numeric value in charge file {charge_file_path} on line {line_num + 1}: {line_content.strip()}"
                        print(f"WARNING: {warn_msg}")
                        write_to_log(warn_msg, is_warning=True)
                elif parts:
                    warn_msg = f"Malformed line in charge file {charge_file_path} on line {line_num + 1} (expected 4 values, got {len(parts)}): {line_content.strip()}"
                    print(f"WARNING: {warn_msg}")
                    write_to_log(warn_msg, is_warning=True)
    except FileNotFoundError:
        warn_msg = f"MM Monomer charge file not found: {charge_file_path}. Returning empty charges for this file."
        print(f"WARNING: {warn_msg}")
        write_to_log(warn_msg, is_warning=True)
    return charges_data

def assign_charges_to_solvent_molecules(solvent_groups: List[SolventGroupType], solvent_metadata_entry: SolventMeta) -> List[MMChargeTupleType]:
    mm_solvent_charges_list: List[MMChargeTupleType] = []
    defined = solvent_metadata_entry.charges
    if len(defined) != solvent_metadata_entry.nAtoms:
        raise ValueError(f"Mismatch in solvent definition: nAtoms {solvent_metadata_entry.nAtoms} does not match charges array ({len(defined)}).")
    for atoms_in_mol, coords_in_mol in solvent_groups:
        if len(atoms_in_mol) != solvent_metadata_entry.nAtoms:
            write_to_log(f"Solvent molecule has {len(atoms_in_mol)} atoms but {solvent_metadata_entry.nAtoms} defined. Skipping.", is_warning=True)
            continue
        for idx, _ in enumerate(atoms_in_mol):
            charge_value = defined[idx].charge
            c = coords_in_mol[idx]
            mm_solvent_charges_list.append((c[0], c[1], c[2], charge_value))
    return mm_solvent_charges_list

def calculate_centroid_distance_between_first_two(core_coords_all_array: CoordType, n_atoms_per_monomer_list: List[int]) -> Optional[float]:
    if len(n_atoms_per_monomer_list) < 2:
        return None
    n_atoms_m1 = n_atoms_per_monomer_list[0]
    n_atoms_m2 = n_atoms_per_monomer_list[1]
    if core_coords_all_array.shape[0] < n_atoms_m1 + n_atoms_m2:
        return None
    centroid_m1 = np.mean(core_coords_all_array[:n_atoms_m1], axis=0)
    centroid_m2 = np.mean(core_coords_all_array[n_atoms_m1:n_atoms_m1+n_atoms_m2], axis=0)
    return float(np.linalg.norm(centroid_m1 - centroid_m2))

# --- Gaussian utilities ---
def load_keywords_from_file(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def get_solvent_descriptor_suffix(entity_has_added_qm_solvent: bool, system_has_mm_solvent: bool) -> str:
    if entity_has_added_qm_solvent and system_has_mm_solvent:
        return "_qm_mm"
    elif entity_has_added_qm_solvent:
        return "_qm"
    elif system_has_mm_solvent:
        return "_mm"
    else:
        return ""

def write_com_file(path: str, keywords: List[str], title: str, charge: Union[int, float], spin: int, atoms_list: AtomListType, coords_array: CoordType, mm_charges_list: Optional[List[MMChargeTupleType]] = None, fragment_definitions: Optional[List[Tuple[AtomListType, CoordType, int, int]]] = None) -> None:
    final_keywords = list(keywords)
    if mm_charges_list:
        charge_keyword_present = any(re.fullmatch(r"#.*charge", kw.lower().strip()) or kw.lower().strip() == "charge" for kw in final_keywords)
        if not charge_keyword_present:
            inserted = False
            for i, kw_line in enumerate(final_keywords):
                if kw_line.strip().startswith("#"):
                    final_keywords.insert(i + 1, "# charge")
                    inserted = True
                    break
            if not inserted:
                final_keywords.append("# charge")
    with open(path, "w") as f:
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
                    f.write(f" {atom_sym}(Fragment={frag_idx+1}) {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")
        else:
            f.write(f"{charge} {spin}\n")
            for atom_sym, xyz_coords in zip(atoms_list, coords_array):
                f.write(f" {atom_sym} {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")
        if mm_charges_list:
            f.write("\n")
            for x_mm, y_mm, z_mm, q_mm in mm_charges_list:
                f.write(f" {x_mm:14.5f} {y_mm:12.5f} {z_mm:12.5f} {q_mm:12.5f}\n")
        f.write("\n")
