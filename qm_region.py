import json
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional

from constants import (
    JSON_KEY_NAME,
    JSON_KEY_NATOMS,
    JSON_KEY_CHARGE,
    JSON_KEY_SPIN_MULT,
    JSON_KEY_MOL_FORMULA,
    JSON_KEY_CHARGES_ARRAY,
    JSON_KEY_ELEMENT,
)
from logconfig import write_to_log
from datatypes import (
    AtomListType,
    CoordType,
    SolventGroupType,
    MMChargeTupleType,
    MonomerMeta,
    SolventMeta,
    SolventCharge,
)


def load_system_info(system_info_path: str, n_dyes: int) -> Tuple[List[MonomerMeta], SolventMeta]:
    """Load monomer and solvent metadata from a JSON file."""
    write_to_log(f"Loading system info from: {system_info_path}")
    with open(system_info_path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) < n_dyes + 1:
        raise ValueError(
            f"system_info must be a JSON array with at least {n_dyes + 1} entries ({n_dyes} monomers + 1 solvent)."
        )
    monomer_objs: List[MonomerMeta] = []
    for mono in data[:n_dyes]:
        monomer_objs.append(
            MonomerMeta(
                name=mono[JSON_KEY_NAME],
                nAtoms=mono[JSON_KEY_NATOMS],
                charge=mono[JSON_KEY_CHARGE],
                spin_mult=mono[JSON_KEY_SPIN_MULT],
                mol_formula=mono.get(JSON_KEY_MOL_FORMULA, ""),
            )
        )
    sol = data[n_dyes]
    charges = [SolventCharge(element=c[JSON_KEY_ELEMENT], charge=c[JSON_KEY_CHARGE]) for c in sol[JSON_KEY_CHARGES_ARRAY]]
    solvent_obj = SolventMeta(
        name=sol.get(JSON_KEY_NAME, "solvent"),
        mol_formula=sol[JSON_KEY_MOL_FORMULA],
        nAtoms=sol[JSON_KEY_NATOMS],
        charges=charges,
    )
    write_to_log("System info loaded and validated successfully.")
    return monomer_objs, solvent_obj


def parse_formula(fmt: str) -> Dict[str, int]:
    """Parse molecular formula into element counts."""
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", fmt)
    counts: Dict[str, int] = {}
    for elem, num_str in tokens:
        counts[elem] = int(num_str) if num_str else 1
    return counts


def group_qm_molecules(atoms: AtomListType, coords: CoordType, fmt_counts: Dict[str, int]) -> List[SolventGroupType]:
    """Group solvent atoms into molecules using the formula counts."""
    size = sum(fmt_counts.values())
    if size == 0:
        raise ValueError(f"Cannot group solvent molecules: formula '{fmt_counts}' results in zero atoms per molecule.")
    if len(atoms) % size != 0:
        raise ValueError(
            f"Total solvent atom count ({len(atoms)}) is not divisible by formula size ({size}). Ensure correct solvent definition."
        )
    groups: List[SolventGroupType] = []
    for i in range(0, len(atoms), size):
        groups.append((atoms[i:i+size], coords[i:i+size]))
    return groups


def is_within_radius(core_coords_list: List[CoordType], mol_coords_array: CoordType, radius: float) -> bool:
    """Check if any atom in mol_coords_array lies within radius of any core atom set."""
    if radius < 0:
        return False
    for core_coord_set in core_coords_list:
        diff = core_coord_set[:, np.newaxis, :] - mol_coords_array[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        if np.any(dists_sq < radius**2):
            return True
    return False


def flatten_groups(groups_of_molecules: List[SolventGroupType]) -> Tuple[AtomListType, CoordType]:
    """Flatten grouped molecules to atom list and coordinate array."""
    flat_atoms: AtomListType = []
    flat_coords_list: List[CoordType] = []
    for atoms_in_mol, coords_in_mol in groups_of_molecules:
        flat_atoms.extend(atoms_in_mol)
        if coords_in_mol.size > 0:
            flat_coords_list.append(coords_in_mol)
    if not flat_coords_list:
        return flat_atoms, np.array([])
    return flat_atoms, np.vstack(flat_coords_list)


def load_monomer_charges_from_file(charge_file_path: str) -> List[MMChargeTupleType]:
    """Load MM charges from a file for monomer embedding."""
    charges_data: List[MMChargeTupleType] = []
    try:
        with open(charge_file_path, 'r') as f:
            for line_content in f:
                parts = line_content.strip().split()
                if len(parts) == 4:
                    charge_val = float(parts[0])
                    x_val, y_val, z_val = float(parts[1]), float(parts[2]), float(parts[3])
                    charges_data.append((x_val, y_val, z_val, charge_val))
    except FileNotFoundError:
        write_to_log(f"MM Monomer charge file not found: {charge_file_path}.", is_warning=True)
    return charges_data


def assign_charges_to_solvent_molecules(solvent_groups: List[SolventGroupType], solvent_metadata_entry: SolventMeta) -> List[MMChargeTupleType]:
    """Assign charges to solvent molecules based on metadata."""
    mm_solvent_charges_list: List[MMChargeTupleType] = []
    defined = solvent_metadata_entry.charges
    if len(defined) != solvent_metadata_entry.nAtoms:
        raise ValueError(
            f"Mismatch in solvent definition: nAtoms {solvent_metadata_entry.nAtoms} does not match charges array ({len(defined)})."
        )
    for atoms_in_mol, coords_in_mol in solvent_groups:
        if len(atoms_in_mol) != solvent_metadata_entry.nAtoms:
            write_to_log(
                f"Solvent molecule has {len(atoms_in_mol)} atoms but {solvent_metadata_entry.nAtoms} defined. Skipping.",
                is_warning=True,
            )
            continue
        for idx, _ in enumerate(atoms_in_mol):
            charge_value = defined[idx].charge
            c = coords_in_mol[idx]
            mm_solvent_charges_list.append((c[0], c[1], c[2], charge_value))
    return mm_solvent_charges_list


def calculate_centroid_distance_between_first_two(core_coords_all_array: CoordType, n_atoms_per_monomer_list: List[int]) -> Optional[float]:
    """Calculate distance between centroids of the first two monomers."""
    if len(n_atoms_per_monomer_list) < 2:
        return None
    n_atoms_m1 = n_atoms_per_monomer_list[0]
    n_atoms_m2 = n_atoms_per_monomer_list[1]
    if core_coords_all_array.shape[0] < n_atoms_m1 + n_atoms_m2:
        return None
    centroid_m1 = np.mean(core_coords_all_array[:n_atoms_m1], axis=0)
    centroid_m2 = np.mean(core_coords_all_array[n_atoms_m1:n_atoms_m1+n_atoms_m2], axis=0)
    return float(np.linalg.norm(centroid_m1 - centroid_m2))
