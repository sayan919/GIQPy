import os
import re
from typing import List, Tuple, Optional, Union
import numpy as np

AtomListType = List[str]
CoordType = np.ndarray
MMChargeTupleType = Tuple[float, float, float, float]


def load_keywords_from_file(path: str) -> List[str]:
    """Load Gaussian route keywords from a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_solvent_descriptor_suffix(entity_has_added_qm_solvent: bool, system_has_mm_solvent: bool) -> str:
    """Return suffix describing solvent components for file names."""
    if entity_has_added_qm_solvent and system_has_mm_solvent:
        return "_qm_mm"
    elif entity_has_added_qm_solvent:
        return "_qm"
    elif system_has_mm_solvent:
        return "_mm"
    else:
        return ""


def write_com_file(
    path: str,
    keywords: List[str],
    title: str,
    charge: Union[int, float],
    spin: int,
    atoms_list: AtomListType,
    coords_array: CoordType,
    mm_charges_list: Optional[List[MMChargeTupleType]] = None,
    fragment_definitions: Optional[List[Tuple[AtomListType, CoordType, int, int]]] = None,
) -> None:
    """Write a Gaussian .com input file."""
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
                    f.write(
                        f" {atom_sym}(Fragment={frag_idx+1}) {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n"
                    )
        else:
            f.write(f"{charge} {spin}\n")
            for atom_sym, xyz_coords in zip(atoms_list, coords_array):
                f.write(f" {atom_sym} {xyz_coords[0]:12.5f} {xyz_coords[1]:12.5f} {xyz_coords[2]:12.5f}\n")

        if mm_charges_list:
            f.write("\n")
            for x_mm, y_mm, z_mm, q_mm in mm_charges_list:
                f.write(f" {x_mm:14.5f} {y_mm:12.5f} {z_mm:12.5f} {q_mm:12.5f}\n")

        f.write("\n")

