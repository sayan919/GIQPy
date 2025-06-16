from typing import List, Tuple, Optional, Union
import os
import numpy as np
import sys

from constants import TEMP_FRAME_XYZ_FILENAME
from logconfig import write_to_log
from datatypes import AtomListType, ChargeListType, CoordType


def split_frames(
    traj_xyz_path: str,
    num_frames_to_extract: Optional[int],
    base_output_dir_for_traj: str
) -> List[Tuple[str, str, str]]:
    """Split multi-frame XYZ into individual frame files."""
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
        write_to_log(
            f"First line of trajectory {traj_xyz_path} must be the number of atoms.",
            is_error=True,
        )
        raise

    block_size = total_atoms_per_frame + 2
    if block_size <= 2:
        write_to_log(
            f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}.",
            is_error=True,
        )
        raise ValueError(
            f"Invalid atom count ({total_atoms_per_frame}) in trajectory {traj_xyz_path}."
        )

    max_possible_frames = len(lines) // block_size
    actual_frames_to_process = (
        num_frames_to_extract
        if num_frames_to_extract and num_frames_to_extract <= max_possible_frames
        else max_possible_frames
    )
    if num_frames_to_extract and num_frames_to_extract > max_possible_frames:
        warn_msg = (
            f"Requested {num_frames_to_extract} frames, but trajectory only contains {max_possible_frames}. Processing all available frames."
        )
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
    """Write an XYZ file."""
    coords_array = np.array(coords)
    if coords_array.ndim == 1 and len(atoms) > 1:
        coords_array = coords_array.reshape(len(atoms), 3)
    with open(path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom_sym, xyz_row in zip(atoms, coords_array):
            if isinstance(atom_sym, (float, np.floating)):
                atom_col = f"{atom_sym:<12.5f}"
            else:
                atom_col = f"{str(atom_sym):<3}"
            f.write(f"{atom_col} {xyz_row[0]:12.5f} {xyz_row[1]:12.5f} {xyz_row[2]:12.5f}\n")
