#!/usr/bin/env python3
#=================================================================================================
# Sayan Adhikari | June 25, 2025 | https://github.com/sayan919
#=================================================================================================
"""
Generate QM-region and MM point-charge XYZ files for QM/MM systems.

This script handles geometry only. To turn the resulting XYZ files into Gaussian .com inputs,
run xyz_to_gaussian.py afterwards (see run_giqpy.sh).

Per frame it writes (into the frame's output directory):
    {term}_qm.xyz        : aggregate QM region   (core + all QM solvent)
    monomer{i}_qm.xyz    : monomer QM region     (core_i + its UNIQUE QM solvent)
    {term}_mm.xyz        : aggregate MM charges  (MM solvent only)            [if MM requested]
    monomer{i}_mm.xyz    : monomer MM charges    (other-monomer embedding + MM solvent) [if requested]
where {term} is "dimer" for --nDyes 2 and "aggregate" otherwise.

Flags:
    --traj          (Required) : Multi-frame trajectory XYZ file (use --nFrames 1 for a single frame).
    --nFrames       (Optional) : Number of frames to process (default: all).
    --nDyes         (Required) : Number of core monomer units.
    --system_info   (Required) : Single JSON defining all monomers and the solvent.
    --qmSol_radius  (Optional) : QM solvent shell radius (Å) around core atoms (default 5.0; negative disables).
    --mm_monomer    (Optional) : MM embedding from other monomers. '0' = zero charges at their positions;
                                  or one charge file ("charge x y z") per monomer.
    --mm_solvent    (Optional) : MM solvent. Flag alone = non-QM solvent charged from system_info;
                                  or a path to an XYZ-like file ("charge x y z", 2 header lines).
    --logfile       (Optional) : Log file name (default: giqpy_run.log).
"""
import argparse
import os
import sys
import datetime
from typing import List, Optional, Dict, Any, Tuple

import functions as fn


def resolve_mm_solvent(
    mm_solvent_arg: Optional[str],
    non_qm_sol_groups: List[fn.SolventGroupType],
    solvent_metadata: Dict[str, Any],
) -> Tuple[Optional[List[fn.MMChargeTupleType]], bool]:
    """Return (mm_solvent_point_charges as (x,y,z,q) or None, system_has_mm_solvent)."""
    if mm_solvent_arg is None:
        return None, False

    if mm_solvent_arg == fn.AUTO_MM_SOLVENT_TRIGGER:
        if not non_qm_sol_groups:
            fn.write_to_log("No non-QM solvent molecules found to treat as MM solvent.")
            return None, False
        try:
            charges = fn.assign_charges_to_solvent_molecules(non_qm_sol_groups, solvent_metadata)
        except ValueError as e:
            fn.write_to_log(f"Error assigning charges to auto-detected MM solvent: {e}", is_error=True)
            return None, False
        if charges:
            fn.write_to_log(f"Generated {len(charges)} MM solvent point charges from auto-detection.")
            return charges, True
        fn.write_to_log("Auto-detection found non-QM solvent groups, but no charges were assigned.")
        return None, False

    # A path to an XYZ-like MM solvent file was provided.
    charges = fn.read_charge_file(mm_solvent_arg, skip_header=2)
    if charges:
        fn.write_to_log(f"Loaded {len(charges)} MM solvent point charges from file: {mm_solvent_arg}.")
        return charges, True
    warn = f"No valid MM solvent charges loaded from file {mm_solvent_arg}."
    print(f"\nWARNING: {warn}")
    fn.write_to_log(warn, is_warning=True)
    return None, False


def resolve_mm_monomer(
    mm_monomer_arg: Optional[List[str]],
    n_dyes: int,
    core_coords: fn.CoordType,
    n_atoms_per_monomer: List[int],
) -> List[List[fn.MMChargeTupleType]]:
    """
    Build the inter-monomer MM embedding charges for each monomer (charges from every OTHER monomer).
    Returns a list (length n_dyes) of (x,y,z,q) lists.
    """
    embedding: List[List[fn.MMChargeTupleType]] = [[] for _ in range(n_dyes)]
    if mm_monomer_arg is None:
        return embedding

    if n_dyes == 1:
        fn.write_to_log("--mm_monomer specified with --nDyes 1; no other monomers to embed.", is_warning=True)
        return embedding

    # Per-monomer core coordinate slices.
    slices: List[fn.CoordType] = []
    idx = 0
    for n in n_atoms_per_monomer:
        slices.append(core_coords[idx:idx + n])
        idx += n

    if mm_monomer_arg == ['0']:
        fn.write_to_log("Applying zero charges for other-monomer embedding (--mm_monomer 0).")
        for i in range(n_dyes):
            for j in range(n_dyes):
                if i == j:
                    continue
                for row in slices[j]:
                    embedding[i].append((row[0], row[1], row[2], 0.0))
        return embedding

    if len(mm_monomer_arg) != n_dyes:
        warn = (f"--mm_monomer was given {len(mm_monomer_arg)} charge file(s), but --nDyes is {n_dyes}. "
                f"Expected {n_dyes} file(s). MM monomer embedding charges will be skipped.")
        print(f"\nWARNING: {warn}")
        fn.write_to_log(warn, is_warning=True)
        return embedding

    fn.write_to_log(f"Loading inter-monomer embedding charges from {n_dyes} file(s).")
    per_file = [fn.load_monomer_charges_from_file(f) for f in mm_monomer_arg]
    for i in range(n_dyes):
        for j in range(n_dyes):
            if i == j:
                continue
            embedding[i].extend(per_file[j])
    return embedding


def write_mm_xyz(path: str, charges_xyzq: List[fn.MMChargeTupleType], comment: str) -> None:
    """Write a list of (x,y,z,q) point charges as an MM XYZ file (charge in the first column)."""
    charges_col = [q for _, _, _, q in charges_xyzq]
    coords = [(x, y, z) for x, y, z, _ in charges_xyzq]
    fn.write_xyz(path, charges_col, coords, comment=comment)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate QM-region and MM point-charge XYZ files from a trajectory XYZ.",
    )
    parser.add_argument('--traj', type=str, required=True,
                        help='Multi-frame trajectory XYZ file. Use --nFrames 1 for a single XYZ input.')
    parser.add_argument('--nFrames', type=int, default=None,
                        help='Number of frames to process (default: all).')
    parser.add_argument('--nDyes', type=int, required=True,
                        help='Number of core monomer units.')
    parser.add_argument('--system_info', type=str, required=True,
                        help='JSON file with monomer and solvent metadata.')
    parser.add_argument('--qmSol_radius', type=float, default=5.0,
                        help='Radius (Å) for QM solvent selection (default: 5.0). Negative disables QM solvent.')
    parser.add_argument('--mm_monomer', type=str, nargs='*',
                        help="MM embedding for other monomers: '0' for zero charges, or one charge file per monomer.")
    parser.add_argument('--mm_solvent', type=str, nargs='?', const=fn.AUTO_MM_SOLVENT_TRIGGER, default=None,
                        help='MM solvent: XYZ-like charge file, or flag alone to auto-detect from non-QM solvent.')
    parser.add_argument('--logfile', type=str, default="giqpy_run.log",
                        help='Log file name (default: giqpy_run.log).')
    args = parser.parse_args()

    # --- Setup log ---
    try:
        fn.log_file_handle = open(args.logfile, 'w')
        fn.log_file_handle.write(f"GIQPy Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fn.log_file_handle.write(f"Arguments: {vars(args)}\n\n")
        fn.log_file_handle.flush()
    except IOError as e:
        print(f"CRITICAL ERROR: Could not open log file {args.logfile}: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.nDyes < 1:
        err = "--nDyes must be at least 1."
        print(f"ERROR: {err}", file=sys.stderr)
        fn.write_to_log(err, is_error=True)
        parser.error(err)

    base_output_dir = os.getcwd()
    combined_system_term = "dimer" if args.nDyes == 2 else "aggregate"

    # --- Load metadata once (was previously re-parsed every frame) ---
    try:
        monomers_meta, solvent_meta = fn.load_system_info(args.system_info, args.nDyes)
    except Exception as e:
        print(f"CRITICAL ERROR loading system_info: {e}", file=sys.stderr)
        fn.write_to_log(f"CRITICAL ERROR loading system_info: {e}", is_error=True)
        sys.exit(1)
    solvent_name = solvent_meta.get(fn.JSON_KEY_NAME, "solvent")

    # --- Split frames ---
    frames = fn.split_frames(args.traj, args.nFrames, base_output_dir)
    total = len(frames)
    if total == 0:
        err = f"No frames could be processed from trajectory: {args.traj}"
        print(f"ERROR: {err}", file=sys.stderr)
        fn.write_to_log(err, is_error=True)
        sys.exit(1)

    monomer_names = [m.get(fn.JSON_KEY_NAME, f'm{i + 1}') for i, m in enumerate(monomers_meta)]
    if len(set(monomer_names)) == 1:
        base_system_name = monomer_names[0]
    else:
        base_system_name = "_".join(monomer_names)

    for frame_idx, (frame_id, temp_xyz_path, out_dir) in enumerate(frames):
        print(f"[{frame_idx + 1}/{total}] Processing frame {frame_id} -> {out_dir}")
        fn.write_to_log(f"\n\n--- Processing Frame {frame_idx + 1} / {total} (ID: {frame_id}) -> '{out_dir}' ---")

        try:
            (monomer_qm_regions, aggregate_qm_region, non_qm_sol_groups,
             qm_solvent_flags, core_coords, n_atoms_per_monomer) = \
                fn.localize_solvent_and_prepare_regions(temp_xyz_path, monomers_meta, solvent_meta, args.qmSol_radius)
        except Exception as e:
            err = f"Error during region preparation for frame {frame_id}: {e}"
            print(f"\nERROR: {err}", file=sys.stderr)
            fn.write_to_log(err, is_error=True)
            fn.write_to_log(f"Skipping frame {frame_id}.", is_warning=True)
            continue

        # Title/comment base with optional m1-m2 centroid distance.
        distance_str = ""
        if args.nDyes >= 2:
            dist = fn.calculate_centroid_distance_between_first_two(core_coords, n_atoms_per_monomer)
            if dist is not None:
                distance_str = f"(m1-m2 centroid dist: {dist:.2f} A)"
        comment_base = f"{base_system_name} {combined_system_term} {distance_str}".strip()

        # --- Resolve MM charges (solvent + inter-monomer embedding) ---
        mm_solvent_charges, system_has_mm_solvent = resolve_mm_solvent(args.mm_solvent, non_qm_sol_groups, solvent_meta)
        mm_embedding = resolve_mm_monomer(args.mm_monomer, args.nDyes, core_coords, n_atoms_per_monomer)

        # --- QM region XYZ files ---
        agg_qm_atoms, agg_qm_coords = aggregate_qm_region
        agg_qm_sol = " + qm " + solvent_name if qm_solvent_flags.get('aggregate_has_added_qm_solvent') else ""
        fn.write_xyz(os.path.join(out_dir, f'{combined_system_term}_qm.xyz'),
                     agg_qm_atoms, agg_qm_coords, comment=f"{comment_base} qm{agg_qm_sol}")

        for i, (mono_atoms, mono_coords) in enumerate(monomer_qm_regions):
            has_qm_sol = qm_solvent_flags.get(f'monomer_{i}_has_added_qm_solvent', False)
            qm_sol_desc = f" + its unique qm {solvent_name}" if has_qm_sol else ""
            fn.write_xyz(os.path.join(out_dir, f'monomer{i + 1}_qm.xyz'),
                         mono_atoms, mono_coords,
                         comment=f"{monomer_names[i]} monomer{i + 1} qm{qm_sol_desc}")

        # --- MM charge XYZ files ---
        # Aggregate: MM solvent only.
        if mm_solvent_charges:
            write_mm_xyz(os.path.join(out_dir, f'{combined_system_term}_mm.xyz'),
                         mm_solvent_charges,
                         comment=f"mm {solvent_name} for {base_system_name} {combined_system_term}")
            fn.write_to_log(f"Wrote aggregate MM charges ({len(mm_solvent_charges)} solvent point charges).")

        # Monomers: other-monomer embedding + MM solvent.
        for i in range(args.nDyes):
            combined: List[fn.MMChargeTupleType] = []
            combined.extend(mm_embedding[i])
            if mm_solvent_charges:
                combined.extend(mm_solvent_charges)
            if combined:
                has_embedding = bool(mm_embedding[i])
                if has_embedding:
                    comment = f"mm monomer + mm {solvent_name} for {monomer_names[i]} monomer{i + 1}"
                else:
                    comment = f"mm {solvent_name} for {monomer_names[i]} monomer{i + 1}"
                write_mm_xyz(os.path.join(out_dir, f'monomer{i + 1}_mm.xyz'), combined, comment=comment)
                fn.write_to_log(f"Wrote monomer {i + 1} MM charges ({len(combined)} point charges).")
            elif args.mm_monomer or args.mm_solvent:
                warn = f"No MM charges to write for monomer{i + 1}; 'monomer{i + 1}_mm.xyz' skipped."
                fn.write_to_log(warn, is_warning=True)

        # Cleanup temporary per-frame XYZ.
        if os.path.exists(temp_xyz_path) and fn.TEMP_FRAME_XYZ_FILENAME in temp_xyz_path:
            try:
                os.remove(temp_xyz_path)
            except OSError as e:
                fn.write_to_log(f"Could not remove temporary file {temp_xyz_path}: {e}", is_warning=True)

    print("All XYZ generation complete.")
    fn.write_to_log("All XYZ generation complete.")
    if fn.log_file_handle:
        fn.log_file_handle.close()


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        msg = f"CRITICAL ERROR: {e}"
        print(f"ERROR: {msg}", file=sys.stderr)
        if fn.log_file_handle and not fn.log_file_handle.closed:
            fn.write_to_log(msg, is_error=True)
            import traceback
            fn.write_to_log(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        msg = f"AN UNHANDLED CRITICAL ERROR OCCURRED: {e}"
        print(f"ERROR: {msg}", file=sys.stderr)
        if fn.log_file_handle and not fn.log_file_handle.closed:
            fn.write_to_log(msg, is_error=True)
            import traceback
            fn.write_to_log(traceback.format_exc())
        sys.exit(1)
    finally:
        if fn.log_file_handle and not fn.log_file_handle.closed:
            fn.log_file_handle.close()
