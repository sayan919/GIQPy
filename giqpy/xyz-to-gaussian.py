#!/usr/bin/env python3
#=================================================================================================
# Sayan Adhikari | June 25, 2025 | https://github.com/sayan919
#=================================================================================================
"""
Convert GIQPy XYZ output into Gaussian .com input files.

Reads the per-frame XYZ files produced by giqpy.py:
    {agg}-qm.xyz, {system}-qm.xyz   : QM geometries
    {agg}-mm.xyz, {system}-mm.xyz   : MM point charges ("charge x y z")
and writes the corresponding Gaussian .com files. ({agg} is the aggregate base name
"{name}-{dimer|trimer|...}" built from the JSON monomer "name"s; {system} is the per-monomer
label from the JSON "system" key.)

Flags:
    --input-dir     (Optional) : Directory holding the XYZ files, or a parent containing numbered
                                 per-frame subdirectories (default: current directory).
    --num-monomers  (Required) : Number of core monomer units (must match the giqpy run).
    --system-info   (Required) : Same JSON used by giqpy (provides charge / spin / names).
    --gauss-keywords(Required) : File with the Gaussian route section keywords.
    --com-files     (Optional) : monomer | dimer | both (default: both).
    --eetg          (Optional) : Generate only the EETG dimer input (requires --num-monomers 2).
    --tag           (Optional) : Custom tag appended to .com filenames (e.g. ...-TAG.com).
    --log-file      (Optional) : Log file name (default: xyz_to_gaussian.log).
"""
import argparse
import os
import sys
import datetime
from typing import List, Optional, Dict, Any, Tuple

import giqpy_common as fn


def find_frame_dirs(indir: str, term: str) -> List[str]:
    """
    Return the directories to process. If ``indir`` contains numbered subdirectories with QM files,
    return those (numerically sorted); otherwise process ``indir`` itself.
    """
    marker = f"{term}-qm.xyz"  # giqpy always writes the aggregate QM file

    def has_xyz(d: str) -> bool:
        return os.path.exists(os.path.join(d, marker))

    numbered = []
    try:
        for entry in os.listdir(indir):
            full = os.path.join(indir, entry)
            if os.path.isdir(full) and entry.isdigit() and has_xyz(full):
                numbered.append((int(entry), full))
    except OSError:
        pass
    if numbered:
        return [full for _, full in sorted(numbered)]
    if has_xyz(indir):
        return [indir]
    return []


def load_mm(path: str) -> Optional[List[fn.MMChargeTupleType]]:
    """Read an MM charge XYZ file (2 header lines) if present and non-empty; else None."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        charges = fn.read_charge_file(path, skip_header=2)
        return charges or None
    return None


def build_title_base(out_dir: str, agg_base: str, term: str, base_system_name: str, n_dyes: int,
                     n_atoms_per_monomer: List[int]) -> str:
    """Reconstruct the aggregate title base, including the m1-m2 centroid distance when available.

    ``agg_base`` is the aggregate file base (e.g. 'cv-dimer') used to locate the QM file;
    ``term`` is the multiplicity word (e.g. 'dimer') shown in the title text.
    """
    distance_str = ""
    if n_dyes >= 2:
        agg_qm = os.path.join(out_dir, f"{agg_base}-qm.xyz")
        if os.path.exists(agg_qm):
            _, coords, _ = fn.read_xyz(agg_qm)
            total_core = sum(n_atoms_per_monomer)
            dist = fn.calculate_centroid_distance_between_first_two(coords[:total_core], n_atoms_per_monomer)
            if dist is not None:
                distance_str = f"(m1-m2 centroid dist: {dist:.2f} A)"
    return f"{base_system_name} {term} {distance_str}".strip()


def generate_for_dir(
    src_dir: str,
    dest_dir: str,
    term: str,
    agg_base: str,
    monomers_meta: List[Dict[str, Any]],
    solvent_name: str,
    n_dyes: int,
    keywords: List[str],
    gen_monomer: bool,
    gen_aggregate: bool,
    eetg: bool,
    tag_suffix: str,
) -> None:
    """Generate the requested .com files for a single frame directory.

    XYZ inputs are read from ``src_dir``; .com outputs are written to ``dest_dir``.
    """
    n_atoms_per_monomer = fn.monomer_atom_counts(monomers_meta)
    monomer_names = [m.get(fn.JSON_KEY_NAME, f'Monomer {i + 1}') for i, m in enumerate(monomers_meta)]
    monomer_labels = fn.system_labels(monomers_meta)  # output names from the JSON 'system' key
    base_system_name = monomer_names[0] if len(set(monomer_names)) == 1 else "_".join(monomer_names)

    total_charge = sum(m[fn.JSON_KEY_CHARGE] for m in monomers_meta)
    total_spin = 1
    for m in monomers_meta:
        total_spin *= m[fn.JSON_KEY_SPIN_MULT]
    total_spin = total_spin if total_spin > 0 else 1

    # MM solvent presence is signalled by the aggregate MM file (solvent-only by construction).
    agg_mm = load_mm(os.path.join(src_dir, f"{agg_base}-mm.xyz"))
    system_has_mm_solvent = agg_mm is not None

    title_base = build_title_base(src_dir, agg_base, term, base_system_name, n_dyes, n_atoms_per_monomer)

    # --- EETG dimer input ---
    if eetg:
        if n_dyes != 2:
            fn.write_to_log("EETG requires --num-monomers 2; skipping EETG.", is_warning=True)
        elif gen_aggregate:
            m1_path = os.path.join(src_dir, f"{monomer_labels[0]}-qm.xyz")
            m2_path = os.path.join(src_dir, f"{monomer_labels[1]}-qm.xyz")
            if not (os.path.exists(m1_path) and os.path.exists(m2_path)):
                fn.write_to_log("EETG: monomer QM files missing; skipping EETG.", is_warning=True)
            else:
                m1_atoms, m1_coords, _ = fn.read_xyz(m1_path)
                m2_atoms, m2_coords, _ = fn.read_xyz(m2_path)
                frags_have_qm = (len(m1_atoms) > n_atoms_per_monomer[0]) or (len(m2_atoms) > n_atoms_per_monomer[1])
                suffix = fn.get_solvent_descriptor_suffix(frags_have_qm, system_has_mm_solvent)
                title = f"{title_base}{fn.get_solvent_title_fragment(frags_have_qm, system_has_mm_solvent, solvent_name)} EET Analysis".strip()
                frag_defs = [
                    (m1_atoms, m1_coords, monomers_meta[0][fn.JSON_KEY_CHARGE], monomers_meta[0][fn.JSON_KEY_SPIN_MULT]),
                    (m2_atoms, m2_coords, monomers_meta[1][fn.JSON_KEY_CHARGE], monomers_meta[1][fn.JSON_KEY_SPIN_MULT]),
                ]
                fn.write_com_file(
                    os.path.join(dest_dir, f"{agg_base}-eetg{suffix}{tag_suffix}.com"),
                    keywords, title, total_charge, total_spin,
                    [], [], mm_charges_list=agg_mm, fragment_definitions=frag_defs,
                )
                fn.write_to_log(f"Wrote EETG file in {dest_dir}.")
        return  # EETG mode generates only the EETG file

    # --- Monomer inputs ---
    if gen_monomer:
        for i in range(n_dyes):
            label = monomer_labels[i]
            qm_path = os.path.join(src_dir, f"{label}-qm.xyz")
            if not os.path.exists(qm_path):
                fn.write_to_log(f"{label}-qm.xyz not found in {src_dir}; skipping.", is_warning=True)
                continue
            atoms, coords, _ = fn.read_xyz(qm_path)
            has_qm_sol = len(atoms) > n_atoms_per_monomer[i]
            mm_charges = load_mm(os.path.join(src_dir, f"{label}-mm.xyz"))
            suffix = fn.get_solvent_descriptor_suffix(has_qm_sol, system_has_mm_solvent)
            title = f"{monomer_names[i]} {label}{fn.get_solvent_title_fragment(has_qm_sol, system_has_mm_solvent, solvent_name)}".strip()
            fn.write_com_file(
                os.path.join(dest_dir, f"{label}{suffix}{tag_suffix}.com"),
                keywords, title,
                monomers_meta[i][fn.JSON_KEY_CHARGE], monomers_meta[i][fn.JSON_KEY_SPIN_MULT],
                atoms, coords, mm_charges_list=mm_charges,
            )
            fn.write_to_log(f"Wrote {label}.com in {dest_dir}.")

    # --- Aggregate input ---
    if gen_aggregate:
        qm_path = os.path.join(src_dir, f"{agg_base}-qm.xyz")
        if not os.path.exists(qm_path):
            fn.write_to_log(f"{agg_base}-qm.xyz not found in {src_dir}; skipping aggregate.", is_warning=True)
        else:
            atoms, coords, _ = fn.read_xyz(qm_path)
            has_qm_sol = len(atoms) > sum(n_atoms_per_monomer)
            suffix = fn.get_solvent_descriptor_suffix(has_qm_sol, system_has_mm_solvent)
            title = f"{title_base}{fn.get_solvent_title_fragment(has_qm_sol, system_has_mm_solvent, solvent_name)}".strip()
            fn.write_com_file(
                os.path.join(dest_dir, f"{agg_base}{suffix}{tag_suffix}.com"),
                keywords, title, total_charge, total_spin,
                atoms, coords, mm_charges_list=agg_mm,
            )
            fn.write_to_log(f"Wrote {agg_base}.com in {dest_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert GIQPy XYZ output into Gaussian .com input files.",
    )
    parser.add_argument('--input-dir', type=str, default=os.getcwd(),
                        help='Directory with the XYZ files, or a parent of numbered frame subdirectories (default: cwd).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to write the .com files into, mirroring frame subdirs (default: same as --input-dir).')
    parser.add_argument('--num-monomers', type=int, required=True, help='Number of core monomer units.')
    parser.add_argument('--system-info', type=str, required=True,
                        help='JSON file with monomer and solvent metadata (provides charge/spin/names).')
    parser.add_argument('--gauss-keywords', type=str, required=True,
                        help='File with the Gaussian route section keywords.')
    parser.add_argument('--com-files', choices=['monomer', 'dimer', 'both'], default='both',
                        help='Which .com files to generate (default: both).')
    parser.add_argument('--eetg', action='store_true',
                        help='Generate only the EETG dimer input (requires --num-monomers 2).')
    parser.add_argument('--tag', type=str, default="", help='Optional custom tag for .com filenames.')
    parser.add_argument('--log-file', type=str, default="xyz-to-gaussian.log",
                        help='Log file name (default: xyz-to-gaussian.log).')
    args = parser.parse_args()

    try:
        fn.log_file_handle = open(args.log_file, 'w')
        fn.log_file_handle.write(f"xyz_to_gaussian Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fn.log_file_handle.write(f"Arguments: {vars(args)}\n\n")
        fn.log_file_handle.flush()
    except IOError as e:
        print(f"CRITICAL ERROR: Could not open log file {args.log_file}: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.num_monomers < 1:
        parser.error("--num-monomers must be at least 1.")
    if args.eetg and args.num_monomers != 2:
        parser.error("--eetg requires --num-monomers 2.")
    if not os.path.exists(args.gauss_keywords) or os.path.getsize(args.gauss_keywords) == 0:
        err = f"Gaussian keywords file '{args.gauss_keywords}' not found or is empty."
        print(f"ERROR: {err}", file=sys.stderr)
        fn.write_to_log(err, is_error=True)
        sys.exit(1)

    keywords = fn.load_keywords_from_file(args.gauss_keywords)
    try:
        monomers_meta, solvent_meta = fn.load_system_info(args.system_info, args.num_monomers)
    except Exception as e:
        print(f"CRITICAL ERROR loading system_info: {e}", file=sys.stderr)
        fn.write_to_log(f"CRITICAL ERROR loading system_info: {e}", is_error=True)
        sys.exit(1)
    solvent_name = solvent_meta.get(fn.JSON_KEY_NAME, "solvent")

    term = fn.multiplicity_word(args.num_monomers)
    agg_base = fn.aggregate_basename(monomers_meta, args.num_monomers)  # e.g. 'cv-dimer'
    gen_monomer = args.com_files in ('monomer', 'both')
    gen_aggregate = args.com_files in ('dimer', 'both')
    tag_suffix = f"-{args.tag}" if args.tag else ""

    frame_dirs = find_frame_dirs(args.input_dir, agg_base)
    if not frame_dirs:
        err = f"No GIQPy XYZ files found under '{args.input_dir}'. Run giqpy.py first."
        print(f"ERROR: {err}", file=sys.stderr)
        fn.write_to_log(err, is_error=True)
        sys.exit(1)

    output_base = args.output_dir if args.output_dir is not None else args.input_dir

    total = len(frame_dirs)
    for idx, src_dir in enumerate(frame_dirs):
        # Mirror the per-frame subdirectory structure in the output directory.
        if os.path.abspath(src_dir) == os.path.abspath(args.input_dir):
            dest_dir = output_base
        else:
            dest_dir = os.path.join(output_base, os.path.basename(os.path.normpath(src_dir)))
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as e:
            err = f"Could not create output directory {dest_dir}: {e}"
            print(f"\nERROR: {err}", file=sys.stderr)
            fn.write_to_log(err, is_error=True)
            continue
        print(f"[{idx + 1}/{total}] Writing .com files in {dest_dir}")
        fn.write_to_log(f"\n\n--- Frame dir {idx + 1}/{total}: {src_dir} -> {dest_dir} ---")
        try:
            generate_for_dir(src_dir, dest_dir, term, agg_base, monomers_meta, solvent_name, args.num_monomers,
                             keywords, gen_monomer, gen_aggregate, args.eetg, tag_suffix)
        except Exception as e:
            err = f"Error generating .com files in {dest_dir}: {e}"
            print(f"\nERROR: {err}", file=sys.stderr)
            fn.write_to_log(err, is_error=True)
            continue

    print("All .com generation complete.")
    fn.write_to_log("All .com generation complete.")
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
