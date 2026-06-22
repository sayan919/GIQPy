#!/bin/bash
#=================================================================================================
# Sayan Adhikari | June 25, 2025 | https://github.com/sayan919
#=================================================================================================
# Usage: bash run-giqpy.sh <input_traj.xyz>
#
# Two stages:
#   1) giqpy.py            : trajectory XYZ  ->  QM-region + MM point-charge XYZ files (per frame)
#   2) xyz-to-gaussian.py  : those XYZ files ->  Gaussian .com inputs
#
# giqpy.py flags:
#   --traj          : Multi-frame trajectory XYZ (use --num-frames 1 for a single frame).
#   --num-frames    : Number of frames to process (default: all).
#   --num-monomers  : Number of core monomer units.
#   --system-info   : Single JSON defining all monomers and the solvent.
#   --qm-radius     : QM solvent shell radius (Å) around core atoms (default 5.0; negative disables).
#   --mm-monomer    : '0' for zero charges at other monomers, or one charge file per monomer.
#   --mm-solvent    : flag alone = non-QM solvent from system_info; or path to an XYZ-like charge file.
#   --log-file      : Log file name (default: giqpy-run.log).
#
# xyz-to-gaussian.py flags:
#   --input-dir      : Directory with the XYZ files / parent of numbered frame dirs (default: cwd).
#   --num-monomers   : Number of core monomer units (must match the giqpy run).
#   --system-info    : Same JSON used above (provides charge / spin / names).
#   --gauss-keywords : File with the Gaussian route section keywords.
#   --com-files      : monomer | dimer | both (default: both).
#   --eetg           : Generate only the EETG dimer input (requires --num-monomers 2).
#   --tag            : Optional custom tag for .com filenames.
#   --log-file       : Log file name (default: xyz-to-gaussian.log).
#=================================================================================================

giqpy='giqpy/giqpy.py'
xyz2gauss='giqpy/xyz-to-gaussian.py'
system_json='path_to_system_json'
keywords='path_to_keywords'
traj=$1

outdir='giqpy-outputs'

# Stage 1: generate QM / MM XYZ files into giqpy-outputs/coordinates
python3 "$giqpy" \
    --traj "$traj" \
    --num-frames 20 \
    --num-monomers 2 \
    --qm-radius 5 \
    --system-info "$system_json" \
    --mm-solvent \
    --output-dir "$outdir/coordinates"

# Stage 2: build Gaussian .com inputs into giqpy-outputs/gaussian-inputs (reads from giqpy-outputs/coordinates)
python3 "$xyz2gauss" \
    --input-dir "$outdir/coordinates" \
    --output-dir "$outdir/gaussian-inputs" \
    --num-monomers 2 \
    --system-info "$system_json" \
    --gauss-keywords "$keywords" \
    --com-files dimer
