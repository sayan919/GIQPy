#!/bin/bash
#=================================================================================================
# Sayan Adhikari | June 25, 2025 | https://github.com/sayan919
#=================================================================================================
# Usage: bash run_giqpy.sh <input_traj.xyz>
#
# Two stages:
#   1) giqpy.py            : trajectory XYZ  ->  QM-region + MM point-charge XYZ files (per frame)
#   2) xyz_to_gaussian.py  : those XYZ files ->  Gaussian .com inputs
#
# giqpy.py flags:
#   --traj          : Multi-frame trajectory XYZ (use --nFrames 1 for a single frame).
#   --nFrames       : Number of frames to process (default: all).
#   --nDyes         : Number of core monomer units.
#   --system_info   : Single JSON defining all monomers and the solvent.
#   --qmSol_radius  : QM solvent shell radius (Å) around core atoms (default 5.0; negative disables).
#   --mm_monomer    : '0' for zero charges at other monomers, or one charge file per monomer.
#   --mm_solvent    : flag alone = non-QM solvent from system_info; or path to an XYZ-like charge file.
#   --logfile       : Log file name (default: giqpy_run.log).
#
# xyz_to_gaussian.py flags:
#   --indir         : Directory with the XYZ files / parent of numbered frame dirs (default: cwd).
#   --nDyes         : Number of core monomer units (must match the giqpy run).
#   --system_info   : Same JSON used above (provides charge / spin / names).
#   --gauss_keywords: File with the Gaussian route section keywords.
#   --gauss_files   : monomer | dimer | both (default: both).
#   --eetg          : Generate only the EETG dimer input (requires --nDyes 2).
#   --tag           : Optional custom tag for .com filenames.
#   --logfile       : Log file name (default: xyz_to_gaussian.log).
#=================================================================================================

giqpy='path_to/giqpy.py'
xyz2gauss='path_to/xyz_to_gaussian.py'
system_json='path_to_system_json'
keywords='path_to_keywords'
traj=$1

# Stage 1: generate QM / MM XYZ files
python3 "$giqpy" \
    --traj "$traj" \
    --nFrames 20 \
    --nDyes 2 \
    --qmSol_radius 5 \
    --system_info "$system_json" \
    --mm_solvent

# Stage 2: build Gaussian .com inputs from the XYZ files
python3 "$xyz2gauss" \
    --indir . \
    --nDyes 2 \
    --system_info "$system_json" \
    --gauss_keywords "$keywords" \
    --gauss_files dimer
