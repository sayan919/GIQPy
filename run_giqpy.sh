#!/bin/bash
#=================================================================================================
# Sayan Adhikari | June 25, 2025 | https://github.com/sayan919
#=================================================================================================
# Usage: bash extract_snapshots.sh <input_traj>
# 
# Flags for gen_qm_mm_files.py:
# --single_xyz / --traj_xyz  : Input coordinate file(s).
# --frames                   : Number of frames (default: all; must be used with --traj_xyz).
# --system_info              : Single JSON defining all monomers and solvent properties.
# --aggregate                : Number of core monomer units.
# --qm_aggregate_xyz         : Optional: Use coordinates from this file for the core aggregate,
#                                 overriding those from the main input XYZ. Atom order must match.
# --qm_solvent               : Defines QM solvent shell by radius (Å) around core atoms.
# --mm_solvent               : Optional. Defines MM solvent.
#                             - path provided: XYZ-like file (charge x y z per line, skips 2 headers).
#                             - flag used alone: non-QM solvent with charges from system_info.json.
# --mm_monomer               : Optional. Defines MM charges for embedding from other monomers.
#                             - '0': Embeds other monomers with zero charges at their atomic positions.
#                             - File(s) provided: File(s) with "charge x y z" for MM charges.
# --eetg                     : Flag to generate only EETG input for dimers (requires --aggregate=2).
# --output_com               : monomer/dimer/both; (default: both).
# --gauss_keywords           : Gaussian keywords (must be used with --output_com).
# --output_xyz               : monomer/dimer/both/none; (default: both).
# --tag                      : Optional custom tag for generated .com filenames (e.g., ..._TAG.com).
# --logfile                  : Specify log file name (default: run.log).
#=================================================================================================

main_script='path_to_script'
system_json='path_to_system_json'
traj=$1
keywords='path_to_keywords'

python3 $main_script \
    --traj_xyz $traj \
    --frames 20 \
    --aggregate 2 \
    --qm_solvent 5 \
    --system_info $system_json \
    --output_com dimer \
    --gauss_keywords $keywords \