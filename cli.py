import argparse
from constants import AUTO_MM_SOLVENT_TRIGGER, LOG_FILENAME


def create_parser() -> argparse.ArgumentParser:
    """Build and return the command line parser."""
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
