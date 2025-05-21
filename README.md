# GIGPy – Gaussian Input Generator in Python

**Version**: 1.0  |  **Date**: 20 May 2025

---

GIGPy (Gaussian **I**nput **G**enerator in **Py**thon) is a single‐entry script that converts molecular
coordinates in XYZ format—either a single structure or a full trajectory—into fully‑formed
Gaussian `.com` input decks.  It automates QM/MM partitioning, solvent
selection, frame management, and optionally writes helper XYZ files and
point‑charge coordinates to streamline vertical excitation‑energy (VEE) and
excitation‑energy‑transfer (EETG) calculations for molecular aggregates.

## Key capabilities

* **Single‐frame or multi‐frame** XYZ ingestion with automatic frame splitting and
  per‑frame output folders.
* **Automated QM/MM Partitioning:**  
  * **Core System Definition:** Clearly defines a "core" system based on the \--aggregate flag and corresponding atom counts in the \--system\_info JSON.  
  * **QM Solvent Localization for Combined System:** Identifies solvent molecules to be included in the Quantum Mechanics (QM) region for the entire combined system (dimer/aggregate) based on a user-defined distance cutoff (--qm\_solvent) from any core atom. This forms the QM region for calculations on the whole system.  
  * **Exclusive QM Solvent Localization for Individual Monomers:** When preparing inputs for individual monomer calculations (e.g., monomer1.com, monomer2.com), the script ensures that QM solvent molecules are uniquely assigned. A solvent molecule is assigned to the QM region of the *first* monomer (in their defined order) that it is localized to. This prevents the same solvent molecule from being part of the specific QM environment of multiple individual monomers in their separate calculations.
* **Generation of Gaussian input files** for

  * Various applications depending on the `--keywords_file` contents.
  * Dimer EETG analyses with fragment labelling (`--eetg`).
* **Flexible point‑charge embedding**

  * Inter‑monomer charges via `--mm_monomer`.
  * Solvent charges: explicit file or auto‑extraction from non‑QM solvent.
* **Rich helper XYZ output** (`*_qm.xyz`, `*_mm_solvent.xyz`) for visual debugging.
* **Robust logging** to `gigpy_run.log` and clear console progress bars.
* **Extensive validation** of input JSON/XML, coordinate consistency, and
  command‑line argument coherence.

## Requirements

* Python ≥ 3.8
* NumPy
* All other imports are from the Python standard library.

## Arguments

Run `python gigpy.py --help` for the exhaustive help text.

---
- `--single_xyz`: (Required)
  - ***Number of inputs:*** 1 file
  - Path to a single-frame XYZ file containing coordinates for entire system (core monomers + solvent)
---
- `--traj_xyz`: (Required)
  - ***Number of inputs:*** 1 file
  - Path to a multi-frame XYZ trajectory; each frame in standard XYZ format
  - This and `--single_xyz` can not be used together
---
- `--frames`: (Required)
  - ***Number of inputs:*** 1 integer
  - Number of frames to process when `--traj_xyz` is used; if omitted all frames are processed
---
- `--aggregate`: (Required)
  - ***Number of inputs:*** 1 integer
  - Number of core monomer units.
  - Filename if aggregate=1: monomer, 2: dimer, >=2: aggregate
---
- `--system_info`: (Required)
  - ***Number of inputs:*** 1 file
  - JSON defining monomer and solvent metadata (format described below)
---
- `--keywords_file`: (Required)
  - ***Number of inputs:*** 1 file
  - Plain-text file of Gaussian route section keywords (one per line)
  - Include PCM/solvation options here as needed
---
- `--eetg`: (Optional)
  - ***Number of inputs:*** 0 (only flag)
  - Generate only EETG `.com` for dimers (requires `--aggregate 2`)
  - Skips dimer VEE `.com`
  - This flag is mutually exclusive with `--output_com`
---
- `--qm_solvent`: (Optional, default 5.0)
  - ***Number of inputs:*** 1 float
  - Radius in Å for selecting explicit QM solvent shell around core atoms 
---
- `--mm_monomer`: (Optional)
  - ***Number of inputs:*** 0 or N files
  - Include MM embedding charges from other monomers
  - `0` = all atoms are assigned zero charges
  -  List charge files with “charge x y z” per line for `N` monomers if `--aggregate N`
  - Omit flag for no MM monomer charges
---
- `--mm_solvent`: (Optional)
  - ***Number of inputs:*** 0 or 1 file
  - Include MM solvent embedding
  - flag alone to auto-detect non-QM solvent and assign charges from `system_info`
  - or provide XYZ-like file path of charges
  - omit flag for no MM solvent
---
- `--output_com`: (Optional, default `both`)
  - ***Number of inputs:*** 0 or 1 string : `monomer`, `dimer`, `both`
  - `monomer` : write only monomer .com files with QM and MM solvent if provided
  - `dimer`   : write only dimer .com files with QM and MM solvent if provided
  - `both`    : write both monomer and dimer .com files 
---
- `--output_xyz`: (Optional, default `both`)
  - ***Number of inputs:*** 0 or 1 string : `monomer`, `dimer`, `both`
  - `monomer` : write monomer + QM solvent .xyz files + their corresponding MM solvent xyz files separately
  - `dimer`   : write dimer + QM solvent .xyz files + their corresponding MM solvent xyz files separately
  - `both`    : write monomer and dimer QM .xyz files and the corresponding MM solvent xyz files separately
  - `none` or omit flag : do not write any XYZ files
---
- `--tag`: (Optional)
  - ***Number of inputs:*** 0 or 1 string
  - Custom tag appended to generated .com filenames
---
- `--logfile`: (Optional)
  - ***Number of inputs:*** 0 or 1 string
  - Name for detailed log file (default `gigpy_run.log`)
---

## Input files
## `keywords.txt`
The file must be a **plain text** file with one entry per line.
Example: [keywords.txt](./examples/keywords.txt)
```text
#p CAM-B3LYP/6-31G*                 ! Functional/basis set
# TDA(Nstates=6)                    ! Excited state calculations
# Density(Transition=1)             ! S0->S1 transition density
# Integral(grid=fine)               ! Grid for two-electron integrals
# SCF(conver=10)                    ! SCF convergence
# NoSymm                            ! No symmetry keyword for dimers
# EmpiricalDispersion=GD3           ! Dispersion interaction
# IOp(9/40=4)                       ! Print eigenvector components threshold
```

## `system_info.json` 

The file must be a **JSON array** with one entry per monomer followed by **one**
entry describing the solvent. 

Example: [cv_dimer_water.json](./examples/cv_dimer_water.json)
```jsonc
[
  {
    "system"      : "monomer1",
    "name"        : "cv",
    "mol_formula" : "C16H12N3O1",
    "nAtoms"      : 32,
    "charge"      : 1,
    "spin_mult"   : 1
  },
  {
    "system"      : "monomer2",
    "name"        : "cv",
    "mol_formula" : "C16H12N3O1",
    "nAtoms"      : 32,
    "charge"      : 1,
    "spin_mult"   : 1
  },
  {
    "system"      : "solvent",
    "name"        : "water",
    "mol_formula" : "H2O",
    "nAtoms"      : 3,
    "charges": 
    [
      { "element": "O", "charge": -0.834 },
      { "element": "H", "charge":  0.417 },
      { "element": "H", "charge":  0.417 }
    ]
  }
]
```

---

## `Charges`

### a) Inter‑monomer charges (`--mm_monomer`)

Plain text with **four** columns (charge x y z) and two header lines (XYZ‑like):

```text
<natoms>
<comment>
-0.123   1.234   0.456   -2.345
…
```

Provide **N** such files when `--aggregate N` so every monomer can be embedded
in the charges of all other monomers.

### b) Explicit MM solvent (`--mm_solvent charges.xyz`)

Same format as above but the first column is **charge**, followed by *x y z*.

---

## Examples

### 1  Single monomer, VEE, helper XYZ

```bash
python gigpy.py --single_xyz M1.xyz \
                --aggregate 1 \
                --system_info system_info.json \
                --keywords_file route.txt \
                --output_com monomer \
                --output_xyz
```

### 2  Dimer trajectory, EETG, auto MM solvent

```bash
python gigpy.py --traj_xyz dimer_traj.xyz --frames 10 \
                --aggregate 2 \
                --system_info system_info.json \
                --keywords_file route.txt \
                --qm_solvent 6.0 \
                --mm_solvent \
                --eetg \
                --output_com dimer --output_xyz both
```

### 3  Trimer, explicit monomer charges & custom solvent charges

```bash
python gigpy.py --single_xyz trimer.xyz \
                --aggregate 3 \
                --system_info system_info.json \
                --keywords_file route.txt \
                --mm_monomer m1.chg m2.chg m3.chg \
                --mm_solvent custom_solvent.xyz
```

---

## Outputs

> monomer `.com` files: `monomer1.com`, `monomer2.com`, etc. : including qm, mm solvent if provided.

> dimer `.com` files: `dimer`, etc. : including qm, mm solvent if provided.

XYZ files when `--output_xyz` is specified :
> monomer XYZ files: `monomer1_qm.xyz` and its corresponding `monomer1_mm_solvent.xyz`

> dimer XYZ files: `dimer_qm.xyz` and its corresponding `dimer_mm_solvent.xyz`

> Temporary files (`_current_frame_data.xyz`) are deleted after use when
processing trajectories.

> `gigpy_run.log` is created in the current working directory.

---

## Logging & error handling

GIGPy writes a concise console progress bar and mirrors all
messages—including stack traces on uncaught exceptions—to `gigpy_run.log`.
Fatal errors return a non‑zero exit status.

---

### Acknowledgements

Developed with ♥ by *Sayan Adhikari* and contributors.

