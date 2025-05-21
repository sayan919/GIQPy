# Generate QM/MM System Input Files for Gaussian or TeraChem

**Version**: 1.0  |  **Date**: 20 May 2025

---
This is single entry script to generate :
- `Gaussian`: **.com** files with given set of keywords and proper formatting for a given system configuration.
- `TeraChem`: **.xyz** files for given system configuration, which can be used paired with **.in** keywords file.

## System configuiration explanation

- **Input:**
  - **Single frame**: A single XYZ file containing the coordinates of the entire system (core monomers + solvent).
  - **Multi-frame**: A multi-frame XYZ trajectory, where each frame is in standard XYZ format.
- **Code capabilities:**
  - **Simple usage** : generate a .com file for a single frame system (.xyz) with a given set of keywords.
  - **Configuring solvent region**: (given a single-frame or multi-frame XYZ file with solute in a solvent)
    - ***`QM solvent`***: user can pick out a localized QM solvent region by specifying a distance cutoff around each atom of the solute. In case of multiple solute molecules, the code will check for common solvent QM atoms between a given pair of localized QM region, and will assign them to one. This explicitely gives a QM region unique to each solute molecule.
    - ***`MM solvent`***: user can provide a file with the coordinates of the solvent region, or the code can auto-detect the non-QM solvent and assign charges from the `system_info` JSON.
    - The user can make combinations: no solvent, QM solvent only, MM solvent only, or both QM and MM solvent.
    - This will be executed for each frame in case of a multi-frame trajectory XYZ file.
  - **Configuring aggregates**:
    - We define, no_of_aggregates: 1 = monomer, 2 = dimer, >=2 = aggregate.
    - **`aggregate files`**: all the solute molecules will be present in the .com file and the .xyz file.
    - **`isolated monomer files`**: user has the following options
      - other monomer can be skipped
      - other monomer can be included with zero charges in the MM region
      - other monomer can be included with its specific MM charges in the MM region
- **Output:**
  - **.com files**:
    - generated for each monomer and dimer (if applicable) with the specified keywords.
    - genearated for **EET Analysis** calculations with proper formatting. (only for dimers)
  - **.xyz files**: a separate .xyz file for QM region and its corresponding MM solvent region.



## Arguments
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
- `--gauss_keywords`: (optional; must be provided if `--output_com` is used)
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
  - Name for detailed log file (default `run.log`)
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
python gen_qm_mm_files.py --single_xyz M1.xyz \
                --aggregate 1 \
                --system_info system_info.json \
                --gauss_keywords route.txt \
                --output_com monomer \
                --output_xyz
```

### 2  Dimer trajectory, EETG, auto MM solvent

```bash
python gen_qm_mm_files.py --traj_xyz dimer_traj.xyz --frames 10 \
                --aggregate 2 \
                --system_info system_info.json \
                --gauss_keywords route.txt \
                --qm_solvent 6.0 \
                --mm_solvent \
                --eetg \
                --output_com dimer --output_xyz both
```

### 3  Trimer, explicit monomer charges & custom solvent charges

```bash
python gen_qm_mm_files.py --single_xyz trimer.xyz \
                --aggregate 3 \
                --system_info system_info.json \
                --gauss_keywords route.txt \
                --mm_monomer m1.chg m2.chg m3.chg \
                --mm_solvent custom_solvent.xyz
```

---

## Outputs

> monomer `.com` files: `monomer1.com`, `monomer2.com`, etc. : including qm, mm solvent if provided.

> dimer `.com` files: `dimer`, etc. : including qm, mm solvent if provided.

XYZ files when `--output_xyz` is specified :
> monomer XYZ files: `monomer<#>_qm.xyz` and its corresponding `monomer<#>_mm_solvent.xyz`

> dimer XYZ files: `dimer_qm.xyz` and its corresponding `dimer_mm_solvent.xyz`

> Temporary files (`_current_frame_data.xyz`) are deleted after use when
processing trajectories.

> `run.log` is created in the current working directory.

---

## Logging & error handling

GIGPy writes a concise console progress bar and mirrors all
messages—including stack traces on uncaught exceptions—to `gigpy_run.log`.
Fatal errors return a non‑zero exit status.

---

### Acknowledgements

Developed with ♥ by *Sayan Adhikari* and contributors.

