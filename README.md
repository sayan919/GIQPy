# GIQPy: Generate Inputs for QM/MM systems

The workflow is split into two scripts:

1. **`giqpy.py`** — turns a trajectory (or single frame) into **.xyz** files: one QM-region file
   and one MM point-charge file per monomer and for the aggregate. 
2. **`xyz_to_gaussian.py`** — converts those `.xyz` files into `Gaussian` **.com** input files
   (per monomer, the aggregate/dimer, or an EET-analysis dimer) with a given set of keywords.

`run_giqpy.sh` chains both stages together.

## Overview

- **Input:**
  - **Single frame**: A single XYZ file containing the coordinates of the entire system (core monomers + solvent).
  - **Multi-frame**: A multi-frame XYZ trajectory, where each frame is in standard XYZ format.
- **Code capabilities:**
  - **Simple usage** : generate a .com file for a single frame system (.xyz) with a given set of keywords.
  - **Configuring aggregates**:
    - We define, no_of_aggregates: 1 = monomer, 2 = dimer, >=2 = aggregate.
    - **`aggregate files`**: all the solute molecules will be present in the .com file and the .xyz file.
    - **`isolated monomer files`**: user has the following options
      - other monomer can be skipped
      - other monomer can be included with zero charges in the MM region
      - other monomer can be included with its specific MM charges in the MM region
  - **Configuring solvent region**: (given a single-frame or multi-frame XYZ file with solute in a solvent)
    - ***`QM solvent`***: user can pick out a localized QM solvent region by specifying a distance cutoff around each atom of the solute. In case of multiple solute molecules, each solvent molecule within the cutoff is assigned to the **single nearest** monomer (by minimum atom-atom distance). This guarantees the QM solvent region is unique to each solute molecule — no solvent atom is ever shared between two monomers.
    - ***`MM solvent`***: user can provide a file with the coordinates of the solvent region, or the code can auto-detect the non-QM solvent and assign charges from the `system_info` JSON.
    - The user can make combinations: no solvent, QM solvent only, MM solvent only, or both QM and MM solvent.
    - This will be executed for each frame in case of a multi-frame trajectory XYZ file.

- **Output:**
  - **.com files**:
    - generated for each monomer and dimer (if applicable) with the specified keywords.
    - genearated for **EET Analysis** calculations with proper formatting. (only for dimers)
  - **.xyz files**: a separate .xyz file for QM region and its corresponding MM solvent region.



## Arguments

### `giqpy.py` (XYZ generation)
`--traj` *(Required)*:
  - ***Number of inputs:*** 1 file
  - Multi-frame trajectory XYZ file. For a single-frame input use `--nFrames 1`.
---
`--nFrames` *(Optional)*:
  - ***Number of inputs:*** 1 integer
  - Number of frames to process (default: all).
---
`--nDyes` *(Required)*:
  - ***Number of inputs:*** 1 integer
  - Number of core monomer units.
---
`--system_info` *(Required)*:
  - ***Number of inputs:*** 1 file
  - JSON defining monomer and solvent metadata (format described below)
---
`--qmSol_radius` *(Optional)*:
  - ***Number of inputs:*** 1 float
  - Radius in Å for selecting explicit QM solvent shell around core atoms
---
`--mm_monomer` *(Optional)*:
  - ***Number of inputs:*** 0 or N files
  - Include MM embedding charges from other monomers
  - `0` = all atoms are assigned zero charges
  -  List charge files with “charge x y z” per line for `N` monomers if `--nDyes N`
  - Omit flag for no MM monomer charges
---
`--mm_solvent` *(Optional)*:
  - ***Number of inputs:*** 0 or 1 file
  - Include MM solvent embedding
  - flag alone to auto-detect non-QM solvent and assign charges from `system_info`
  - or provide XYZ-like file path of charges
  - omit flag for no MM solvent
---
- `--logfile`: (Optional)
  - ***Number of inputs:*** 0 or 1 string
  - Name for detailed log file (default `giqpy_run.log`)

### `xyz_to_gaussian.py` (Gaussian .com generation)
`--indir` *(Optional)*:
  - ***Number of inputs:*** 1 directory
  - Directory holding the XYZ files, or a parent containing numbered per-frame subdirectories (default: current directory).
---
`--nDyes` *(Required)*:
  - ***Number of inputs:*** 1 integer
  - Number of core monomer units (must match the `giqpy.py` run).
---
`--system_info` *(Required)*:
  - ***Number of inputs:*** 1 file
  - Same JSON used by `giqpy.py` (provides charge / spin / names).
---
`--gauss_keywords` *(Required)*:
  - ***Number of inputs:*** 1 file
  - Plain-text file of Gaussian route section keywords (one per line)
---
`--gauss_files` *(Optional)*:
  - ***Number of inputs:*** 1 string : `monomer`, `dimer`, `both` (default `both`)
  - Which `.com` files to generate.
---
`--eetg` *(Optional)*:
  - ***Number of inputs:*** 0 (flag)
  - Generate only EETG `.com` for dimers (requires `--nDyes 2`)
---
- `--tag`: (Optional)
  - ***Number of inputs:*** 0 or 1 string
  - Custom tag appended to generated .com filenames
---
- `--logfile`: (Optional)
  - ***Number of inputs:*** 0 or 1 string
  - Name for detailed log file (default `xyz_to_gaussian.log`)

## Quick Start

```bash
# 1) generate QM-region + MM-charge XYZ files
python giqpy.py --traj my_traj.xyz --nFrames 1 --nDyes 2 \
    --system_info examples/cv_dimer_water.json --qmSol_radius 5 --mm_solvent

# 2) build Gaussian .com files from those XYZ files
python xyz_to_gaussian.py --indir . --nDyes 2 \
    --system_info examples/cv_dimer_water.json \
    --gauss_keywords examples/keywords.txt --gauss_files monomer
```

---

## Input files
### `keywords.txt`
- The file must be a **plain text** file with one entry per line.
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

### `system_info.json` 

- The file must be a **JSON array** with one entry per monomer followed by **one**
entry describing the solvent. 

  Example: [cv_dimer_water.json](./examples/cv_dimer_water.json)
  ```jsonc
  [
    {
      "system"      : "m1",            // label used to NAME this system's output files
      "name"        : "cv",            // descriptive name shown in titles/comments
      "mol_formula" : "C16H12N3O1",
      "nAtoms"      : 32,
      "index"       : "0-31",          // 0-based atom indices for this monomer in the trajectory
      "charge"      : 1,
      "spin_mult"   : 1
    },
    {
      "system"      : "m2",
      "name"        : "cv",
      "mol_formula" : "C16H12N3O1",
      "nAtoms"      : 32,
      "index"       : "32-63",
      "charge"      : 1,
      "spin_mult"   : 1
    },
    {
      "system"      : "solvent",
      "name"        : "water",
      "mol_formula" : "H2O",
      "nAtoms"      : 3,               // atoms per solvent MOLECULE (for grouping)
      "index"       : "64-",           // start at atom 64, take all remaining as solvent
      "charges": 
      [
        { "element": "O", "charge": -0.834 },
        { "element": "H", "charge":  0.417 },
        { "element": "H", "charge":  0.417 }
      ]
    }
  ]
  ```

- **`system`** — the short label used to **name the output files** for that monomer
  (e.g. `m1_qm.xyz`, `m1.com`). Call them `m1`/`m2`, `mA`/`mB`, or anything you like.
  If omitted, the code falls back to `monomer1`, `monomer2`, …
- **`name`** — descriptive name that appears in `.xyz`/`.com` titles and comments (not filenames).
- **`index`** — 0-based atom index spec selecting that system's atoms from each trajectory frame:
  - `"0-31"` → atoms 0 through 31 (inclusive)
  - `"64-"`  → atom 64 through the end (use for the solvent)
  - `"0-9,20-31"` → multiple ranges (atoms need **not** be contiguous)
  - With `index`, the trajectory atom order is flexible (cores need not come first).
  - `index` ranges must not overlap between systems; out-of-range/overlapping indices raise an error.
- **`nAtoms`** — for a monomer it is the atom count (optional when `index` is given; if both are
  present they are cross-checked). For the **solvent** it is atoms-per-molecule and is always required.
- **Legacy mode:** if no monomer has an `index`, atoms are taken sequentially by `nAtoms`
  (`[monomer1 … monomerN][solvent …]`), exactly as before.

---

### `Charges`

- ### (a) Inter‑monomer charges (`--mm_monomer`)

  Plain text with **four** columns (charge x y z) and two header lines (XYZ‑like):

  ```text
  <natoms>
  <comment>
  -0.123   1.234   0.456   -2.345
  …
  ```

  Provide **N** such files when `--nDyes N` so every monomer can be embedded
  in the charges of all other monomers.

- ### (b) Explicit MM solvent (`--mm_solvent`)

  Same format as above but the first column is **charge**, followed by *x y z*.


## Outputs
- **`giqpy.py`** (always, per frame; `{term}` is `dimer` for `--nDyes 2`, else `aggregate`):
  - `{term}_qm.xyz`, `{system}_qm.xyz` : QM-region geometries (core + QM solvent).
  - `{term}_mm.xyz`, `{system}_mm.xyz` : MM point charges (`charge x y z`), only when MM is requested.
    The aggregate MM file holds MM solvent only; monomer MM files also include other-monomer embedding.
    (`{system}` is the per-monomer label from the JSON `system` key, e.g. `m1`, `m2`.)
- **`xyz_to_gaussian.py`** (from the XYZ files above):
  - monomer `.com` files named by the `system` label: `m1.com`, `m2.com`, … (suffix `_qm`/`_mm`/`_qm_mm` reflects solvent).
  - aggregate/dimer `.com` file.
  - EETG `.com` for dimers when `--eetg` is specified.

- Temporary files (`_current_frame_data.xyz`) are deleted after use when processing trajectories.
- Log files (`giqpy_run.log`, `xyz_to_gaussian.log`) are created in the current working directory.

---

## Logging & error handling

GIQPy writes a concise console progress bar and mirrors all
messages—including stack traces on uncaught exceptions—to `giqpy_run.log`.
Fatal errors return a non‑zero exit status.

---

### Acknowledgements

Developed with ♥ by *Sayan Adhikari* and *Claude*
