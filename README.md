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
* **Automatic QM/MM partitioning** using a radial cutoff for solvent and
  user‑defined monomer metadata.
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

```bash
pip install numpy
```

All other imports are from the Python standard library.

## Arguments

| Flag              | Argument(s)        | Description                                                                                                                                                               | Required?               |
| ----------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `--single_xyz`    | `<file.xyz>`       | Path to a single-frame XYZ file containing coordinates for entire system (core monomers + solvent).                                                                       | Yes (or `--traj_xyz`)   |
| `--traj_xyz`      | `<file.xyz>`       | Path to a multi-frame XYZ trajectory; each frame in standard XYZ format.                                                                                                  | Yes (or `--single_xyz`) |
| `--frames`        | `<N>`              | Number of frames to process when `--traj_xyz` is used; if omitted all frames are processed.                                                                               | If `--traj_xyz`         |
| `--aggregate`     | `<M>`              | Number of core monomer units (e.g., 2 for a dimer); must match definitions in `system_info`.                                                                              | Yes                     |
| `--system_info`   | `<file.json>`      | JSON defining monomer and solvent metadata (name, nAtoms, charge, spin\_mult, mol\_formula, per-atom charges, etc.).                                                      | Yes                     |
| `--keywords_file` | `<file.txt>`       | Plain-text file of Gaussian route section keywords (one per line). Include PCM/solvation options here as needed.                                                          | Yes                     |
| `--qm_solvent`    | `<radius>`         | Radius in Å for selecting explicit QM solvent shell around core atoms (default 5.0 Å).                                                                                    | No                      |
| `--mm_solvent`    | `[<file.xyz>]`     | Define MM solvent embedding. Provide XYZ-like file of charges, or use flag alone to auto-detect non-QM solvent and assign charges from `system_info`. Omit flag for none. | No                      |
| `--mm_monomer`    | `[0 \| <file1> …]` | MM embedding charges from other monomers: `0` = zero charges; list charge files with “charge x y z” per line.                                                             | No                      |
| `--eetg`          | —                  | Generate only EETG `.com` for dimers (requires `--aggregate 2`). Skips dimer VEE `.com`.                                                                                  | No                      |
| `--output_com`    | `<choice>`         | Which Gaussian `.com` files to write: `monomer`, `dimer`, or `both` (default `both`).                                                                                     | No                      |
| `--output_xyz`    | `[choice]`         | Which descriptive `.xyz` files to write: `monomer`, `dimer`, `both`, or `none` (default `both` when flag present).                                                        | No                      |
| `--tag`           | `<TAG_STRING>`     | Custom tag appended to generated `.com` filenames.                                                                                                                        | No                      |
| `--logfile`       | `<filename>`       | Name for detailed log file (default `gigpy_run.txt`).                                                                                                                     | No                      |

---
Run `python gigpy.py --help` for the exhaustive help text.

---

## Input file : `system_info.json` 

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

## Input file: charges

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

