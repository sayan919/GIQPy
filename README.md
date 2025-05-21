# GIGPy – Gaussian Input Generator in Python

**Version**: 1.0  |  **Date**: 20 May 2025

---

## Purpose

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

*(All other imports are from the Python standard library.)*

---

## Arguments

| Flag             | Input type | Value                | Default | Description                                                                                 |
| ---------------- | ---------- | -------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `--single_xyz`    | str        | —                    | —       | Single-frame XYZ file for one calculation                                                   |
| `--traj_xyz`      | str        | —                    | —       | Multi-frame XYZ trajectory file                                                             |
| `--frames`         | int        | —                    | —       | Number of frames to process (required with `--traj_xyz`)                                     |
| `--aggregate`      | int        | —                    | —       | Number of monomers in the system (e.g., 1 for single monomer, 2 for a dimer)                |
| `--system_info`   | str        | —                    | —       | JSON file with monomer and solvent metadata                                                 |
| `--keywords_file` | str        | —                    | —       | Plain‑text file with Gaussian route keywords (one line).                                    |
| `--qm_solvent`    | float      | —                    | 5.0     | Radius cutoff (Å) for QM solvent selection (default: 5.0 Å)                                 |
| `--mm_monomer`    | str        | —                    | —       | Optional: Provide charge files for MM embedding of partner monomers (one per monomer).      |
| `--mm_solvent`    | str        | —                    | —       | Optional: Specify MM solvent charge file or trigger auto‑detection when flag alone is used. |
| `--output_com`    | str        | monomer, dimer, both | both    | Which Gaussian `.com` files to generate (monomers, aggregate, or both).                     |
| `--output_xyz`    | str        | monomer, dimer, both | both    | Which descriptive XYZ files to write; flag with no value defaults to `both`.                |
| `--eetg`           | bool       | —                    | —       | Generate only EETG dimer `.com` (skips VEE files). Requires `--aggregate 2`.                |

---
Run `python gigpy.py --help` for the exhaustive help text.

---

## `system_info.json` schema

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

### Required keys

| Key           | In                | Type      | Notes                                           |
| ------------- | ----------------- | --------- | ----------------------------------------------- |
| `name`        | monomer           | str       | Label used in filenames & titles                |
| `nAtoms`      | monomer + solvent | int       | Atom count in XYZ block                         |
| `charge`      | monomer           | int/float | Formal charge                                   |
| `spin_mult`   | monomer           | int       | Spin multiplicity                               |
| `mol_formula` | monomer + solvent | str       | Empirical formula                               |
| `charges`     | solvent           | list      | One object per atom with `element` and `charge` |

---

## Charge‑file formats

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

| File               | When written                           | Contents                          |                                |
| ------------------ | -------------------------------------- | --------------------------------- | ------------------------------ |
| `monomer<i>.com`   | \`--output\_com monomer                | both\`                            | VEE deck for monomer *i*       |
| \`\<dimer          | aggregate>.com\`                       | ditto                             | VEE deck for whole system      |
| \`\<dimer          | aggregate>\_eetg.com\`                 | `--eetg`                          | EETG deck with fragment labels |
| `*_qm.xyz`         | `--output_xyz`                         | QM atoms (core + QM solvent)      |                                |
| `*_mm_solvent.xyz` | if MM solvent present & `--output_xyz` | Point charges for visualisation   |                                |
| `gigpy_run.log`    | always                                 | Full run log with warnings/errors |                                |

Temporary files (`_current_frame_data.xyz`) are deleted after use when
processing trajectories.

---

## Logging & error handling

GIGPy writes a concise console progress bar and mirrors all
messages—including stack traces on uncaught exceptions—to `gigpy_run.log`.
Fatal errors return a non‑zero exit status.

---

## Development & contributing

1. Fork and clone the repo.
2. Create a virtual environment and install `numpy`.
3. Ensure code passes `flake8` and is formatted with `black`.
4. Submit a pull request.

Feel free to open issues for feature requests or bug reports.

### Acknowledgements

Developed with ♥ by *Sayan Adhikari* and contributors.

