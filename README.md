# The Dimensional Loss Theorem

**Rigorous proof and neural network validation of universal information loss in lattice embeddings**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18319430.svg)](https://doi.org/10.5281/zenodo.18319430)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

This repository contains the complete validation dataset, analysis code, and supplementary materials for:

**The Dimensional Loss Theorem: Proof and Neural Network Validation**  
*Nathan M. Thornhill*  
Independent Researcher, Fort Wayne, IN  
January 2026

### Abstract

This work presents rigorous component-wise proofs for the Dimensional Loss Theorem, demonstrating that 2D→3D dimensional embedding via middle-slice placement results in:
- **69.2% connectivity loss** (S-component: exact geometric constant 18/26)
- **1/N volumetric dilution** (R-component: proven transformation)
- **Entropy reduction** (D-component: Shannon formula)
- **84-86% total information loss** (combined prediction)

Validation using GPT-2 and Gemma-2 attention maps (N=60) shows **0.000% error** across all theoretical components.

---

## Key Results

| Component | Theoretical Prediction | Observed Error |
|-----------|----------------------|----------------|
| S (Connectivity) | S₃D = (4/13)S₂D | 0.000% ± 0.000% |
| R (Volumetric) | R₃D = R₂D/N | 0.000% ± 0.000% |
| D (Entropy) | D₃D = H(R₂D/N) | 0.000% ± 0.000% |
| **Total Φ Loss** | **84-86%** | **84.39% ± 1.55%** |

**Content-Blindness:** Truth vs lies show no significant difference (p=0.478, Cohen's d=0.18), confirming geometric methods cannot distinguish semantic validity.

---

## Repository Structure
```
dimensional-loss-theorem/
├── README.md                    # This file
├── LICENSE                      # CC BY 4.0
├── requirements.txt             # Python dependencies
├── paper/                       # Published paper
│   └── Thornhill_2026_Dimensional_Loss_Theorem.pdf
├── data/                        # Validation datasets
│   ├── dimensional_stress_data.csv          # Main validation data (N=60)
│   ├── detailed_validation_results.csv      # Extended validation results
│   └── test_sentences.py                   # The 60 test sentences
└── code/                        # Validation scripts
    ├── validate_from_csv.py                # Quick validation + figure generation (RECOMMENDED)
    ├── verification_script.py              # Full neural network validation (advanced)
    └── validation_results/                 # Generated output (not in git)
        ├── validation_results.png          # Main validation figure
        └── detailed_validation_results.csv # Full validation results
```

---

## Quick Start

### Installation

**Method 1: Clone with Git (recommended)**
```bash
git clone https://github.com/existencethreshold/dimensional-loss-theorem.git
cd dimensional-loss-theorem

# Install core dependencies
# Windows:
pip install -r requirements.txt

# Linux/Mac:
pip3 install -r requirements.txt
```

**Method 2: Download ZIP**
```bash
# After downloading and extracting the ZIP file:
cd dimensional-loss-theorem-main

# Install core dependencies
# Windows:
pip install -r requirements.txt

# Linux/Mac:
pip3 install -r requirements.txt
```

### Run Validation

**Option 1: Quick Test (uses pre-computed data) - RECOMMENDED**

This validates the theorem using the pre-computed CSV data. No additional dependencies required.

```bash
cd code

# Windows:
python validate_from_csv.py

# Linux/Mac:
python3 validate_from_csv.py
```

This takes ~30 seconds and proves the theorem without needing transformers.

---

**Option 2: Full Verification (tests on neural networks)**

This tests the theorem on actual GPT-2 and Gemma-2 attention maps. **Note:** First run downloads ~500MB of model weights.

```bash
# Install transformers library first
# Windows:
pip install transformers torch

# Linux/Mac:
pip3 install transformers torch --break-system-packages

cd code

# Windows:
python verification_script.py

# Linux/Mac:
python3 verification_script.py
```

First run takes ~10 minutes (downloads models). Subsequent runs are faster.

---

### Python Version Note

**Windows users:** Use `python` and `pip`  
**Linux/Mac users:** Use `python3` and `pip3`

The scripts work with Python 3.8+

### Expected Output

**From validate_from_csv.py:**
```
======================================================================
 DIMENSIONAL LOSS THEOREM - CSV DATA VALIDATION
 Nathan M. Thornhill - January 21, 2026
======================================================================

Loading validation data from: dimensional_stress_data.csv
✓ Loaded 60 patterns

======================================================================
 VALIDATION RESULTS
======================================================================

Patterns tested: 60
Grid sizes: 8-18 (mean: 10.9)
Average density: 0.1006 ± 0.0069

COMPONENT VALIDATION:
  S-Component error: 0.000% ± 0.000%  ✓ EXACT
  R-Component error: 0.000% ± 0.000%  ✓ EXACT
  D-Component error: 0.000% ± 0.000%  ✓ EXACT

Information Loss:
  Observed: 84.39% ± 1.55%
  Expected: 84-86%
  ✓ WITHIN TOLERANCE

✓✓✓ DIMENSIONAL LOSS THEOREM VALIDATED

Figure saved to: validation_results/validation_results.png
```

**Output files:**
- `validation_results/validation_results.png` - Validation figure
- `validation_results/detailed_validation_results.csv` - Full results

---
```
======================================================================
 DIMENSIONAL LOSS THEOREM VALIDATION
 Nathan M. Thornhill - January 21, 2026
======================================================================

Extracting attention from 60 sentences using gpt2...
[Downloads models on first run - ~500MB]

Patterns tested: 60
Total Φ Loss: 84.39% ± 1.55%

✓✓✓ DIMENSIONAL LOSS THEOREM VALIDATED
     Proceed with publication.

Analysis complete. Check validation_results/ for full results.
```

---

## Troubleshooting

### "FileNotFoundError: attention_maps.npy"

**Problem:** The verification script is looking for pre-saved attention map data that doesn't exist.

**Solution 1 (Quick):** Use the CSV validation instead:
```bash
# Windows:
python validate_from_csv.py

# Linux/Mac:
python3 validate_from_csv.py
```

**Solution 2 (Full test):** Install transformers and generate the data:
```bash
# Windows:
pip install transformers torch

# Linux/Mac:
pip3 install transformers torch --break-system-packages

# Then run verification script
# Windows:
python verification_script.py

# Linux/Mac:
python3 verification_script.py
```

---

### "ModuleNotFoundError: No module named 'transformers'"

**Problem:** The transformers library isn't installed.

**Solution:** Either install transformers OR use the CSV validation:
```bash
# Option 1: Install transformers (for full neural network validation)
# Windows:
pip install transformers torch

# Linux/Mac:
pip3 install transformers torch --break-system-packages

# Option 2: Use CSV validation (faster, no transformers needed)
# Windows:
python validate_from_csv.py

# Linux/Mac:
python3 validate_from_csv.py
```

---

### "command not found: python" (Linux/Mac)

**Problem:** Linux/Mac systems use `python3` instead of `python`.

**Solution:** Use `python3` instead:
```bash
python3 validate_from_csv.py
python3 verification_script.py
```

---

### First run is very slow

**This is normal!** The first run of `verification_script.py` downloads ~500MB of model weights (GPT-2, Gemma-2). This only happens once. Subsequent runs are much faster.

If you want faster validation, use `validate_from_csv.py` instead.

---

## Data Description

### dimensional_stress_data.csv

Complete validation dataset with 60 attention patterns (30 truth/30 lies) from GPT-2 and Gemma-2 models.

**Columns:**
- `sentence`: Input text
- `phi_2d`, `phi_3d`: Integrated information (2D and 3D)
- `loss_pct`: Percentage information loss
- `R_2d`, `S_2d`, `D_2d`: 2D components
- `R_3d`, `S_3d`, `D_3d`: 3D components
- `grid_size`: Attention matrix dimension (N×N)
- `category`: 'truth' or 'lie'

### test_sentences.py

The 60 test sentences used for validation:
- 30 veridical factual statements
- 30 confident hallucinations (structured lies)
- Syntactically matched for length and complexity

---

## Methodology

### Φ Metric Definition

```
Φ = R·S + D

where:
  R = information processing rate (active cells / total cells)
  S = system integration (neighbor-sum method)
  D = disorder (Shannon entropy)
```

### Component Transformations

**S-Component (Connectivity Tax):**
```
S₃D = (8/26) × S₂D = (4/13) × S₂D

Loss = 18/26 ≈ 69.23% (exact geometric constant)
```

**R-Component (Volumetric Dilution):**
```
R₃D = R₂D / N

where N = grid size
```

**D-Component (Entropy Dilution):**
```
D₃D = H(R₂D/N)

where H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{thornhill2026dimensional,
  title={The Dimensional Loss Theorem: Proof and Neural Network Validation},
  author={Thornhill, Nathan M.},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18319430}
}
```

### Related Work

This theorem extends prior empirical findings:

```bibtex
@article{thornhill2026pattern,
  title={Pattern Loss at Dimensional Boundaries: The 86\% Scaling Law},
  author={Thornhill, Nathan M.},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18262424}
}
```

---

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit and indicate if changes were made

---

## Contact

**Nathan M. Thornhill**  
Email: existencethreshold@gmail.com  
ORCID: [0009-0009-3161-528X](https://orcid.org/0009-0009-3161-528X)

---

## Acknowledgments

Computational assistance for manuscript preparation was provided by Claude (Anthropic) and Gemini (Google DeepMind). These large language models operated as software tools under continuous human direction and verification. All theoretical concepts, mathematical proofs, experimental design, data collection, statistical analysis, and scientific interpretations are the sole original intellectual contribution of the author.

---

## Version History

- **v1.0** (January 2026): Initial release with complete validation dataset and theorem proofs
