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
├── paper/                       # Published paper (PDF + LaTeX)
├── data/                        # Validation datasets
│   ├── dimensional_stress_data.csv          # Main validation data (N=60)
│   ├── detailed_validation_results.csv      # Extended results
│   ├── residue_data_gpt2.csv               # GPT-2 residue patterns
│   ├── residue_data_google_gemma-2-2b-it.csv  # Gemma-2 residue patterns
│   └── test_sentences.py                   # The 60 test sentences
├── code/                        # Validation scripts
│   ├── verification_script.py              # Main theorem validation
│   └── validate_from_csv.py                # Quick validation from CSV
└── figures/                     # Validation visualizations
    ├── validation_results.png              # Figure 1 from paper
    ├── batch_validation_results.png
    └── results_validate_csv.png
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/existencethreshold/dimensional-loss-theorem.git
cd dimensional-loss-theorem

# Install dependencies
pip install -r requirements.txt
```

### Run Validation

**Option 1: Validate from saved data (fastest)**
```bash
cd code
python validate_from_csv.py
```

**Option 2: Full verification script**
```bash
cd code
python verification_script.py
```

### Expected Output

```
==================================================
 DIMENSIONAL LOSS THEOREM VALIDATION
==================================================

Patterns tested: 60
Grid sizes: 8-18 (mean: 10.9)
Average density: 0.1006 ± 0.0069

COMPONENT VALIDATION:
  S-Component error: 0.000% ± 0.000%  ✓
  R-Component error: 0.000% ± 0.000%  ✓
  D-Component error: 0.000% ± 0.000%  ✓

Total Φ Loss: 84.39% ± 1.55%
Theoretical:  84-86%

✓✓✓ DIMENSIONAL LOSS THEOREM VALIDATED
```

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
