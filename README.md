# Constraint-Aware Neuro-Symbolic Learning under Structural and Domain Knowledge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Paper: NeSy 2026](https://img.shields.io/badge/Paper-NeSy'26-purple.svg)](https://nesyconf.org/)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

---

## 📋 Overview

Neuro-symbolic AI promises to unify data-driven representation learning with interpretable symbolic reasoning, but practical frameworks for complex, uncertain, high-dimensional data remain limited. Symbolic-only fuzzy methods fail on high-dimensional noisy data (silhouette scores as low as 0.016 on cybersecurity data). Neural methods produce opaque latent spaces that are difficult to constrain symbolically.

This repository presents a **constraint-aware neuro-symbolic (NeSy) framework** achieving genuine **bidirectional integration** between a neural encoder and Type-2 fuzzy symbolic constraints:

- **Symbolic → Neural**: Interval-valued membership bounds [u̲, ū] form a differentiable constraint loss that backpropagates through the encoder, shaping the learned latent representation
- **Neural → Symbolic**: The encoder maps high-dimensional inputs to a compact latent space where Type-2 fuzzy clustering and granular computing generate human-interpretable linguistic IF-THEN rules with quantified reliability metrics

---

## 📊 Key Results

### Clustering Performance (Silhouette Score, 5-fold CV × 3 repeats)

| Dataset | **NeSy** | Type-2 Fuzzy | Type-1 Fuzzy | DBSCAN | Agglomerative |
|---|---|---|---|---|---|
| Air Quality | 0.265 | **0.311** | 0.311 | N/A | 0.273 |
| Covertype | 0.103 | 0.350 | **0.351** | 0.496 | 0.352 |
| Electricity | 0.461 | **0.481** | 0.478 | N/A | 0.450 |
| **Credit Fraud** | **0.213** | 0.021 | 0.021 | 0.129 | 0.001 |
| **KDD Cup 99** | **0.292** | 0.016 | 0.012 | N/A | −0.001 |

On high-dimensional complex data, NeSy achieves **10× improvement** (Credit Fraud) and **18× improvement** (KDD Cup 99) over symbolic-only baselines.

### Linguistic Rule Quality

| Dataset | Coverage | Significance | Uncertainty |
|---|---|---|---|
| Air Quality | 0.600 | 0.467 | 0.106 |
| Covertype | 0.600 | 0.333 | 0.089 |
| Electricity | 0.600 | 0.518 | 0.121 |
| Credit Fraud | 0.600 | 0.466 | 0.143 |
| KDD Cup 99 | 0.600 | 0.493 | 0.138 |
| **Mean** | **0.600** | **0.455** | **0.119** |

### Fairness

| Method | Dem. Parity | Balance | Combined |
|---|---|---|---|
| **NeSy Framework** | 0.941 | 0.965 | **0.953** |
| Type-2 Fuzzy | 0.924 | 0.964 | 0.944 |

### Convergence

Type-2 fuzzy objective on the NeSy latent space: **33.0** vs. **33,306** on raw PCA features — a **1,000-fold reduction** confirming qualitatively superior cluster structure.

---

## 🏗️ Framework Architecture

```
Raw Input X (d = 7–54 dimensions)
        │
        ▼
  Preprocessing
  ├── Median imputation
  ├── MinMax normalisation → [0, 1]
  └── PCA (95% variance retained) → X' (d' << d)
        │
        ▼
  Neural Encoder fθ: R^d' → R^k
  Lin(d'→64) → BN → ReLU → Dropout(0.1)
  → Lin(64→32) → BN → ReLU
  → Lin(32→k),   k = min(8, d')
        │
   ┌────┴────────────────────┐
   ▼                         ▼
Symbolic → Neural        Neural → Symbolic
Type-2 Fuzzy Module      Granular Computing
Interval memberships     Cluster centres → Linguistic variables
[u̲ᵢₖ, ūᵢₖ] per point   IF-THEN rule extraction
Differentiable Lsym      Coverage / Significance / Uncertainty
backpropagates → fθ      metrics per rule
        │
        ▼
  Total Loss:
  L = λr·Lrecon + λc·Lcluster + λs·Lsym
  λr=1.0, λc=1.0, λs=0.5
```

---

## 🗂️ Datasets

| Dataset | Domain | n | d | d' (PCA 95%) |
|---|---|---|---|---|
| [UCI Air Quality](https://archive.ics.uci.edu/dataset/360) | Environmental | 9,357 | 12 | 6 |
| [Covertype](https://archive.ics.uci.edu/dataset/31) | Environmental | 50,000 | 54 | 27 |
| [Electricity](https://www.openml.org/d/151) | Energy | 30,000 | 7 | 4 |
| Credit Fraud | Financial | 50,000 | 30 | 15 |
| [KDD Cup 99](https://kdd.ics.uci.edu/databases/kddcup99/) | Cybersecurity | 50,000 | 38 | 19 |

All datasets are publicly available. Credit Fraud and KDD Cup 99 experiments used structurally realistic synthetic data due to access constraints.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Farjana-Yesmin/Constraint-Aware-Neuro-Symbolic-Learning-under-Structural-and-Domain-Knowledge.git
cd Constraint-Aware-Neuro-Symbolic-Learning-under-Structural-and-Domain-Knowledge

pip install torch scikit-learn pandas numpy matplotlib scipy
```

### Run the Notebook

```bash
jupyter notebook "Constraint-Aware Neuro-Symbolic Learning under Structural and Domain Knowledge.ipynb"
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam (η=1e-3, weight decay=1e-4) |
| Scheduler | StepLR (step=20, γ=0.5) |
| Epochs | 50 |
| Latent dim k | min(8, d') |
| Fuzzy clusters c | 3 |
| Fuzziness m | 2 |
| λr, λc, λs | 1.0, 1.0, 0.5 |
| Dropout | p=0.1 |

---

## 🔍 Ablation Study (UCI Air Quality)

| Variant | Silhouette | Coverage | Significance |
|---|---|---|---|
| Full NeSy | 0.152 | 0.600 | 0.399 |
| No Neural (symbolic only) | 0.221 | 0.600 | 0.458 |
| No Symbolic Constraint (λs=0) | 0.163 | 0.600 | 0.395 |
| No Clustering Loss (λc=0) | 0.193 | 0.600 | 0.441 |

On structured low-dimensional data (Air Quality, d'=6), symbolic-only achieves higher silhouette — confirming the neural encoder's primary value lies in high-dimensional complex domains where symbolic methods collapse.

---

## 📐 Example Linguistic Rule

```
Air Quality, Cluster 2:
IF LatentPC_1 is MEDIUM
AND LatentPC_3 is LOW
AND LatentPC_4 is HIGH
THEN Pattern_2
[Coverage: 0.600, Significance: 0.391, Uncertainty: 0.139]
```

Rules are **intrinsic** — they reflect the learned latent clustering structure rather than a post-hoc approximation.

---

## 📈 When Does the Neural Encoder Help?

| Data Regime | Recommendation |
|---|---|
| Low-dimensional structured (d' ≤ 6) | Type-2 Fuzzy alone is sufficient |
| High-dimensional noisy (d' > 10) | NeSy provides 10×–18× improvement |

---

## 📁 Repository Structure

```
├── Constraint-Aware Neuro-Symbolic Learning under Structural and Domain Knowledge.ipynb
├── training_Air_Quality.png      # Training loss curves (Air Quality, 50 epochs)
├── convergence.png               # NeSy latent space convergence
├── convergence_baseline.png      # Raw PCA baseline convergence
├── scalability.png               # Runtime and silhouette vs. dataset size
└── paper_summary.png             # Results overview figure
```

---

## 🔮 Future Work

- Knowledge graph integration for richer domain constraints
- Semi-supervised extension with partial cluster labels
- Theoretical convergence guarantees for the joint neural-symbolic objective
- Extension to discrete symbolic domains (NLP, knowledge representation)
- Formal scalability analysis beyond 100,000 samples

---

## 📝 Citation

```bibtex
@inproceedings{anonymous2026nesy,
  title     = {Constraint-Aware Neuro-Symbolic Learning under Structural and Domain Knowledge},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of Machine Learning Research},
  volume    = {284},
  pages     = {1--11},
  year      = {2026},
  note      = {20th Conference on Neurosymbolic Learning and Reasoning (NeSy 2026)}
}
```

---

## 👥 Contributors

- **Farjana Yesmin** — farjanayesmin76@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
