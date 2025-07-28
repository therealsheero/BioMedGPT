
---

````markdown
# 🧬 BioMedGPT: AI-Driven Drug Discovery Platform

BioMedGPT is an end-to-end AI system for **generating, optimizing, and evaluating drug-like molecules** using **generative models, graph neural networks (GNNs)**, and **reinforcement learning**. Inspired by DeepMind’s AlphaFold and modern biomedical AI pipelines, this project aims to automate the **design of molecules** targeting specific biological properties or proteins.

---

## 🚧 TODO: Roadmap Checklist

- [ ] ✅ **Phase 1: Literature Review + Paper Replication**
  - [ ] Molecule Generation (VAE, RNN, Diffusion)
  - [ ] Property Prediction (GNN on ESOL, Tox21)
  - [ ] RL for Molecule Optimization (PPO/REINFORCE)

- [ ] ✅ **Phase 2: Modular Codebase**
  - [ ] `generator/`: Molecule generator module
  - [ ] `predictor/`: GNN-based property predictor
  - [ ] `optimizer/`: RL agent & reward engine
  - [ ] `evaluator/`: QED, SA, novelty scoring

- [ ] ✅ **Phase 3: Integrations & Evaluation**
  - [ ] Add protein embeddings (ESM/AlphaFold)
  - [ ] Create molecule evaluation dashboard
  - [ ] Benchmark against MOSES & REINVENT

- [ ] ✅ **Phase 4: Deployment & Presentation**
  - [ ] Create Streamlit/Gradio frontend
  - [ ] Export best candidates to CSV
  - [ ] Create video demo and write Medium post

---

## 📦 Folder Structure

```bash
BioMedGPT/
├── generator/          # VAE / Diffusion model for molecule generation
├── predictor/          # GNN model for property prediction
├── optimizer/          # PPO or REINFORCE-based fine-tuner
├── evaluator/          # Reward function, scoring scripts
├── data/               # Datasets (ZINC, ESOL, etc.)
├── notebooks/          # Paper replications & experiments
├── scripts/            # Training / Evaluation entrypoints
├── app/                # Streamlit or Gradio frontend
└── README.md           # You are here
````

---

## 🔧 Installation Instructions

> ⚠️ Use Python 3.8 or 3.9. Avoid 3.12 due to TorchDrug, MOSES, RDKit issues.

### 🐍 Create Environment

```bash
conda create -n biomedgpt python=3.9
conda activate biomedgpt
```

### 📦 Install Dependencies

```bash
# Core libraries
pip install torch torchvision
pip install pandas scikit-learn tqdm matplotlib seaborn

# Chemistry tools
conda install -c rdkit rdkit
pip install selfies deepchem

# GNNs
pip install torch-geometric

# RL
pip install stable-baselines3

# Optional but recommended
pip install gradio streamlit umap-learn
```

---

## 📚 Datasets

| Dataset       | Purpose                      | Format             |
| ------------- | ---------------------------- | ------------------ |
| ZINC-250K     | Molecule generation          | SMILES             |
| ESOL          | Solubility prediction        | CSV                |
| Tox21         | Toxicity classification      | SDF/CSV            |
| BindingDB     | Binding affinity prediction  | Molecule + Protein |
| UniProt / ESM | Target embeddings (optional) | FASTA              |

---

## 🚀 Project Flow

1. **Train generator** on ZINC to sample novel molecules
2. **Train GNN** to predict QED, solubility, and toxicity
3. **Define reward function** combining properties
4. **Train RL agent** to optimize generated molecules
5. (Optional) **Add protein-conditioning**
6. **Evaluate** on novelty, diversity, drug-likeness
7. (Optional) Build web app or CLI for molecule exploration

---

## 🧠 Key Tools & Libraries

| Category          | Tools/Libraries                             |
| ----------------- | ------------------------------------------- |
| Molecular graphs  | `rdkit`, `selfies`, `torchdrug`, `deepchem` |
| Generative Models | `PyTorch`, `transformers`, `MOSES`          |
| GNNs              | `PyTorch Geometric`, `DGL`                  |
| RL                | `Stable-Baselines3`, `REINVENT`             |
| Protein features  | `ESM`, `BioPython`, `AlphaFold2`            |

---

## 📈 Evaluation Metrics

* **QED**: Drug-likeness
* **SA Score**: Synthetic accessibility
* **LogP / Solubility**
* **Toxicity Score**
* **Validity / Novelty / Uniqueness**
* **(Optional)** Docking Score (via AutoDock or DiffDock)

---

## 🤝 Credits & References

* Olivecrona et al., *"Molecular De Novo Design through Deep Reinforcement Learning"*, 2017
* Gilmer et al., *"Neural Message Passing for Quantum Chemistry"*, 2017

---

