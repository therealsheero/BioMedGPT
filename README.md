# Work is in Progress 
#  BioMedGPT: End-to-End Workflow

##  **Overview**

BioMedGPT is a modular AI pipeline for **de novo drug design**.
It involves **generating molecules**, **predicting their properties**, and **optimizing them** for drug-likeness and target specificity.

---

##  **Step 1: Molecule Generation**

| Component  | Description                                                                              |
| ---------- | ---------------------------------------------------------------------------------------- |
| **Input**  | - Random latent vector (noise) or SMILES seed |
| **Model**  | - VAE / Diffusion Model / RNN - Trained on SMILES from ZINC/ChEMBL                  |
| **Output** | - Novel molecule SMILES string(s) like `CC1=CC=CC=C1`                                    |
| **Goal**   | Generate **unique, valid, diverse** molecules that haven't been seen before              |

---

##  **Step 2: Molecular Graph Conversion**

| Component  | Description                                       |
| ---------- | ------------------------------------------------- |
| **Input**  | SMILES string                                     |
| **Tool**   | `RDKit` or `DeepChem`                             |
| **Output** | Molecular graph (nodes = atoms, edges = bonds)    |
| **Goal**   | Convert string to graph format for GNN processing |

---

##  **Step 3: Property Prediction (via GNN)**

| Component     | Description                                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**     | Molecular graph                                                                                                                                   |
| **Model**     | GNN (GCN, MPNN, GAT) trained on labeled datasets (e.g., ESOL, Tox21, BindingDB)                                                                   |
| **Predicted** | - **QED** (drug-likeness score)<br> - **LogP** (solubility) - **Toxicity** (binary/multi) |
| **Output**    | A property score vector per molecule                                                                                                              |
| **Goal**      | Evaluate if the molecule is "promising" chemically and biologically                                                                               |

---

##  **Step 4: Reward Calculation**

| Component  | Description                                                             |
| ---------- | ----------------------------------------------------------------------- |
| **Input**  | Property scores from GNN predictor                                      |
| **Logic**  | Weighted reward = `QED + 1/LogP - Toxicity + Binding Affinity`          |
| **Tool**   | Custom reward function (Python logic or JSON-defined if using REINVENT) |
| **Output** | Single reward score per molecule                                        |
| **Goal**   | Quantify how "good" a molecule is for optimization                      |

---

## **Step 5: Molecule Optimization (Reinforcement Learning)**

| Component   | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| **Input**   | - Initial SMILES <br> - Reward signal                                       |
| **Model**   | - RL agent (REINFORCE / PPO) <br> - Frameworks: REINVENT, Stable-Baselines3 |
| **Actions** | - Edit/add/remove atoms/bonds <br> - Sample new SMILES strings              |
| **Output**  | - Improved molecule(s) with higher reward                                   |
| **Goal**    | Optimize molecules towards drug-like, non-toxic, target-specific properties |

---

##  **Step 6: Protein-Conditioned Generation**

| Component       | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| **Input**       | Protein target sequence (FASTA) or structure (PDB)                 |
| **Tool**        | - ESM-2 or AlphaFold to generate embeddings                        |
| **Integration** | Add protein embedding to molecule generator as conditioning vector |
| **Goal**        | Generate molecules tailored to bind to a specific protein target   |

---

##  **Step 7: Candidate Selection & Evaluation**

| Component   | Description                                                                     |
| ----------- | ------------------------------------------------------------------------------- |
| **Input**   | Top N molecules from generation + optimization pipeline                         |
| **Process** | - Filter on: validity, uniqueness, QED threshold, toxicity flags, novelty check |
| **Tool**    | MOSES metrics / TorchDrug / RDKit scoring                                       |
| **Output**  | Final shortlist of promising candidate molecules                                |
| **Goal**    | Select molecules worth synthesizing, docking, or experimentally testing         |

---

##  **Step 8: (Optional) Visualization + Web Interface**

| Component    | Description                                                              |
| ------------ | ------------------------------------------------------------------------ |
| **Input**    | Final molecule list and property scores                                  |
| **Tool**     | Streamlit / Gradio dashboard                                             |
| **Features** | - SMILES viewer <br> - Score plots <br> - Molecule editing/saving/export |
| **Goal**     | Let user interactively explore and download generated drug candidates    |

---

##  Iterative Loop

After each round:

* You **retrain** or **fine-tune** the generator or predictor
* You improve reward functions (multi-objective weighting)
* You generate and test **better** molecules each cycle

---

## Summary

| Stage                  | Model/Tool         | Input                  | Output                    |
| ---------------------- | ------------------ | ---------------------- | ------------------------- |
| 1. Generate Molecules  | VAE / Diffusion    | Random latent / target | New SMILES                |
| (Optional) 2. Graph Conversion    | RDKit              | SMILES                 | Molecular graph           |
| 3. Property Prediction | GNN (GCN/MPNN)     | Graph                  | Solubility, QED, toxicity |
| 4. Reward Function     | Custom logic       | Property scores        | Single scalar reward      |
| 5. RL Optimization     | PPO / REINVENT     | SMILES + reward        | Better molecules          |
| 6. Target Conditioning | AlphaFold / ESM    | Protein sequence       | Protein-aware molecules   |
| 7. Evaluation          | MOSES / TorchDrug  | Molecules              | Final shortlist           |
| 8. (Optional UI)       | Streamlit / Gradio | Molecules + scores     | Visualization + export    |

---


---

#  BioMedGPT: AI-Driven Drug Discovery Platform

BioMedGPT is an end-to-end AI system designed to generate, optimize, and evaluate novel drug-like molecules using a combination of generative deep learning models, graph neural networks (GNNs), and reinforcement learning.

The system aims to automate de novo drug design, where the objective is to create entirely new molecular structures with desired biochemical properties — such as high solubility, low toxicity, and strong binding affinity to disease-relevant proteins (e.g., cancer targets).

Inspired by breakthroughs like AlphaFold and DiffDock, BioMedGPT goes beyond simple property prediction — it actively designs new candidate molecules that could function as future drugs, tailored to specific protein targets.

---

##  Folder Structure

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

##  Installation Instructions

> ⚠️ Use Python 3.8 or 3.9. Avoid 3.12 due to TorchDrug, MOSES, RDKit issues.

###  Create Environment

```bash
conda create -n biomedgpt python=3.9
conda activate biomedgpt
```

###  Install Dependencies

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

##  Datasets

| Dataset       | Purpose                      | Format             |
| ------------- | ---------------------------- | ------------------ |
| ZINC-250K     | Molecule generation          | SMILES             |
| ESOL          | Solubility prediction        | CSV                |
| Tox21         | Toxicity classification      | SDF/CSV            |
| BindingDB     | Binding affinity prediction  | Molecule + Protein |
| UniProt / ESM | Target embeddings (optional) | FASTA              |

---

##  Project Flow

1. **Train generator** on ZINC to sample novel molecules
2. **Train GNN** to predict QED, solubility, and toxicity
3. **Define reward function** combining properties
4. **Train RL agent** to optimize generated molecules
5. (Optional) **Add protein-conditioning**
6. **Evaluate** on novelty, diversity, drug-likeness
7. (Optional) Build web app or CLI for molecule exploration

---

##  Key Tools & Libraries

| Category          | Tools/Libraries                             |
| ----------------- | ------------------------------------------- |
| Molecular graphs  | `rdkit`, `selfies`, `torchdrug`, `deepchem` |
| Generative Models | `PyTorch`, `transformers`, `MOSES`          |
| GNNs              | `PyTorch Geometric`, `DGL`                  |
| RL                | `Stable-Baselines3`, `REINVENT`             |
| Protein features  | `ESM`, `BioPython`, `AlphaFold2`            |

---

##  Evaluation Metrics

* **QED**: Drug-likeness
* **SA Score**: Synthetic accessibility
* **LogP / Solubility**
* **Toxicity Score**
* **Validity / Novelty / Uniqueness**
* **(Optional)** Docking Score (via AutoDock or DiffDock

---

##  Credits & References

* Olivecrona et al., *"Molecular De Novo Design through Deep Reinforcement Learning"*, 2017
* Gilmer et al., *"Neural Message Passing for Quantum Chemistry"*, 2017

---

