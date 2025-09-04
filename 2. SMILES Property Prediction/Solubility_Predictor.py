import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

model = joblib.load("models/logS_model_xgb_hybrid.pkl")
def compute_rdkit_descriptors(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol)
    ]

def predict_logS(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "❌ Invalid SMILES"

    # Morgan fingerprint (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp)

    # RDKit descriptors (8 features)
    descriptors = compute_rdkit_descriptors(mol)
    feature_vector = np.concatenate([fp_array, descriptors])  # shape: (2056,)

    logS = model.predict([feature_vector])[0]
    return round(logS, 3)
