from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rdkit import Chem
import torch
import torch.nn.functional as F

# Load pretrained model from HuggingFace
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def predict_toxicity(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "❌ Invalid SMILES"
        
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            toxic_prob = probs[0][1].item()
            return f"Toxicity Probability: {toxic_prob:.3f}"
    
    except Exception as e:
        return f"Error: {str(e)}"
