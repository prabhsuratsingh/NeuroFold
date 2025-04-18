import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import requests
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

protbert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
protbert_model = BertModel.from_pretrained("Rostlab/prot_bert")
protbert_model.eval()

def encode_protein_protbert(sequence):
    sequence = ' '.join(list(sequence))
    sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    inputs = protbert_tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = protbert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding

def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(fp, dtype=torch.float32)

def fetch_smiles(drug_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
    r = requests.get(url)
    if r.status_code == 200:
        try:
            return r.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
        except:
            return None
    return None

def fetch_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url)
    if r.status_code == 200:
        lines = r.text.split('\n')
        return ''.join(lines[1:])
    return None

class NewDrugTargetModel(nn.Module):
    def __init__(self, protein_dim=1024, drug_dim=1024):
        super(NewDrugTargetModel, self).__init__()
        self.fc1 = nn.Linear(protein_dim + drug_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, protein, drug):
        x = torch.cat((protein, drug), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train():
    train_df = pd.read_csv("bindingdb_sample.csv")
    train_data = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        drug_name = row['drug_name']
        uniprot_id = row['uniprot_id']
        label = row['label']

        smiles = fetch_smiles(drug_name)
        sequence = fetch_sequence(uniprot_id)

        if not smiles or not sequence:
            continue

        try:
            protein_tensor = encode_protein_protbert(sequence)
            drug_tensor = smiles_to_fingerprint(smiles)
            label_tensor = torch.tensor([label], dtype=torch.float32)
            train_data.append((protein_tensor, drug_tensor, label_tensor))
        except Exception as e:
            print(f"Error processing {drug_name}, {uniprot_id}: {e}")
            continue

    model = NewDrugTargetModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    for epoch in range(10):
        total_loss = 0
        for protein_tensor, drug_tensor, label_tensor in train_data:
            optimizer.zero_grad()
            output = model(protein_tensor.unsqueeze(0), drug_tensor.unsqueeze(0))
            loss = loss_fn(output, label_tensor.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "new_drug_target_model.pth")

if __name__ == "__main__":
    train()
