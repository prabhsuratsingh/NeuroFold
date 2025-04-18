import torch
from transformers import BertModel, BertTokenizer
import requests

from new_model import NewDrugTargetModel, encode_protein_protbert, smiles_to_fingerprint

model = NewDrugTargetModel()
model.load_state_dict(torch.load("new_drug_target_model.pth"))
model.eval()

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
bert_model.eval()

def fetch_sequence(uniprot_id):
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
    if r.status_code == 200:
        lines = r.text.split("\n")
        return "".join(lines[1:])
    return None

def fetch_smiles(drug_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
    r = requests.get(url)
    if r.status_code == 200:
        try:
            return r.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
        except:
            return None
    return None

uniprot_id = "P01308"
drug_name = "aspirin"

sequence = fetch_sequence(uniprot_id)
smiles = fetch_smiles(drug_name)

if sequence and smiles:
    protein_tensor = encode_protein_protbert(sequence).unsqueeze(0)
    drug_tensor = smiles_to_fingerprint(smiles).unsqueeze(0)

    with torch.no_grad():
        prediction = model(protein_tensor, drug_tensor).item()
    
    print(f"Predicted binding probability for {drug_name} and {uniprot_id}: {prediction:.4f}")
else:
    print("Invalid protein or drug input")
