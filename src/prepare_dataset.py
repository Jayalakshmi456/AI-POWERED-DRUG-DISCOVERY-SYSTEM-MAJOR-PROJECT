# src/prepare_dataset.py
import pandas as pd
import numpy as np
from random import choice, random, seed
import os

# Create necessary directories
os.makedirs('../data/raw', exist_ok=True)
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

def generate_protein_sequence(length=500):
    """Generate synthetic protein sequence"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(choice(amino_acids) for _ in range(length))

def generate_drug_smiles(length=50):
    """Generate synthetic SMILES-like string"""
    atoms = ['C', 'N', 'O', 'P', 'S']
    symbols = ['=', '#', '(', ')', '[', ']', '-']
    smiles = ''
    for _ in range(length):
        if random() < 0.7:
            smiles += choice(atoms)
        else:
            smiles += choice(symbols)
    return smiles

def create_synthetic_dataset(num_drugs=100, num_proteins=500, interactions_per_drug=20):
    """Create synthetic drug-protein interaction dataset"""
    seed(42)
    
    # Create data directories
    os.makedirs('../data/raw', exist_ok=True)
    
    # Generate drugs and proteins
    drugs = [generate_drug_smiles() for _ in range(num_drugs)]
    proteins = [generate_protein_sequence() for _ in range(num_proteins)]
    
    final_data = []
    
    # Generate positive interactions
    for drug in drugs:
        # Select random proteins for each drug
        selected_proteins = np.random.choice(proteins, size=interactions_per_drug, replace=False)
        for protein in selected_proteins:
            final_data.append({
                'drug_sequence': drug,
                'protein_sequence': protein,
                'interaction': 1
            })
    
    # Generate equal number of negative interactions
    num_positive = len(final_data)
    while len(final_data) < 2 * num_positive:
        drug = choice(drugs)
        protein = choice(proteins)
        
        # Check if this pair doesn't exist in positive samples
        if not any(d['drug_sequence'] == drug and d['protein_sequence'] == protein for d in final_data):
            final_data.append({
                'drug_sequence': drug,
                'protein_sequence': protein,
                'interaction': 0
            })
    
    # Create DataFrame
    final_df = pd.DataFrame(final_data)
    
    # Save to CSV
    output_path = '../data/raw/disease.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(final_df)}")
    print(f"Positive interactions: {num_positive}")
    print(f"Negative interactions: {len(final_df) - num_positive}")
    
    return final_df

if __name__ == "__main__":
    create_synthetic_dataset()