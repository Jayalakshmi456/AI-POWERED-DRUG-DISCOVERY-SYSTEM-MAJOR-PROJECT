import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Standard amino acid vocabulary
PROTEIN_VOCAB = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

# Common SMILES characters vocabulary
DRUG_VOCAB = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4,
    'P': 5, 'Cl': 6, 'Br': 7, 'I': 8, '(': 9,
    ')': 10, '[': 11, ']': 12, '=': 13, '#': 14,
    '+': 15, '-': 16, '.': 17, '/': 18, '\\': 19
}

class SequenceEncoder:
    """Class for encoding protein and drug sequences"""
    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.protein_vocab = PROTEIN_VOCAB
        self.drug_vocab = DRUG_VOCAB
        
    def encode_sequence(self, sequence, is_protein=True):
        """
        Encode a single sequence
        
        Args:
            sequence (str): Input sequence to encode
            is_protein (bool): True if protein sequence, False if drug SMILES
        """
        # Select appropriate vocabulary
        vocab = self.protein_vocab if is_protein else self.drug_vocab
        
        # Encode sequence using vocabulary
        encoded = []
        for char in sequence.upper():
            if char in vocab:
                encoded.append(vocab[char])
            else:
                encoded.append(0)  # Use 0 for unknown characters
                
        # Pad sequence
        padded = pad_sequences(
            [encoded],
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        return padded[0]

def preprocess_input(drug_sequence, protein_sequence, max_length=1000):
    """
    Preprocess a single drug-protein pair for prediction
    """
    # Create encoder instance
    encoder = SequenceEncoder(max_length=max_length)
    
    # Encode sequences
    protein_encoded = encoder.encode_sequence(protein_sequence, is_protein=True)
    drug_encoded = encoder.encode_sequence(drug_sequence, is_protein=False)
    
    # Reshape for model input (add batch dimension)
    protein_encoded = np.expand_dims(protein_encoded, axis=0)
    drug_encoded = np.expand_dims(drug_encoded, axis=0)
    
    return protein_encoded, drug_encoded 