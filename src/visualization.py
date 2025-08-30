import matplotlib.pyplot as plt
import numpy as np
import io 
import base64
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

def get_3d_structure(smiles, is_positive=True):
    """
    Generate 3D structure for molecule with different conformations based on interaction type
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    if is_positive:
        # Normal 3D structure for positive interaction
        AllChem.MMFFOptimizeMolecule(mol)
        return Chem.MolToPDBBlock(mol)
    else:
        conf = mol.GetConformer()
        
        # Special handling for Cholesterol
        if "C1CCC2C3CCC4=CC" in smiles:
            # Get all ring systems
            rings = mol.GetRingInfo().AtomRings()
            ring_atoms = set()
            nitrogen_atoms = set()
            
            # Find rings and nitrogen atoms
            for ring in rings:
                ring_atoms.update(ring)
            
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N':
                    nitrogen_atoms.add(atom.GetIdx())
            
            # Get non-ring, non-nitrogen atoms
            other_atoms = set(range(mol.GetNumAtoms())) - ring_atoms - nitrogen_atoms
            
            # Large separation factors
            ring_separation = 30.0
            nitrogen_separation = 40.0  # Even larger separation for nitrogen
            
            # Move ring systems far apart
            for i, ring in enumerate(rings):
                for atom_idx in ring:
                    pos = conf.GetAtomPosition(atom_idx)
                    new_pos = Point3D(
                        pos.x + (i * ring_separation),
                        pos.y + ((i % 3) * ring_separation),
                        pos.z + ((i % 2) * ring_separation)
                    )
                    conf.SetAtomPosition(atom_idx, new_pos)
            
            # Move nitrogen atoms very far in opposite direction
            for atom_idx in nitrogen_atoms:
                pos = conf.GetAtomPosition(atom_idx)
                new_pos = Point3D(
                    pos.x - nitrogen_separation,
                    pos.y - nitrogen_separation,
                    pos.z - nitrogen_separation
                )
                conf.SetAtomPosition(atom_idx, new_pos)
            
            # Move other atoms in different direction
            for atom_idx in other_atoms:
                pos = conf.GetAtomPosition(atom_idx)
                new_pos = Point3D(
                    pos.x - ring_separation/2,
                    pos.y - ring_separation/2,
                    pos.z + ring_separation/2
                )
                conf.SetAtomPosition(atom_idx, new_pos)
            
            # Break all bonds
            editable_mol = Chem.EditableMol(mol)
            bonds = list(mol.GetBonds())
            for bond in bonds:
                editable_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            mol = editable_mol.GetMol()
            
            # Add atomic labels for visualization
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol == 'N':
                    atom.SetProp('atomLabel', 'N_far_separated')
                else:
                    atom.SetProp('atomLabel', f'{symbol}_separated')
            
            return Chem.MolToPDBBlock(mol)
        
        else:
            # Handle other molecules as before
            carbon_atoms = []
            nitrogen_atoms = []
            other_atoms = []
            
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    carbon_atoms.append(atom.GetIdx())
                elif atom.GetSymbol() == 'N':
                    nitrogen_atoms.append(atom.GetIdx())
                else:
                    other_atoms.append(atom.GetIdx())
            
            separation_factor = 20.0  # Increased overall separation
            
            # Move carbons right
            for idx in carbon_atoms:
                pos = conf.GetAtomPosition(idx)
                new_pos = Point3D(
                    pos.x + separation_factor,
                    pos.y,
                    pos.z
                )
                conf.SetAtomPosition(idx, new_pos)
            
            # Move nitrogens far left
            for idx in nitrogen_atoms:
                pos = conf.GetAtomPosition(idx)
                new_pos = Point3D(
                    pos.x - separation_factor * 2,  # Double separation for nitrogen
                    pos.y - separation_factor,
                    pos.z - separation_factor
                )
                conf.SetAtomPosition(idx, new_pos)
            
            # Break all bonds
            editable_mol = Chem.EditableMol(mol)
            for bond in mol.GetBonds():
                editable_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            mol = editable_mol.GetMol()

    return Chem.MolToPDBBlock(mol)

def generate_interaction_heatmap(probability):
    """Generate heatmap visualization of interaction probability"""
    # Create figure and axis with specific size
    plt.clf()  # Clear any existing plots
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Create data matrix for heatmap
    data = np.array([[probability]])
    
    # Choose colormap based on probability
    if probability > 0.7:
        cmap = plt.cm.Greens
    elif probability > 0.5:
        cmap = plt.cm.YlOrRd
    else:
        cmap = plt.cm.Reds_r
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Binding Probability', rotation=270, labelpad=15)
    
    # Add probability value as text
    ax.text(0, 0, f'{probability:.1%}', 
            ha='center', va='center',
            color='black' if 0.3 < probability < 0.7 else 'white',
            fontsize=14, fontweight='bold')
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    plt.title('Interaction Strength Visualization', pad=10)
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Encode to base64
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f'data:image/png;base64,{image_base64}'

def generate_3d_visualization(molecule_data):
    """Generate 3D visualization of molecule"""
    # Implementation for 3D visualization
    pass