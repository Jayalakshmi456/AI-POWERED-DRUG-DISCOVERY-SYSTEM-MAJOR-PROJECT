Drug-Protein Interaction Prediction System
Overview
A web-based application that predicts interactions between drugs and proteins using machine learning. 
The system provides detailed analysis, visualization, and interpretation of potential drug-protein binding 
interactions.

Features

Drug-protein interaction prediction
3D molecular visualization
Interactive heatmap generation
Binding probability analysis
Detailed interaction reports
Known interaction database
Real-time molecular structure viewing

Pre-configured Examples

Proteins
Acetylcholinesterase (Alzheimer's target)
Insulin Receptor (Diabetes target)
DNA Polymerase (DNA replication)
Drugs
Donepezil (Alzheimer's treatment)
Metformin (Diabetes medication)
Cholesterol (Steroid molecule)

Technical Requirements

Python 3.8+
Flask
TensorFlow
RDKit
Matplotlib
NumPy

Installation

Clone the repository:
git clone https://github.com/yourusername/drug-protein-interaction-prediction.git
cd drug-protein-interaction-prediction

Create and activate virtual environment:
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Project Structure
drug-protein-interaction-prediction/
├── src/
│   ├── app.py                  # Main Flask application
│   ├── data_preprocessing.py   # Data processing utilities
│   ├── visualization.py        # Visualization functions
│   ├── history_manager.py      # Prediction history management
│   └── train_model.py         # Model training script
├── templates/
│   └── index.html             # Web interface
├── models/
│   └── drug_protein_model.h5  # Trained model
├── static/
│   ├── css/
│   └── js/
└── requirements.txt

Usage

Start the server:
python src/app.py

Access the web interface at http://localhost:5000

Input or select:

Drug SMILES sequence
Protein sequence
Click "Predict Interaction

Special Cases
Pre-configured interactions with known results:

Donepezil + Acetylcholinesterase: 53.7% (moderate positive)
Metformin + Insulin Receptor: 95% (strong positive)
Cholesterol + DNA Polymerase: 0% (strong negative)
Output Analysis
Binding Probability (0-100%)
Binding Strength (Very Weak to Very Strong)
Confidence Level (Low to High)
Interaction Type (Positive/Negative)
Detailed molecular analysis
Interactive 3D visualization
Interaction heatmap

Contributing
Fork the repository
Create feature branch
Commit changes
Push to branch
Create Pull Request
License
MIT License

Authors
[Your Name]

Acknowledgments
RDKit for molecular operations
Flask for web framework
TensorFlow for machine learning
3Dmol.js for molecular visualization