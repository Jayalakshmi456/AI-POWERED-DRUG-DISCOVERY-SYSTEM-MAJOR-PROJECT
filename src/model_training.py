import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate, Embedding
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns
import os

def create_model(seq_length=1000, vocab_size=20):
    """Create a deep learning model for protein-drug interaction prediction"""
    # Protein sequence input
    protein_input = Input(shape=(seq_length,))
    protein_embedded = Embedding(vocab_size, 32)(protein_input)
    protein_conv = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(protein_embedded)
    protein_pooled = GlobalMaxPooling1D()(protein_conv)
    
    # Drug sequence input
    drug_input = Input(shape=(seq_length,))
    drug_embedded = Embedding(vocab_size, 32)(drug_input)
    drug_conv = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(drug_embedded)
    drug_pooled = GlobalMaxPooling1D()(drug_conv)
    
    # Combine protein and drug features
    combined = concatenate([protein_pooled, drug_pooled])
    dense1 = Dense(128, activation='relu')(combined)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=[protein_input, drug_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate_model(X_protein, X_drug, y, epochs=200, batch_size=32):
    """Train and evaluate the model"""
    # Create results directory
    os.makedirs('../results', exist_ok=True)
    
    # Split data with stratification
    X_protein_train, X_protein_test, X_drug_train, X_drug_test, y_train, y_test = train_test_split(
        X_protein, X_drug, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    model = create_model()
    
    # Train model
    history = model.fit(
        [X_protein_train, X_drug_train],
        y_train,
        validation_data=([X_protein_test, X_drug_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Generate predictions
    y_pred_proba = model.predict([X_protein_test, X_drug_test]).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)  # Use FPR and TPR for ROC AUC
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    # Plot ROC curve
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    
    # Plot training history
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/model_evaluation.png')
    plt.close()
    
    # Save model
    model.save('../models/drug_protein_model.h5')
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc
    }
    
    return model, metrics

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    
    # Load and preprocess data
    X_protein, X_drug, y = load_and_preprocess_data('../data/raw/disease.csv')
    
    # Train and evaluate model
    model, metrics = train_and_evaluate_model(X_protein, X_drug, y)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")