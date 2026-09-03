import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc
from datetime import datetime

# Create charts directory if it doesn't exist
charts_dir = os.path.dirname(__file__)
os.makedirs(charts_dir, exist_ok=True)

# Set style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')

# Load model performance data from CSV files
all_models_df = pd.read_csv(os.path.join(charts_dir, 'all_models_metrics.csv'))
efficientnet_df = pd.read_csv(os.path.join(charts_dir, 'efficientnet_b0_with_clinical.csv'))

# Define colors for consistency
colors = {
    'train': '#1f77b4',  # Blue
    'validation': '#ff7f0e',  # Orange
    'normal': '#2ca02c',  # Green
    'benign': '#d62728',  # Red
    'malignant': '#9467bd'  # Purple
}

# Create training history plots
def create_training_history_plots():
    # Sample training history data (in a real scenario, this would be loaded from a file)
    # This is dummy data for illustration - replace with actual data
    epochs = range(0, 32)
    
    # Training metrics
    train_loss = [1.1, 0.9, 0.8, 0.75, 0.72, 0.7, 0.68, 0.65, 0.6, 0.7, 0.68, 0.55, 0.52, 0.5, 0.48, 0.45, 0.48, 0.5, 0.45, 0.42, 0.4, 0.63, 0.5, 0.45, 0.4, 0.38, 0.36, 0.35, 0.33, 0.35, 0.38, 0.38]
    val_loss = [1.12, 1.05, 0.95, 0.9, 0.85, 0.78, 0.72, 0.68, 0.65, 0.6, 0.55, 0.5, 0.45, 0.43, 0.42, 0.41, 0.41, 0.41, 0.4, 0.38, 0.35, 0.32, 0.3, 0.28, 0.28, 0.27, 0.26, 0.27, 0.26, 0.27, 0.26, 0.27]
    
    train_acc = [0.32, 0.4, 0.48, 0.55, 0.58, 0.65, 0.7, 0.68, 0.72, 0.68, 0.78, 0.67, 0.78, 0.8, 0.82, 0.83, 0.85, 0.83, 0.87, 0.85, 0.9, 0.85, 0.82, 0.86, 0.88, 0.85, 0.9, 0.88, 0.9, 0.87, 0.9, 0.9]
    val_acc = [0.4, 0.48, 0.55, 0.58, 0.65, 0.72, 0.75, 0.68, 0.72, 0.85, 0.75, 0.87, 0.85, 0.89, 0.88, 0.9, 0.88, 0.9, 0.91, 0.91, 0.92, 0.91, 0.92, 0.91, 0.92, 0.92, 0.92, 0.91, 0.92, 0.91, 0.9, 0.9]
    
    train_f1 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.68, 0.72, 0.68, 0.78, 0.66, 0.78, 0.8, 0.81, 0.82, 0.85, 0.82, 0.87, 0.85, 0.89, 0.85, 0.82, 0.85, 0.88, 0.86, 0.91, 0.88, 0.9, 0.88, 0.9, 0.9]
    val_f1 = [0.35, 0.45, 0.58, 0.65, 0.72, 0.75, 0.68, 0.7, 0.75, 0.68, 0.85, 0.7, 0.85, 0.9, 0.88, 0.91, 0.89, 0.91, 0.92, 0.92, 0.93, 0.92, 0.91, 0.92, 0.91, 0.91, 0.91, 0.9, 0.91, 0.9, 0.9, 0.91]
    
    train_auc = [0.55, 0.6, 0.65, 0.7, 0.75, 0.78, 0.82, 0.85, 0.87, 0.85, 0.9, 0.85, 0.9, 0.92, 0.91, 0.93, 0.92, 0.93, 0.94, 0.93, 0.95, 0.94, 0.95, 0.94, 0.96, 0.95, 0.96, 0.97, 0.96, 0.97, 0.97, 0.97]
    val_auc = [0.52, 0.6, 0.7, 0.78, 0.85, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.92, 0.94, 0.95, 0.96, 0.94, 0.95, 0.96, 0.97, 0.96, 0.97, 0.97, 0.98, 0.97, 0.98, 0.97, 0.98, 0.97, 0.98, 0.97, 0.98, 0.98]
    
    train_spec = [0.68, 0.72, 0.75, 0.78, 0.82, 0.85, 0.88, 0.84, 0.89, 0.88, 0.85, 0.9, 0.84, 0.91, 0.92, 0.91, 0.95, 0.91, 0.92, 0.93, 0.94, 0.95, 0.93, 0.92, 0.94, 0.92, 0.95, 0.94, 0.93, 0.95, 0.95, 0.95]
    val_spec = [0.7, 0.75, 0.8, 0.85, 0.88, 0.85, 0.93, 0.88, 0.94, 0.85, 0.95, 0.9, 0.93, 0.95, 0.95, 0.94, 0.95, 0.95, 0.96, 0.95, 0.96, 0.95, 0.96, 0.95, 0.96, 0.96, 0.95, 0.96, 0.95, 0.95, 0.95, 0.95]

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot Loss
    axs[0, 0].plot(epochs, train_loss, color=colors['train'], label='Train Loss')
    axs[0, 0].plot(epochs, val_loss, color=colors['validation'], label='Validation Loss')
    axs[0, 0].set_title('Training and Validation Loss', fontsize=16)
    axs[0, 0].set_xlabel('Epoch', fontsize=12)
    axs[0, 0].set_ylabel('Loss', fontsize=12)
    axs[0, 0].legend()
    
    # Plot Accuracy
    axs[0, 1].plot(epochs, train_acc, color=colors['train'], label='Train Accuracy')
    axs[0, 1].plot(epochs, val_acc, color=colors['validation'], label='Validation Accuracy')
    axs[0, 1].set_title('Training and Validation Accuracy', fontsize=16)
    axs[0, 1].set_xlabel('Epoch', fontsize=12)
    axs[0, 1].set_ylabel('Accuracy', fontsize=12)
    axs[0, 1].legend()
    
    # Plot F1 Score
    axs[1, 0].plot(epochs, train_f1, color=colors['train'], label='Train F1')
    axs[1, 0].plot(epochs, val_f1, color=colors['validation'], label='Validation F1')
    axs[1, 0].set_title('Training and Validation F1 Score', fontsize=16)
    axs[1, 0].set_xlabel('Epoch', fontsize=12)
    axs[1, 0].set_ylabel('F1 Score', fontsize=12)
    axs[1, 0].legend()
    
    # Plot AUC
    axs[1, 1].plot(epochs, train_auc, color=colors['train'], label='Train AUC')
    axs[1, 1].plot(epochs, val_auc, color=colors['validation'], label='Validation AUC')
    axs[1, 1].set_title('Training and Validation AUC', fontsize=16)
    axs[1, 1].set_xlabel('Epoch', fontsize=12)
    axs[1, 1].set_ylabel('AUC', fontsize=12)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate plot for Specificity
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_spec, color=colors['train'], label='Train Specificity')
    plt.plot(epochs, val_spec, color=colors['validation'], label='Validation Specificity')
    plt.title('Training and Validation Specificity', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Specificity', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'specificity_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create confusion matrix
def create_confusion_matrix():
    # Sample confusion matrix data (replace with actual data)
    conf_matrix = np.array([
        [19, 1, 0],
        [1, 12, 1],
        [0, 3, 17]
    ])
    
    # Create labels
    class_names = ['Normal', 'Benign', 'Malignant']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create ROC curve
def create_roc_curve():
    plt.figure(figsize=(10, 8))
    
    # Sample ROC curve data for each class (replace with actual data)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # Normal class
    fpr['normal'] = [0, 0.02, 0.05, 0.1, 1.0]
    tpr['normal'] = [0.08, 0.95, 1.0, 1.0, 1.0]
    roc_auc['normal'] = 1.0
    
    # Benign class
    fpr['benign'] = [0, 0.02, 0.05, 0.1, 0.2, 1.0]
    tpr['benign'] = [0.08, 0.64, 0.78, 0.85, 1.0, 1.0]
    roc_auc['benign'] = 0.93
    
    # Malignant class
    fpr['malignant'] = [0, 0.01, 0.02, 0.05, 1.0]
    tpr['malignant'] = [0.8, 0.95, 0.99, 1.0, 1.0]
    roc_auc['malignant'] = 0.99
    
    # Plot ROC curves
    plt.plot(fpr['normal'], tpr['normal'], color=colors['normal'], lw=2,
             label=f'Normal (AUC = {roc_auc["normal"]:.2f})')
    plt.plot(fpr['benign'], tpr['benign'], color=colors['benign'], lw=2,
             label=f'Benign (AUC = {roc_auc["benign"]:.2f})')
    plt.plot(fpr['malignant'], tpr['malignant'], color=colors['malignant'], lw=2,
             label=f'Malignant (AUC = {roc_auc["malignant"]:.2f})')
    
    # Add reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create additional metrics visualization
def create_additional_metrics():
    # Get the latest efficientnet model data
    model_data = efficientnet_df.iloc[0]
    
    # Create a bar chart for key metrics
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC', 'Specificity']
    values = [
        model_data['Accuracy'],
        model_data['F1-Score'], 
        model_data['Precision'],
        model_data['Recall'],
        model_data['AUC'],
        model_data['Specificity']
    ]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics, values, color='#1f77b4')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12)
    plt.title('EfficientNet-B0 with Clinical Features - Performance Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'additional_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Generate all charts
create_training_history_plots()
create_confusion_matrix()
create_roc_curve()
create_additional_metrics()

print(f"Detailed charts generated successfully and saved to {charts_dir}") 