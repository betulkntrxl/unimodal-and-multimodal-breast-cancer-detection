import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

# Create charts directory if it doesn't exist
charts_dir = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(charts_dir, exist_ok=True)

# Prepare data for charts
models = {
    'Image+Clinical': {
        'accuracy': 0.814815,
        'f1_score': 0.810000,
        'precision': 0.830000,
        'recall': 0.850000,
        'auc': 0.9200,
        'specificity': 0.920000,
        'training_loss': 0.0664,
        'testing_loss': 0.2091
    },
    'Text': {
        'accuracy': 0.962963,
        'f1_score': 0.962420,
        'precision': 0.966330,
        'recall': 0.962963,
        'auc': 0.9800,
        'specificity': 0.952381,
        'training_loss': 0.1240,
        'testing_loss': 0.1520
    },
    'Random Modality': {
        'accuracy': 0.925926,
        'f1_score': 0.930000,
        'precision': 0.930000,
        'recall': 0.930000,
        'auc': 0.9500,
        'specificity': 0.935000,
        'training_loss': 0.3314,
        'testing_loss': 0.2576
    },
    'GAN-Generated': {
        'accuracy': 0.963000,
        'f1_score': 0.960000,
        'precision': 0.960000,
        'recall': 0.960000,
        'auc': 0.9800,
        'specificity': 0.950000,
        'training_loss': 0.1052,
        'testing_loss': 0.1231
    },
    'Image+LLM': {
        'accuracy': 0.767000,
        'f1_score': 0.773100,
        'precision': 0.805100,
        'recall': 0.766700,
        'auc': 0.8500,
        'specificity': 0.820000,
        'training_loss': 0.2890,
        'testing_loss': 0.3120
    }
}

# Colors for each model
colors = {
    'Image+Clinical': 'green',
    'Text': 'blue',
    'Random Modality': 'gold',
    'GAN-Generated': 'red',
    'Image+LLM': 'purple'
}

# Function to create and save bar charts
def create_bar_chart(metric, ylabel, title):
    plt.figure(figsize=(12, 6))
    model_names = list(models.keys())
    metric_values = [models[model][metric] for model in models]
    
    bars = plt.bar(model_names, metric_values, color=[colors[model] for model in models])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.0)
    plt.ylabel(ylabel)
    plt.xlabel('Model Modality')
    plt.title(f'Model Performance - {title}')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(charts_dir, f'{metric}_chart.png'), dpi=300)
    plt.close()

# Create individual charts for metrics
create_bar_chart('accuracy', 'Accuracy', 'Accuracy')
create_bar_chart('f1_score', 'F1-Score', 'F1-Score')
create_bar_chart('precision', 'Precision', 'Precision')
create_bar_chart('recall', 'Recall', 'Recall')
create_bar_chart('auc', 'AUC', 'AUC')
create_bar_chart('specificity', 'Specificity', 'Specificity')

# Create bar charts for training and testing loss
def create_loss_chart(metric, ylabel, title):
    plt.figure(figsize=(12, 6))
    model_names = list(models.keys())
    metric_values = [models[model][metric] for model in models]
    
    bars = plt.bar(model_names, metric_values, color=[colors[model] for model in models])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylabel(ylabel)
    plt.xlabel('Model Modality')
    plt.title(f'Model Performance - {title}')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(charts_dir, f'{metric}_chart.png'), dpi=300)
    plt.close()

create_loss_chart('training_loss', 'Training Loss', 'Training Loss')
create_loss_chart('testing_loss', 'Testing Loss', 'Testing Loss')

# Create radar chart
def create_radar_chart():
    # Set data
    categories = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Specificity', 'AUC']
    
    # Convert data to coordinates
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw y labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data for each model
    for model_name in models:
        values = [
            models[model_name]['accuracy'],
            models[model_name]['f1_score'],
            models[model_name]['precision'],
            models[model_name]['recall'],
            models[model_name]['specificity'],
            models[model_name]['auc']
        ]
        values += values[:1]  # Close the polygon
        
        # Plot values
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name, color=colors[model_name])
        ax.fill(angles, values, color=colors[model_name], alpha=0.1)
    
    # Add legend
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.1))
    
    # Add title
    plt.title("Model Performance Comparison", size=15, y=1.1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create radar chart
create_radar_chart()

print(f"Charts generated successfully and saved to {charts_dir}") 