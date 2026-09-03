import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from PIL import Image
import cv2
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

#Import the classes from the ablation_train.py file
from ablation_train import (
    AblationDataset, TextOnlyModel, RandomModalityModel, 
    Generator, DEVICE, custom_collate_fn, get_image_path,
    AblationConfig
)

RESULTS_DIR = "ablation_experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

#Load the trained model for each ablation type
def load_trained_model(ablation_type):
    config = AblationConfig(ablation_type)
    model_path = os.path.join(config.save_dir, f'best_model_{ablation_type}.pth')
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    if ablation_type == "text_only":
        model = TextOnlyModel().to(DEVICE)
    else:
        model = RandomModalityModel().to(DEVICE)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['best_f1'], checkpoint['epoch']

#Load the trained GAN generator
def load_gan_generator():
    config = AblationConfig("gan")
    gen_path = os.path.join(config.save_dir, 'generator.pth')
    
    if not os.path.exists(gen_path):
        print(f"Generator file not found: {gen_path}")
        return None
    
    generator = Generator(config.latent_dim, config.gen_features).to(DEVICE)
    generator.load_state_dict(torch.load(gen_path))
    generator.eval()
    
    return generator

#Prepare the test data from the excel file
def prepare_test_data():
    excel_path = os.path.join("AISSLab", "data2.xlsx")
    df = pd.read_excel(excel_path)
    
    print(f"Original data shape: {df.shape}")
    
    df["ImagePath"] = df.apply(lambda row: get_image_path(row), axis=1)
    df["ImageExists"] = df["ImagePath"].apply(os.path.exists)
    
    print(f"Images found: {df['ImageExists'].sum()} out of {len(df)}")
    print(f"Class distribution before filtering:")
    print(df["Class"].value_counts())
    df_filtered = df[df["ImageExists"]].reset_index(drop=True)
    print(f"Data shape after filtering: {df_filtered.shape}")
    print(f"Class distribution after filtering:")
    print(df_filtered["Class"].value_counts())
    
    if len(df_filtered["Class"].unique()) < 2:
        print(f"WARNING: Only one class ({df_filtered['Class'].unique()[0]}) found in the dataset after filtering!")
        print("This will affect evaluation metrics and may not represent real-world performance.")
    
    
    if len(df_filtered["Class"].unique()) > 1:
        train_df, test_df = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered["Class"], random_state=42)
    else:
        train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
    
    print(f"Test set samples: {len(test_df)}")
    print(f"Test set class distribution:")
    print(test_df["Class"].value_counts())
    
    return test_df

#Test the model and return metrics
def test_model(model, test_df, ablation_type, generator=None):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = AblationDataset(test_df, ablation_type, transform, train=False)
    collate_fn = custom_collate_fn if ablation_type == "random" else None
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16,
        collate_fn=collate_fn
    )
    all_preds = []
    all_labels = []
    
    if ablation_type == "gan" and generator is not None:
        with torch.no_grad():
            
            noise = torch.randn(16, generator.main[0].in_channels, 1, 1).to(DEVICE)
            generated_images = generator(noise)
            
            sample_dir = os.path.join(RESULTS_DIR, "gan_samples")
            os.makedirs(sample_dir, exist_ok=True)
            
            grid = make_grid(generated_images.cpu(), nrow=4, normalize=True)
            sample_path = os.path.join(sample_dir, f"gan_test_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            save_image(grid, sample_path)
            print(f"Saved GAN test samples to {sample_path}")
    
    class_mapping = {0: "Normal", 1: "Benign", 2: "Malignant"}
    
    with torch.no_grad():
        for batch in test_loader:
            if ablation_type == "text_only":
                features, labels = batch
                features = features.to(DEVICE)
                outputs = model(features)
            else:
                images, features, labels = batch
                if ablation_type == "gan" and generator is not None:
                    batch_size = images.size(0)
                    latent_dim = generator.main[0].in_channels
                    noise = torch.randn(batch_size, latent_dim, 1, 1).to(DEVICE)
                    images = generator(noise)
                    print(f"Using {batch_size} GAN-generated images for testing")
                
                if images is not None:
                    images = images.to(DEVICE)
                if features is not None:
                    features = features.to(DEVICE)
                
                outputs = model(images, features)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    unique_labels = np.unique(all_labels)
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    present_class_names = [class_mapping[label] for label in unique_labels]
    
    class_report = classification_report(
        all_labels, all_preds, 
        labels=unique_labels,
        target_names=present_class_names,
        output_dict=True
    )
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "predictions": all_preds,
        "labels": all_labels,
        "unique_labels": unique_labels,
        "class_names": present_class_names
    }

#Plot the confusion matrix
def plot_confusion_matrix(cm, class_names, title, save_path):
    if len(class_names) == 1:
        
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, f"Only one class: {class_names[0]}\nSamples: {cm[0][0]}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#Plot the comparative metrics
def plot_comparative_metrics(results, save_path):
    ablation_types = list(results.keys())
    accuracies = [results[t]["accuracy"] for t in ablation_types]
    f1_scores = [results[t]["f1_score"] for t in ablation_types]
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(ablation_types))
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    plt.xticks(x, [t.replace('_', ' ').title() for t in ablation_types])
    plt.ylabel('Score')
    plt.title('Comparative Performance of Ablation Approaches')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#Save the detaied results to a text file
def save_results_to_text(results, save_path):
    with open(save_path, 'w') as f:
        f.write(f"ABLATION RESULTS\n")
        f.write(f"=====================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        
        f.write("COMPARATIVE PERFORMANCE\n")
        f.write("-----------------------\n")
        for ablation_type, metrics in results.items():
            f.write(f"{ablation_type.replace('_', ' ').title()}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n\n")
        
        
        for ablation_type, metrics in results.items():
            f.write(f"\nDETAILED RESULTS: {ablation_type.replace('_', ' ').upper()}\n")
            f.write(f"{'-' * (len(ablation_type) + 16)}\n")
            
            f.write("Classification Report:\n")
            report = metrics["classification_report"]
            
            class_mapping = {i: name for i, name in enumerate(metrics["class_names"])}
            
            for class_name, values in report.items():
                if class_name in ['macro avg', 'weighted avg', 'accuracy']:
                    f.write(f"\n{class_name}:\n")
                    if class_name == 'accuracy':
                        f.write(f"  Score: {values:.4f}\n")
                    else:
                        f.write(f"  Precision: {values['precision']:.4f}\n")
                        f.write(f"  Recall: {values['recall']:.4f}\n")
                        f.write(f"  F1-score: {values['f1-score']:.4f}\n")
                        f.write(f"  Support: {values['support']}\n")
                elif class_name.isdigit() or class_name in ['0', '1', '2']:
                    class_idx = int(class_name)
                    if class_idx in class_mapping:
                        class_label = class_mapping[class_idx]
                        f.write(f"\n{class_label}:\n")
                        f.write(f"  Precision: {values['precision']:.4f}\n")
                        f.write(f"  Recall: {values['recall']:.4f}\n")
                        f.write(f"  F1-score: {values['f1-score']:.4f}\n")
                        f.write(f"  Support: {values['support']}\n")
            
            f.write("\nConfusion Matrix:\n")
            cm = metrics["confusion_matrix"]
            class_names = metrics["class_names"]
            
            if len(class_names) == 1:
                f.write(f"  Only one class detected: {class_names[0]}\n")
                f.write(f"  Samples: {cm[0][0]}\n")
            else:
                
                f.write(f"  {' ' * 10}")
                for class_name in class_names:
                    f.write(f"{class_name:^12}")
                f.write("\n")
                
                for i, row in enumerate(cm):
                    f.write(f"  {class_names[i]:^10}")
                    for cell in row:
                        f.write(f"{cell:^12}")
                    f.write("\n")
            
            f.write("\n" + "=" * 50 + "\n")

#Main function that runss the ablation experiments
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_df = prepare_test_data()
    generator = load_gan_generator()
    ablation_types = ["text_only", "random", "gan"]
    results = {}
    
    for ablation_type in ablation_types:
        print(f"\nEvaluating {ablation_type} ablation model...")
        model_data = load_trained_model(ablation_type)
        model, best_f1, best_epoch = model_data
        print(f"Model loaded successfully (best F1: {best_f1:.4f} at epoch {best_epoch+1})")
        
        metrics = test_model(model, test_df, ablation_type, generator if ablation_type == "gan" else None)
        results[ablation_type] = metrics
        
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 score: {metrics['f1_score']:.4f}")
        
        cm_title = f"Confusion Matrix - {ablation_type.replace('_', ' ').title()}"
        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{ablation_type}_{timestamp}.png")
        plot_confusion_matrix(metrics['confusion_matrix'], metrics['class_names'], cm_title, cm_path)
        print(f"Confusion matrix saved to {cm_path}")
    
    if results:
        compare_path = os.path.join(RESULTS_DIR, f"comparative_metrics_{timestamp}.png")
        plot_comparative_metrics(results, compare_path)
        print(f"Comparative metrics plot saved to {compare_path}")
        
        text_path = os.path.join(RESULTS_DIR, f"ablation_results_{timestamp}.txt")
        save_results_to_text(results, text_path)
        print(f"Detailed results saved to {text_path}")
        
        json_path = os.path.join(RESULTS_DIR, f"ablation_results_{timestamp}.json")
        results_json = {k: {
            "accuracy": v["accuracy"],
            "f1_score": v["f1_score"],
            "confusion_matrix": v["confusion_matrix"].tolist(),
            "classification_report": v["classification_report"],
            "unique_labels": v["unique_labels"].tolist(),
            "class_names": v["class_names"]
        } for k, v in results.items()}
        
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=4)
    print("\nProcess completed!")

if __name__ == "__main__":
    main() 