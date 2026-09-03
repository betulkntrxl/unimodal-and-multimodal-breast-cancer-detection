import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

from tqdm import tqdm
import json
from datetime import datetime

from torch.cuda.amp import autocast, GradScaler
import timm
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import clip
import requests

#Root directory for project data
ROOT_DIR = "AISSLab"

#Helper function to determine the correct image folder path based on class and BI-RADS score
def get_image_folder(row):
    class_name = row["Class"]
    birads = row["BI-RADS"]
    if class_name == "Normal":
        return os.path.join(ROOT_DIR, "Normal", "JPG")
    elif class_name == "Benign":
        subfolder = birads.replace("-", " ").replace("BI RADS", "BI-RADS")
        return os.path.join(ROOT_DIR, "Benign", subfolder, "JPG")
    else:
        subfolder = birads.replace("-", " ").replace("BI RADS", "BI-RADS")
        return os.path.join(ROOT_DIR, "Malignant", subfolder, "JPG")

#Builds the complete file path for an image by combining folder and image name
def build_full_path(row):
    img_folder = get_image_folder(row)
    image_name = row["ImageName"]
    return os.path.join(img_folder, image_name)

#Converts clinical features (side, view, BI-RADS) into a normalized vector format for model input
def process_clinical_features(row):
    side = row["Side"].strip().lower()
    side_vec = [1, 0] if side == "left" else [0, 1]
    view = row["View"].strip().upper()
    view_vec = [1, 0] if view == "CC" else [0, 1]
    br_num = float(row["BI-RADS"].split("-")[-1])
    br_norm = br_num / 5.0
    return np.array(side_vec + view_vec + [br_norm], dtype=np.float32)

#Custom dataset class that handles image loading, transformations, and augmentation
class SimpleDataset(Dataset):
    def __init__(self, df, train=True, fusion_mode=0, augment_times=1):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.fusion_mode = fusion_mode
        self.augment_times = augment_times
        self.class2idx = {"Normal": 0, "Benign": 1, "Malignant": 2}
        
        #Calculate augmentation factors to balance classes
        class_counts = df["Class"].value_counts()
        max_count = class_counts.max()
        self.class_augment = {
            cls: int(max_count / count * augment_times)
            for cls, count in class_counts.items()
        }
        
        #Create augmented indices list to achieve class balance
        self.augmented_indices = []
        for idx in range(len(df)):
            cls = df.iloc[idx]["Class"]
            aug_times = self.class_augment[cls]
            self.augmented_indices.extend([idx] * aug_times)
    
    def __len__(self):
        return len(self.augmented_indices)
    
    #Retrieves and processes an image and its label, applying transformations
    def __getitem__(self, idx):
        #Get the actual index from augmented list
        real_idx = self.augmented_indices[idx]
        row = self.df.iloc[real_idx]
        
        #Load image using OpenCV
        img_path = row["ImagePath"]
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        #Convert BGR to RGB format
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        #Convert to tensor and normalize
        tensor_image = torch.from_numpy(cv_image).permute(2, 0, 1).float() / 255.0
        
        #Apply normalization using ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor_image = (tensor_image - mean) / std
        
        #Resize to 224x224 if needed
        if tensor_image.shape[1] != 224 or tensor_image.shape[2] != 224:
            tensor_image = F.interpolate(tensor_image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        #Apply data augmentation for training
        if self.train:
            #Random horizontal flip with 50% probability
            if torch.rand(1) < 0.5:
                tensor_image = tensor_image.flip(dims=[2])  
            
            #Random vertical flip with 20% probability
            if torch.rand(1) < 0.2:
                tensor_image = tensor_image.flip(dims=[1])  
        
        #Get label for the image
        label = torch.tensor(self.class2idx[row["Class"]], dtype=torch.long)
        
        #Return based on fusion mode
        if self.fusion_mode == 0:
            return tensor_image, label
        
        #For clinical data modes
        if self.fusion_mode >= 1:
            #Process clinical features
            clinical = process_clinical_features(row)
            clinical_tensor = torch.tensor(clinical, dtype=torch.float32)
            
            #Create dummy LLM features (zeros)
            llm_features = torch.zeros(768)
            
            #Combine clinical and LLM features
            combined_features = torch.cat([clinical_tensor, llm_features], dim=0)
            
            #Return appropriate combination based on fusion mode
            if self.fusion_mode == 1:
                return combined_features, label
            else:
                return tensor_image, combined_features, label

#Manages text processing and feature extraction using a medical language model
class LLMProcessor:
    def __init__(self, model_name="microsoft/BiomedVLP-CXR-BERT-specialized"):
        print("\nLoading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cache = {} 
        print("LLM model loaded!")
        
    #Generates prompt for medical description based on clinical data
    def generate_medical_description(self, clinical_data, class_name):
        """Generate detailed medical description using clinical data and class information"""
        prompt = f"Mammogram analysis: {class_name} case, {clinical_data['Side']} breast, {clinical_data['View']} view, BI-RADS {clinical_data['BI-RADS']}. Typical findings and recommended follow-up."
        return prompt
    
    #Encodes text into embeddings using the medical language model
    def encode_text(self, text):
        if text in self.cache:
            return self.cache[text]
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1) 
            
        self.cache[text] = features.cpu()
        return self.cache[text]

    #Precomputes and caches LLM features for the entire dataset to improve efficiency
    def precompute_features(self, df):
        """Precompute LLM features for entire dataset"""
        print("\nCalculating LLM features...")
        for idx in tqdm(range(len(df))):
            row = df.iloc[idx]
            clinical_data = {
                'Side': row['Side'],
                'View': row['View'],
                'BI-RADS': row['BI-RADS']
            }
            medical_description = self.generate_medical_description(
                clinical_data, row['Class']
            )
            _ = self.encode_text(medical_description)
        print("LLM features calculated and cached.")

#Advanced dataset class that combines image, clinical, and LLM features
class EnhancedDataset(Dataset):  
    def __init__(self, df, transform=None, fusion_mode=0, llm_processor=None, augment_times=1):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fusion_mode = fusion_mode
        self.augment_times = augment_times
        self.llm_processor = llm_processor
        self.class2idx = {"Normal": 0, "Benign": 1, "Malignant": 2}
        
        class_counts = df["Class"].value_counts()
        max_count = class_counts.max()
        self.class_augment = {
            cls: int(max_count / count * augment_times)
            for cls, count in class_counts.items()
        }
        
        self.augmented_indices = []
        for idx in range(len(df)):
            cls = df.iloc[idx]["Class"]
            aug_times = self.class_augment[cls]
            self.augmented_indices.extend([idx] * aug_times)
    
    def __len__(self):
        return len(self.augmented_indices)
    
    #Gets an item from the dataset with appropriate processing based on fusion mode    
    def __getitem__(self, idx):
        real_idx = self.augmented_indices[idx]
        row = self.df.iloc[real_idx]
        
        #Image processing for modes that use images
        if self.fusion_mode == 0 or self.fusion_mode >= 2:
            #Load image with OpenCV
            img_path = row["ImagePath"]
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            #Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #Convert NumPy array to PIL Image
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            
            #Apply transformations
            if self.transform is not None:
                tensor_image = self.transform(pil_image)
            else:
                #Manual tensor conversion
                tensor_image = torch.tensor(np.array(pil_image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        #Clinical features processing
        if self.fusion_mode >= 1:
            clinical_data = {
                'Side': row['Side'],
                'View': row['View'],
                'BI-RADS': row['BI-RADS']
            }
            
            clinical = process_clinical_features(row)
            clinical_tensor = torch.tensor(clinical, dtype=torch.float32)
            
            #LLM features (using zeros since LLM is disabled)
            llm_features = torch.zeros(768)
            
            #Combine clinical and LLM features
            clinical_llm = torch.cat([clinical_tensor, llm_features], dim=0)
        
        #Get label and return appropriate data based on fusion mode
        label = torch.tensor(self.class2idx[row["Class"]], dtype=torch.long)
        
        if self.fusion_mode == 0:
            return tensor_image, label
        elif self.fusion_mode == 1:
            return clinical_llm, label
        else:
            return tensor_image, clinical_llm, label

#Attention mechanism for combining image and clinical features
class MultiModalAttention(nn.Module):
    def __init__(self, img_dim, clinical_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.img_attention = nn.MultiheadAttention(img_dim, num_heads)
        self.clinical_projection = nn.Linear(clinical_dim, img_dim)
        self.norm1 = nn.LayerNorm(img_dim)
        self.norm2 = nn.LayerNorm(img_dim)
        
    #Combines features using attention mechanisms
    def forward(self, img_features, clinical_features):
        clinical_proj = self.clinical_projection(clinical_features).unsqueeze(0)
        attended_features, _ = self.img_attention(
            clinical_proj, img_features.unsqueeze(0), img_features.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)
        features = self.norm1(img_features + attended_features)
        return self.norm2(features)

#Feature pyramid network for enhanced image feature extraction
class FeaturePyramidFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return F.relu(x)

#Main model architecture combining ResNet backbone with feature fusion capabilities
class EnhancedFusionNet(nn.Module):
    def __init__(self, fusion_mode=0, clinical_dim=5, llm_dim=768, num_classes=3, classifier_input_size=None):
        super().__init__()
        self.fusion_mode = fusion_mode
        
        #ResNet-18 backbone (lighter model)
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        #Feature Pyramid Network layer
        self.fpn = nn.Sequential(
            nn.Conv2d(in_features, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        
        #Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    #Forward pass through the network
    def forward(self, x, clinical_data=None):
        #Using only image features (fusion_mode=0)
        img_features = self.backbone(x)
        img_features = self.fpn(img_features.unsqueeze(-1).unsqueeze(-1))
        img_features = F.adaptive_avg_pool2d(img_features, (1, 1))
        features = img_features.squeeze(-1).squeeze(-1)
        
        return self.classifier(features)

    #Handles missing modalities by generating synthetic data
    def handle_missing_modality(self, x, clinical_data):
        has_image = x is not None
        has_clinical = clinical_data is not None
        
        if not has_image:
            x = self.generate_synthetic_image(clinical_data)
        if not has_clinical:
            clinical_data = self.generate_synthetic_clinical(x)
            
        return x, clinical_data

#Main function for training and evaluating the model
def train_and_evaluate(config):
    #Load data from Excel file
    excel_path = os.path.join(ROOT_DIR, "data2.xlsx")
    df = pd.read_excel(excel_path)
    df["ImagePath"] = df.apply(lambda row: build_full_path(row), axis=1)
    df["ImageExists"] = df["ImagePath"].apply(os.path.exists)
    df = df[df["ImageExists"]].copy()
    print(f"\nTotal data count: {len(df)}")
    
    #Print class distribution
    print("\nClass Distribution:")
    class_dist = df["Class"].value_counts()
    for cls, count in class_dist.items():
        print(f"{cls}: {count} samples ({count/len(df)*100:.2f}%)")
    
    #Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #Set up device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    #Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42, 
        stratify=df["Class"]
    )
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    #Print training class distribution
    print("\nTraining class distribution:")
    train_class_dist = train_df["Class"].value_counts()
    for cls, count in train_class_dist.items():
        print(f"{cls}: {count} samples ({count/len(train_df)*100:.2f}%)")
        
    #Print validation class distribution
    print("\nValidation class distribution:")
    val_class_dist = val_df["Class"].value_counts()
    for cls, count in val_class_dist.items():
        print(f"{cls}: {count} samples ({count/len(val_df)*100:.2f}%)")
    
    #Create dataset instances
    train_dataset = SimpleDataset(
        train_df,
        train=True,
        fusion_mode=config.fusion_mode,
        augment_times=config.augment_times
    )
    
    val_dataset = SimpleDataset(
        val_df,
        train=False,
        fusion_mode=config.fusion_mode,
        augment_times=1  #No augmentation for validation
    )
    
    #Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0  
    )
    
    #Initialize model
    model = EnhancedFusionNet(
        fusion_mode=config.fusion_mode,
        clinical_dim=5,
        llm_dim=768,
        num_classes=3
    ).to(device)
    
    #Calculate balanced class weights
    unique_classes = sorted(df["Class"].unique())
    class_counts = df["Class"].value_counts().sort_index()
    
    print("\nClass distribution for weight calculation:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} samples ({count/len(df)*100:.2f}%)")

    #Set class weights to give more emphasis to Malignant class
    class_weights = torch.tensor(
        [1.0, 1.5, 3.0],  #Manual weights for Normal, Benign, Malignant
        device=device,
        dtype=torch.float32
    )

    print(f"\nClass weights: {class_weights}")
    
    #Loss function with weighted classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    #Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay * 2,  
        amsgrad=True
    )
    
    #Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    #Mixed precision training
    scaler = GradScaler()
    
    #Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    #Variables to track best model and early stopping
    best_val_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    best_epoch = -1
    patience_counter = 0
    results = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'best_model_metrics': None,
        'confusion_matrix': None,
        'classification_report': None
    }
    
    #Training loop
    print("\nStarting training...")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        #Training phase with progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            if config.fusion_mode == 0:
                images, labels = batch
                clinical_data = None
            elif config.fusion_mode == 1:
                clinical_data, labels = batch
                images = None
            else:
                images, clinical_data, labels = batch
            
            labels = labels.to(device)
            if images is not None:
                images = images.to(device)
            if clinical_data is not None:
                clinical_data = clinical_data.to(device)
            
            #Zero gradients
            optimizer.zero_grad()
            
            #Forward pass with mixed precision
            with autocast():
                outputs = model(images, clinical_data)
                loss = criterion(outputs, labels)
            
            #Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #Update metrics
            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            #Update progress bar
            pbar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
        
        #Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        results['train_losses'].append(train_loss)
        results['train_accuracies'].append(train_acc)
        
        #Validation phase
        model.eval()
        val_predictions = []
        val_labels_list = []
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        print(f"\nPerforming validation...")
        with torch.no_grad():
            for batch in val_loader:
                if config.fusion_mode == 0:
                    images, labels = batch
                    clinical_data = None
                elif config.fusion_mode == 1:
                    clinical_data, labels = batch
                    images = None
                else:
                    images, clinical_data, labels = batch
                
                labels = labels.to(device)
                if images is not None:
                    images = images.to(device)
                if clinical_data is not None:
                    clinical_data = clinical_data.to(device)
                
                #Forward pass
                outputs = model(images, clinical_data)
                loss = criterion(outputs, labels)
                
                #Update metrics
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        #Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_metrics = calculate_metrics(val_labels_list, val_predictions)
        
        results['val_losses'].append(val_loss)
        results['val_accuracies'].append(val_acc)
        
        #Print epoch results
        print(f"\nEpoch {epoch+1}/{config.epochs} Results:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1-Score: {val_metrics['f1']:.4f}")
        
        #Update learning rate based on validation F1 score
        scheduler.step(val_metrics['f1'])
        
        #Check if current model is the best so far
        if val_metrics['f1'] > best_val_metrics['f1']:
            best_val_metrics = val_metrics
            best_epoch = epoch
            patience_counter = 0
            print(f"New best F1 score: {val_metrics['f1']:.4f}")
            
            #Generate confusion matrix and classification report
            cm = confusion_matrix(val_labels_list, val_predictions)
            cr = classification_report(val_labels_list, val_predictions, 
                                      target_names=['Normal', 'Benign', 'Malignant'])
            
            results['confusion_matrix'] = cm
            results['classification_report'] = cr
            results['best_model_metrics'] = best_val_metrics
            
            #Save the best model
            os.makedirs('models', exist_ok=True)
            model_path = f'models/best_model_{config.fusion_mode}.pth'
            
            #Create specific best model for test_model.py
            if config.main_model:
                model_path = 'models/best_model_1.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metrics': best_val_metrics,
                'config': vars(config) if hasattr(config, '__dict__') else config,
            }, model_path)
            
            print(f"Model saved to {model_path}")
            
            #Per-class performance
            print("\nPer-class performance:")
            for i, class_name in enumerate(['Normal', 'Benign', 'Malignant']):
                class_correct = (np.array(val_labels_list) == i) & (np.array(val_predictions) == i)
                class_total = (np.array(val_labels_list) == i)
                if np.sum(class_total) > 0:
                    class_acc = np.sum(class_correct) / np.sum(class_total)
                    print(f"{class_name}: {class_acc:.4f}")
                else:
                    print(f"{class_name}: No samples")
                    
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best F1: {best_val_metrics['f1']:.4f} at epoch {best_epoch+1}")
            if patience_counter >= config.patience:
                print(f"\nEarly stopping! No improvement for {config.patience} epochs.")
                break
    
    #Print final results
    print("\n" + "="*50)
    print(f"Training completed. Best results at epoch {best_epoch+1}:")
    print(f"Accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"Precision: {best_val_metrics['precision']:.4f}")
    print(f"Recall: {best_val_metrics['recall']:.4f}")
    print(f"F1-Score: {best_val_metrics['f1']:.4f}")
    
    #Create and save training visualization plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results['train_accuracies'], label='Train Accuracy')
    plt.plot(results['val_accuracies'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'models/training_metrics_{config.fusion_mode}.png')
    
    #Create and save confusion matrix visualization
    if results['confusion_matrix'] is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Benign', 'Malignant'],
                   yticklabels=['Normal', 'Benign', 'Malignant'])
        plt.title(f'Confusion Matrix - Fusion Mode {config.fusion_mode}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'models/confusion_matrix_{config.fusion_mode}.png')
    
    #Save detailed results to a text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'models/results_{config.fusion_mode}_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write(f"Fusion Mode: {config.fusion_mode}\n")
        f.write(f"Best Epoch: {best_epoch+1}\n\n")
        f.write("Best Model Metrics:\n")
        for metric, value in best_val_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
        f.write("\n\nClassification Report:\n")
        f.write(results['classification_report'])
    
    return best_val_metrics

#Main function to run the training process
def main():
    config = type('Config', (), {
        'fusion_mode': 0,          
        'epochs': 40,              
        'batch_size': 12,          
        'learning_rate': 1e-4,     
        'weight_decay': 2e-4,      
        'patience': 8,             
        'augment_times': 3,        
        'use_llm': False,          
        'main_model': True         
    })()
    
    print("\n=== Training Image-Only Model (Without LLM) ===")
    metrics = train_and_evaluate(config)
    
    print("\n=== Final Results ===")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print("\nModel saved to models/best_model_1.pth")

#Calculates performance metrics for model evaluation
def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    main()
