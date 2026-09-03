import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
import random
import json
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

#Define transformations
train_transform = T.Compose([
    T.Resize((256, 256)),  
    T.RandomResizedCrop(224),  
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(0.3),  
    T.RandomRotation(20),  
    T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Set the path for the dataset folder
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AISSLab")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "transfer_learning/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PRETRAINED_MODELS = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'densenet121': models.densenet121,
    'efficientnet_b0': models.efficientnet_b0,
    'vgg16': models.vgg16,
    'mobilenet_v2': models.mobilenet_v2
}

#Get the image path based on class, side, and view (same as in other ablation scripts)
def get_image_path(row):
    class_name = row["Class"]
    image_name = row["ImageName"]
    birads = row["BI-RADS"]
    birads_folder = birads.replace("-", " ").replace("BI RADS", "BI-RADS")
    
    if class_name == "Normal":
        folder_path = os.path.join(ROOT_DIR, class_name, "JPG")
        return os.path.join(folder_path, image_name)
    
    elif class_name == "Benign":
        jpg_folder_path = os.path.join(ROOT_DIR, class_name, birads_folder, "JPG")
        jpg_file_path = os.path.join(jpg_folder_path, image_name)
        if os.path.exists(jpg_file_path):
            return jpg_file_path
            
        jpg_folder_path = os.path.join(ROOT_DIR, class_name, birads_folder, "jpg")
        jpg_file_path = os.path.join(jpg_folder_path, image_name)
        if os.path.exists(jpg_file_path):
            return jpg_file_path
    
    else:  
        folder_path = os.path.join(ROOT_DIR, class_name, birads_folder, "JPG")
        return os.path.join(folder_path, image_name)
    
    return jpg_file_path if class_name == "Benign" else os.path.join(folder_path, image_name)

#Process clinical features for model input
def process_clinical_features(df):
    """Process clinical features for the entire DataFrame"""
    #Create side vector
    df['side_left'] = (df['Side'].str.strip().str.lower() == 'left').astype(float)
    df['side_right'] = (df['Side'].str.strip().str.lower() == 'right').astype(float)
    
    #Create view vector
    df['view_cc'] = (df['View'].str.strip().str.upper() == 'CC').astype(float)
    df['view_mlo'] = (df['View'].str.strip().str.upper() == 'MLO').astype(float)
    
    #Process BI-RADS
    df['birads_norm'] = df['BI-RADS'].str.split('-').str[-1].astype(float) / 5.0
    
    return df

#Dataset class for Transfer Learning models
class TransferLearningDataset(Dataset):
    def __init__(self, df, transform=None, use_clinical=False, is_training=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_clinical = use_clinical
        self.is_training = is_training
        self.class2idx = {"Normal": 0, "Benign": 0, "Malignant": 1}  #Binary classification
        
        #Calculate class weights for balanced sampling
        class_counts = df["Class"].value_counts()
        max_count = class_counts.max()
        self.class_weights = {
            cls: max_count / count
            for cls, count in class_counts.items()
        }
        
        #Create weighted indices for balanced sampling
        self.indices = []
        for idx in range(len(df)):
            cls = df.iloc[idx]["Class"]
            weight = self.class_weights[cls]
            self.indices.extend([idx] * int(weight))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        
        #Load and transform image
        img_path = row["ImagePath"]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        #Apply transformations
        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(pil_image)
        
        label = torch.tensor(self.class2idx[row["Class"]], dtype=torch.long)
        
        if self.use_clinical:
            clinical_features = np.array([
                row['side_left'], row['side_right'],
                row['view_cc'], row['view_mlo'],
                row['birads_norm']
            ], dtype=np.float32)
            clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32)
            return image_tensor, clinical_tensor, label
        
        return image_tensor, label

#Transfer Learning model that can be configured with different pretrained backbones
class TransferLearningModel(nn.Module):
    def __init__(self, base_model='resnet50', num_classes=3, use_clinical=False, clinical_dim=5, dropout_rate=0.7):
        super().__init__()
        self.use_clinical = use_clinical
        
        #Load pretrained model
        if base_model in PRETRAINED_MODELS:
            self.model = PRETRAINED_MODELS[base_model](pretrained=True)
        else:
            raise ValueError(f"Model {base_model} not supported. Available models: {list(PRETRAINED_MODELS.keys())}")
        
        #Freeze early layers based on freeze_ratio
        if dropout_rate > 0:
            all_params = list(self.model.parameters())
            num_params = len(all_params)
            num_freeze = int(num_params * dropout_rate)
            for param in all_params[:num_freeze]:
                param.requires_grad = False
                
        #Get the input features of the last FC layer according to model architecture
        if base_model.startswith('resnet'):
            last_layer_in_features = self.model.fc.in_features
            self.model.fc = nn.Identity() 
        elif base_model.startswith('densenet'):
            last_layer_in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif base_model.startswith('efficientnet'):
            last_layer_in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
        elif base_model.startswith('vgg'):
            last_layer_in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Identity()
        elif base_model.startswith('mobilenet'):
            last_layer_in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
        else:
            last_layer_in_features = 2048
        
        #Define new classifier layers
        if use_clinical:
            #Clinical features network
            self.clinical_encoder = nn.Sequential(
                nn.Linear(clinical_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            #Fusion classifier with clinical data
            self.classifier = nn.Sequential(
                nn.Linear(last_layer_in_features + 256, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            #Simple classifier without clinical data
            self.classifier = nn.Sequential(
                nn.Linear(last_layer_in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x, clinical=None):
        #Extract features using pretrained model
        features = self.model(x)
        if self.use_clinical and clinical is not None:
            clinical_features = self.clinical_encoder(clinical)
            combined = torch.cat([features, clinical_features], dim=1)
            return self.classifier(combined)
        
        return self.classifier(features)

#Configuration class for Transfer Learning
class Config:
    def __init__(self):
        #Model configuration
        self.base_model = 'efficientnet_b0'
        self.num_classes = 2  
        self.dropout_rate = 0.7  
        self.use_clinical = True
        
        #Training configuration
        self.batch_size = 32  
        self.num_epochs = 50
        self.learning_rate = 0.0005  
        self.weight_decay = 1e-4
        self.early_stopping_patience = 10
        
        self.train_ratio = 0.8  #80% training, 20% testing
        self.image_size = 224
        self.use_augmentation = True
        
        #Paths
        self.data_dir = os.path.join(ROOT_DIR, 'data')
        self.model_dir = os.path.join(ROOT_DIR, 'models')
        self.results_dir = os.path.join(ROOT_DIR, 'results')
        
        #Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

#Calculate all metrics including specificity, AUC, etc.
def calculate_metrics(y_true, y_pred, y_scores=None, num_classes=2):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    #Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #Calculate specificity for each class (multi-class specificity)
    specificity_values = []
    for i in range(num_classes):
        if i < cm.shape[0]:  
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i]) if i < cm.shape[1] else 0
            if (tn + fp) > 0:
                specificity = tn / (tn + fp)
            else:
                specificity = 0.0
            specificity_values.append(specificity)
    
    #Calculate average specificity
    specificity = np.mean(specificity_values) if specificity_values else 0.0
    
    #Calculate AUC
    auc = 0.0
    if y_scores is not None and len(np.unique(y_true)) > 1:
        if num_classes == 2:
            try:
                auc = roc_auc_score(y_true, y_scores[:, 1]) 
            except:
                auc = 0.0
        else:
            try:
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                #Calculate AUC for each class
                auc_values = []
                for i in range(num_classes):
                    if len(np.unique(y_true_bin[:, i])) > 1:  
                        auc_values.append(roc_auc_score(y_true_bin[:, i], y_scores[:, i]))
                
                #Calculate average AUC
                if auc_values:
                    auc = np.mean(auc_values)
            except:
                auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': auc
    }

#Train and evaluate the transfer learning model
def train_and_evaluate(config):
    #Load data from Excel file
    excel_path = os.path.join(ROOT_DIR, "data2.xlsx")
    df = pd.read_excel(excel_path)
    
    print(f"Original data shape: {df.shape}")
    
    #Add image paths
    df["ImagePath"] = df.apply(lambda row: get_image_path(row), axis=1)
    df["ImageExists"] = df["ImagePath"].apply(os.path.exists)
    
    #Filter out images that don't exist
    df_filtered = df[df["ImageExists"]].reset_index(drop=True)
    print(f"Data shape after filtering: {df_filtered.shape}")
    print(f"Class distribution after filtering:")
    print(df_filtered["Class"].value_counts())
    
    df_filtered = process_clinical_features(df_filtered)
    print(f"Data shape after clinical processing: {df_filtered.shape}")
    
    #Split data into train and test sets
    train_data, test_data = train_test_split(
        df_filtered, 
        test_size=0.2,  #20% for test set
        random_state=42,
        stratify=df_filtered['Class']
    )

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    #Create datasets
    train_dataset = TransferLearningDataset(train_data, transform=train_transform, use_clinical=config.use_clinical, is_training=True)
    test_dataset = TransferLearningDataset(test_data, transform=test_transform, use_clinical=config.use_clinical, is_training=False)
    
    #Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    #Create model
    model = TransferLearningModel(
        base_model=config.base_model,
        num_classes=config.num_classes,  
        use_clinical=config.use_clinical,
        dropout_rate=config.dropout_rate
    ).to(DEVICE)
    
    #Calculate class weights for weighted loss
    class_weights = train_data["Class"].value_counts()
    max_count = class_weights.max()
    
    if config.num_classes == 2:
        class_weights_dict = {
            0: max_count / (class_weights.get("Normal", 0) + class_weights.get("Benign", 0)),
            1: max_count / class_weights.get("Malignant", 0)
        }
        weight_tensor = torch.tensor([class_weights_dict[0], class_weights_dict[1]], 
                                    device=DEVICE, dtype=torch.float32)
    else:
        class_weights_dict = {
            0: max_count / class_weights.get("Normal", 0),
            1: max_count / class_weights.get("Benign", 0),
            2: max_count / class_weights.get("Malignant", 0)
        }
        weight_tensor = torch.tensor([class_weights_dict[0], class_weights_dict[1], class_weights_dict[2]], 
                                    device=DEVICE, dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    #Set up optimizer and scheduler
    optimizer = optim.AdamW(  #AdamW yerine Adam
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    #CosineAnnealingWarmRestarts scheduler for learning rate warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,    #reset every 10 epochs
        T_mult=1, 
        eta_min=1e-6
    )
    
    #Set up gradient scaler for mixed precision training
    scaler = GradScaler()
    
    #Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'train_f1': [],
        'test_f1': [],
        'train_specificity': [],
        'test_specificity': [],
        'train_auc': [],
        'test_auc': []
    }
    
    #Best model tracking
    best_f1 = 0.0
    best_epoch = 0
    
    #Early stopping with patience
    patience = config.early_stopping_patience
    no_improve_epochs = 0
    
    #Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        #Set model to training mode
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_scores = []
        
        #Training loop
        progress_bar = tqdm(train_loader, desc=f"Training")
        for i, batch in enumerate(progress_bar):
            if config.use_clinical:
                images, clinical, labels = batch
                images = images.to(DEVICE)
                clinical = clinical.to(DEVICE)
            else:
                images, labels = batch
                images = images.to(DEVICE)
                clinical = None
            
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images, clinical)
                loss = criterion(outputs, labels)
            
            #Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #Update metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_scores.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
            
            #Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (i + 1)
            })
        
        #Calculate training metrics
        train_metrics = calculate_metrics(
            np.array(train_labels), 
            np.array(train_preds),
            np.array(train_scores)
        )
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_specificity'].append(train_metrics['specificity'])
        history['train_auc'].append(train_metrics['auc'])
        
        #Test evaluation
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []
        test_scores = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
                if config.use_clinical:
                    images, clinical, labels = batch
                    images = images.to(DEVICE)
                    clinical = clinical.to(DEVICE)
                else:
                    images, labels = batch
                    images = images.to(DEVICE)
                    clinical = None
                
                labels = labels.to(DEVICE)
                outputs = model(images, clinical)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_scores.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        
        #Calculate test metrics
        test_metrics = calculate_metrics(
            np.array(test_labels), 
            np.array(test_preds),
            np.array(test_scores)
        )
        avg_test_loss = test_loss / len(test_loader)
        history['test_loss'].append(avg_test_loss)
        history['test_accuracy'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['f1'])
        history['test_specificity'].append(test_metrics['specificity'])
        history['test_auc'].append(test_metrics['auc'])
        
        #Update scheduler
        scheduler.step()
        
        #Print epoch summary
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Train F1: {train_metrics['f1']:.4f}, Train Spec: {train_metrics['specificity']:.4f}, "
              f"Train AUC: {train_metrics['auc']:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, "
              f"Test F1: {test_metrics['f1']:.4f}, Test Spec: {test_metrics['specificity']:.4f}, "
              f"Test AUC: {test_metrics['auc']:.4f}")
        
        #Save the best model
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': vars(config)
            }
            
            model_path = os.path.join(config.model_dir, 'best_model.pth')
            torch.save(checkpoint, model_path)
            print(f"Saved best model with F1: {best_f1:.4f}")
        
        if epoch - best_epoch >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break
    
    plot_training_history(history, config.results_dir)
    
    #Load the best model for final evaluation
    model_path = os.path.join(config.model_dir, 'best_model.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    #Final test evaluation
    test_loss = 0.0
    test_preds = []
    test_labels = []
    test_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Final Testing")):
            if config.use_clinical:
                images, clinical, labels = batch
                images = images.to(DEVICE)
                clinical = clinical.to(DEVICE)
            else:
                images, labels = batch
                images = images.to(DEVICE)
                clinical = None
            
            labels = labels.to(DEVICE)
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_scores.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
    
    #Calculate test metrics
    final_test_metrics = calculate_metrics(
        np.array(test_labels), 
        np.array(test_preds),
        np.array(test_scores)
    )
    final_avg_test_loss = test_loss / len(test_loader)
    
    #Print test results
    print("\nTest Results:")
    print(f"Test Loss: {final_avg_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {final_test_metrics['precision']:.4f}")
    print(f"Test Recall: {final_test_metrics['recall']:.4f}")
    print(f"Test F1: {final_test_metrics['f1']:.4f}")
    print(f"Test Specificity: {final_test_metrics['specificity']:.4f}")
    print(f"Test AUC: {final_test_metrics['auc']:.4f}")
    
    #Plot confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    class_names = ['Normal', 'Benign', 'Malignant']
    plot_confusion_matrix(cm, class_names, config.results_dir)
    
    #Save test results
    test_results = {
        'test_loss': final_avg_test_loss,
        'metrics': final_test_metrics,
        'history': history,
        'confusion_matrix': cm.tolist(),
        'config': vars(config),
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'test_scores': [x.tolist() for x in test_scores],
        'test_labels': [int(x) for x in test_labels]
    }
    
    results_path = os.path.join(config.results_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    #Save classification report
    class_report = classification_report(test_labels, test_preds, target_names=class_names)
    report_path = os.path.join(config.results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Transfer Learning Model: {config.base_model}\n")
        f.write(f"Use Clinical Data: {config.use_clinical}\n")
        f.write(f"Dropout Rate: {config.dropout_rate}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nTest Metrics:\n")
        for metric, value in final_test_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return test_results

#Function to plot training history
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 10))
    
    #Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    #Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['test_accuracy'], label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    #Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['test_f1'], label='Test F1')
    plt.title('Training and Test F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    #Plot AUC
    plt.subplot(2, 2, 4)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['test_auc'], label='Test AUC')
    plt.title('Training and Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    #Plot additional metrics
    plt.figure(figsize=(12, 5))
    
    #Plot specificity
    plt.subplot(1, 2, 1)
    plt.plot(history['train_specificity'], label='Train Specificity')
    plt.plot(history['test_specificity'], label='Test Specificity')
    plt.title('Training and Test Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.legend()
    
    #Add any other metrics here
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'additional_metrics.png'))
    plt.close()

#Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, save_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    df = pd.read_excel(os.path.join(ROOT_DIR, "data2.xlsx"))
    print("Original data shape:", df.shape)
    df["ImagePath"] = df.apply(lambda row: get_image_path(row), axis=1)
    df["ImageExists"] = df["ImagePath"].apply(os.path.exists)
    
    #Filter out images that don't exist
    df = df[df["ImageExists"]].reset_index(drop=True)
    print("Data shape after filtering:", df.shape)
    
    print("\nClass distribution after filtering:")
    print(df['Class'].value_counts())
    
    config = Config()
    
    #Run the train and evaluate function
    train_and_evaluate(config)

if __name__ == "__main__":
    main() 