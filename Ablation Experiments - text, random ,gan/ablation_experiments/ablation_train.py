import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
import seaborn as sns
from tqdm import tqdm
import random
import json
from sklearn.model_selection import train_test_split


ROOT_DIR = "AISSLab"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Get the image path based on class, side, and view
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

#Custom collate function that handles None values in the batch
def custom_collate_fn(batch):
    images = []
    features = []
    labels = []
    for item in batch:
        if len(item) == 2:  
            features.append(item[0])
            labels.append(item[1])
        else:  
            img, feat, lbl = item
            images.append(img)
            features.append(feat)
            labels.append(lbl)
    if images:
        
        sample_img = next((img for img in images if img is not None), None)
        if sample_img is not None:
            for i in range(len(images)):
                if images[i] is None:
                    images[i] = torch.zeros_like(sample_img)
            
            images = torch.stack(images)
        else:
            
            images = torch.zeros((len(images), 3, 64, 64))
    if features:
        
        sample_feat = next((feat for feat in features if feat is not None), None)
        if sample_feat is not None:
            
            for i in range(len(features)):
                if features[i] is None:
                    features[i] = torch.zeros_like(sample_feat)
            
            features = torch.stack(features)
        else:
            features = torch.zeros((len(features), 5))
    labels = torch.stack(labels)
    
    if len(batch[0]) == 2:  
        return features, labels
    else:  
        return images, features, labels

#Ablation config class that sets the parameters for the ablation models
class AblationConfig:
    def __init__(self, ablation_type):
        self.ablation_type = ablation_type
        self.epochs = 100
        self.batch_size = 16
        self.learning_rate = 2e-4
        self.weight_decay = 1e-5
        
        if ablation_type == "gan":
            self.latent_dim = 100
            self.gen_features = 64
            self.disc_features = 64
            self.gan_epochs = 50
        
        elif ablation_type == "random":
            self.dropout_prob = 0.5
            
        self.save_dir = f"ablation_experiments/models_{ablation_type}"
        os.makedirs(self.save_dir, exist_ok=True)

#Generator class for the GAN model
class Generator(nn.Module):
    def __init__(self, latent_dim, features=64):
        super().__init__()
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

#Discriminator class for the GAN model
class Discriminator(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d(3, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

#Text only model class
class TextOnlyModel(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.network(x)

#Random modality model class
class RandomModalityModel(nn.Module):
    def __init__(self, image_features=256, text_features=5):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, image=None, text=None):
        if image is None and text is None:
            batch_size = 1
            if hasattr(image, 'shape') and image.shape[0] > 0:
                batch_size = image.shape[0]
            elif hasattr(text, 'shape') and text.shape[0] > 0:
                batch_size = text.shape[0]
            return torch.zeros(batch_size, 3).to(DEVICE)
            
        if image is not None:
            image_features = self.image_encoder(image).view(image.shape[0], -1)
        else:
            batch_size = text.shape[0]
            image_features = torch.zeros(batch_size, 256).to(DEVICE)
        
        if text is not None:
            text_features = self.text_encoder(text)
        else:
            batch_size = image.shape[0]
            text_features = torch.zeros(batch_size, 128).to(DEVICE)
        
        combined = torch.cat([image_features, text_features], dim=1)
        return self.classifier(combined)

#Ablation dataset class that prepares the dataset for the ablation models   
class AblationDataset(Dataset):
    def __init__(self, df, ablation_type, transform=None, train=True):
        self.df = df
        self.ablation_type = ablation_type
        self.transform = transform
        self.train = train
        self.class_to_idx = {"Normal": 0, "Benign": 1, "Malignant": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(self.class_to_idx[row["Class"]], dtype=torch.long)
        clinical_features = self.process_clinical_features(row)
        if self.ablation_type == "text_only":
            return clinical_features, label
        
        image = self.load_image(row["ImagePath"])
        
        if self.ablation_type == "gan":
            return image, clinical_features, label
        
        elif self.ablation_type == "random":
            if self.train and random.random() < 0.5:
                if random.random() < 0.5:
                    return image, None, label
                else:
                    return None, clinical_features, label
            
            return image, clinical_features, label
        return image, clinical_features, label

    def process_clinical_features(self, row):
        side = row["Side"].strip().lower()
        side_vec = [1, 0] if side == "left" else [0, 1]
        
        view = row["View"].strip().upper()
        view_vec = [1, 0] if view == "CC" else [0, 1]
        
        birads = float(row["BI-RADS"].split("-")[-1])
        birads_norm = birads / 5.0
        
        features = np.array(side_vec + view_vec + [birads_norm], dtype=np.float32)
        return torch.tensor(features)

    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        return image

#Train the GAN model
def train_gan(config, data_loader):
    print("Starting GAN training...")
    latent_dim = config.latent_dim
    
    generator = Generator(latent_dim, config.gen_features).to(DEVICE)
    discriminator = Discriminator(config.disc_features).to(DEVICE)
    
    optimizer_g = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.learning_rate * 0.5, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(16, latent_dim, 1, 1).to(DEVICE)
    sample_dir = os.path.join(config.save_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    real_batch = next(iter(data_loader))
    if len(real_batch) == 3:  
        real_grid = make_grid(real_batch[0][:16].cpu(), normalize=True, nrow=4)
        save_image(real_grid, os.path.join(sample_dir, 'real_samples.png'))
        print(f"Saved real image samples to {os.path.join(sample_dir, 'real_samples.png')}")
    
    generator.train()
    discriminator.train()
    
    best_g_loss = float('inf')
    patience_counter = 0
    patience = 10  
    
    for epoch in range(config.gan_epochs):
        g_losses = []
        d_losses = []
        
        for batch in data_loader:
            if len(batch) == 3:  
                real_images = batch[0].to(DEVICE)
            else:  
                continue
                
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size,), 0.9, device=DEVICE)  
            fake_labels = torch.full((batch_size,), 0.1, device=DEVICE)  
            optimizer_d.zero_grad()
            
            outputs_real = discriminator(real_images)
            d_loss_real = criterion(outputs_real, real_labels)
            
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(DEVICE)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()
            
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(DEVICE)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
        
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_loss = sum(d_losses) / len(d_losses)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"GAN Epoch [{epoch+1}/{config.gan_epochs}], d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}")
            
            with torch.no_grad():
                samples = generator(fixed_noise)
                samples = samples.cpu().detach()
                
                grid = make_grid(samples, nrow=4, normalize=True)
                save_image(grid, os.path.join(sample_dir, f'epoch_{epoch+1}.png'))
                print(f"Saved synthetic image samples to {os.path.join(sample_dir, f'epoch_{epoch+1}.png')}")
        
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            patience_counter = 0
            
            torch.save(generator.state_dict(), os.path.join(config.save_dir, 'best_generator.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    generator.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_generator.pth')))
    torch.save(generator.state_dict(), os.path.join(config.save_dir, 'generator.pth'))
    
    with torch.no_grad():
        final_noise = torch.randn(64, latent_dim, 1, 1).to(DEVICE)
        final_samples = generator(final_noise)
        final_samples = final_samples.cpu().detach()
        
        for i, img in enumerate(final_samples):
            save_image(img, os.path.join(sample_dir, f'final_sample_{i+1}.png'), normalize=True)
        
        final_grid = make_grid(final_samples, nrow=8, normalize=True)
        save_image(final_grid, os.path.join(sample_dir, 'final_samples_grid.png'))
        print(f"Saved final synthetic image samples to {os.path.join(sample_dir, 'final_samples_grid.png')}")
    
    print("GAN training completed.")
    return generator

#Train the ablation models
def train_model(config, train_loader, val_loader, model_type="text_only"):
    if model_type == "text_only":
        model = TextOnlyModel().to(DEVICE)
    else:
        model = RandomModalityModel().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_model = None
    best_epoch = 0
    metrics_history = []
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            
            if model_type == "text_only":
                features, labels = batch
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(features)
            else:
                images, features, labels = batch
                if images is not None:
                    images = images.to(DEVICE)
                if features is not None:
                    features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images, features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if model_type == "text_only":
                    features, labels = batch
                    features = features.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(features)
                else:
                    images, features, labels = batch
                    if images is not None:
                        images = images.to(DEVICE)
                    if features is not None:
                        features = features.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(images, features)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_recall_fscore_support(all_labels, all_preds, average='weighted')[0],
            'recall': precision_recall_fscore_support(all_labels, all_preds, average='weighted')[1],
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        val_metrics = {
            'accuracy': accuracy_score(val_labels, val_preds),
            'precision': precision_recall_fscore_support(val_labels, val_preds, average='weighted')[0],
            'recall': precision_recall_fscore_support(val_labels, val_preds, average='weighted')[1],
            'f1': f1_score(val_labels, val_preds, average='weighted')
        }
        
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Val Metrics: {val_metrics}")
        
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        metrics_history.append(epoch_metrics)
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            best_model = model.state_dict().copy()
            
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config.__dict__
            }
            os.makedirs(config.save_dir, exist_ok=True)
            model_path = os.path.join(config.save_dir, f'best_model_{model_type}.pth')
            torch.save(checkpoint, model_path)
            print(f"Best model saved with F1 score: {best_f1:.4f}")
    
    return metrics_history

#Calculate the metrics for the ablation models
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

#Plot the metrics for the ablation models
def plot_metrics(metrics_history, save_dir):
    epochs = [m['epoch'] for m in metrics_history]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [m['train_loss'] for m in metrics_history], label='Train Loss')
    plt.plot(epochs, [m['val_loss'] for m in metrics_history], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [m['train_metrics']['f1'] for m in metrics_history], label='Train F1')
    plt.plot(epochs, [m['val_metrics']['f1'] for m in metrics_history], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

#Main function that runs the ablation experiments   
def main():
    
    excel_path = os.path.join(ROOT_DIR, "data2.xlsx")
    df = pd.read_excel(excel_path)
    print(f"Initial data shape: {df.shape}")
    print(f"Class distribution before filtering:")
    print(df["Class"].value_counts())
    
    df["ImagePath"] = df.apply(lambda row: get_image_path(row), axis=1)
    df["ImageExists"] = df["ImagePath"].apply(lambda x: os.path.exists(x))
    if not df["ImageExists"].all():
        print(f"WARNING: {(~df['ImageExists']).sum()} images not found!")
        print("Sample missing image paths:")
        missing_samples = df[~df["ImageExists"]].head(5)
        for idx, row in missing_samples.iterrows():
            print(f"  {row['ImagePath']}")
    
    df = df[df["ImageExists"]].reset_index(drop=True)
    
    print(f"Data shape after filtering: {df.shape}")
    print(f"Class distribution after filtering:")
    print(df["Class"].value_counts())
    
    stratify = df["Class"] if len(df["Class"].unique()) > 1 else None
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=stratify, random_state=42)
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Training class distribution:")
    print(train_df["Class"].value_counts())
    
    print(f"Validation set: {len(val_df)} samples")
    print(f"Validation class distribution:")
    print(val_df["Class"].value_counts())
    
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ablation_types = ["text_only", "random", "gan"]
    
    for ablation_type in ablation_types:
        print(f"\nRunning {ablation_type} ablation experiment")
        config = AblationConfig(ablation_type)
        
        train_dataset = AblationDataset(train_df, ablation_type, transform, train=True)
        val_dataset = AblationDataset(val_df, ablation_type, transform, train=False)
        
        collate_fn = custom_collate_fn if ablation_type == "random" else None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            collate_fn=collate_fn
        )
        
        if ablation_type == "gan":
            generator = train_gan(config, train_loader)
        metrics_history = train_model(config, train_loader, val_loader, model_type=ablation_type)
        
        plot_metrics(metrics_history, config.save_dir)
        
        with open(os.path.join(config.save_dir, 'metrics_history.json'), 'w') as f:
            json.dump(metrics_history, f, indent=4)

if __name__ == "__main__":
    main() 