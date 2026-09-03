import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import warnings
import traceback

import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast

#Suppress specific warning messages to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization") 

#Import required classes and functions from train.py
try:
    from train import EnhancedFusionNet, LLMProcessor, process_clinical_features
except ImportError as e:
    print(f"Error importing from train.py: {e}")
    print("Please make sure train.py is in the same directory and contains the required classes/functions.")

#Main analyzer class for breast cancer mammogram classification
class MammogramAnalyzer:
    def __init__(self, model_path='models/best_model_1.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        print("Loading model...")

        try:
            #Load model checkpoint from file
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"Checkpoint keys: {checkpoint.keys()}")
            
            #Create model with the same architecture used during training
            self.model = EnhancedFusionNet(
                fusion_mode=0,
                clinical_dim=5,
                llm_dim=768,
                num_classes=3
            )
            
            #Load model weights and check for any issues
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if missing_keys:
                print("\nWarning: Some parameters are missing:")
                for key in missing_keys[:10]:
                    print(f"  - {key}")
                
            if unexpected_keys:
                print("\nWarning: Unexpected keys in state dict:")
                for key in unexpected_keys[:10]:
                    print(f"  - {key}")
            
            self.model = self.model.to(self.device)
            #Set model to evaluation mode
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("Using default model architecture in fallback mode...")
            self.model = EnhancedFusionNet(
                fusion_mode=0,
                clinical_dim=5,
                llm_dim=768,
                num_classes=3
            ).to(self.device)
            self.model.eval()
            print("Model initialized in fallback mode.")

        print("\nLoading LLM...")
        try:
            #Enable LLM for testing
            self.llm_processor = LLMProcessor()
            print("LLM loaded successfully!")
        except Exception as e:
            print(f"\nWarning: Could not load LLM: {str(e)}")
            print("Disabling LLM features...")
            self.llm_processor = None

        #Standard image transformations for preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.class_names = ['Normal', 'Benign', 'Malignant']

    #Processes an image file for model input
    def preprocess_image(self, image_path):
        try:
            #Read image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            
            #Convert from BGR to RGB color format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #Convert to PIL Image format
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            
            #Apply transformations and prepare for model
            image = self.transform(pil_image)
            image = image.unsqueeze(0)
            return image
            
        except Exception as e:
            print(f"Error in preprocess_image: {e}")
            #Return a black image as fallback
            black_image = torch.zeros(1, 3, 224, 224)
            return black_image

    #Analyzes a mammogram image and returns comprehensive results
    def analyze_mammogram(self, image_path, clinical_data, generate_report=True):
        """
        Analyze a mammogram image and return prediction results
        
        Args:
            image_path: Path to the mammogram image
            clinical_data: Dictionary containing clinical data (Side, View, BI-RADS, etc.)
            generate_report: Whether to generate a detailed LLM report
            
        Returns:
            Tuple of (predicted_class, confidence, report, llm_description, detailed_llm_analysis)
        """
        try:
            #Verify image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            #Preprocess image for model input
            image = self.preprocess_image(image_path).to(self.device)
            
            #Create adapter for clinical data
            class ClinicalRow:
                def __init__(self, data):
                    self.data = data
                    
                def __getitem__(self, key):
                    return self.data.get(key, "Unknown")
            
            #Ensure all required clinical data fields are present
            required_keys = ["Side", "View", "BI-RADS"]
            for key in required_keys:
                if key not in clinical_data:
                    clinical_data[key] = "Unknown"
            
            #Process clinical features for model input
            try:
                clinical_row = ClinicalRow(clinical_data)
                clinical_features = process_clinical_features(clinical_row)
                clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            except Exception as e:
                print(f"Error processing clinical features: {e}")
                #Create fallback tensor for clinical features
                clinical_tensor = torch.zeros((1, 5), dtype=torch.float32, device=self.device)

            #Initialize LLM output variables
            llm_description = None
            detailed_llm_analysis = None
            
            #Process text with LLM if available
            if self.llm_processor:
                try:
                    #First LLM query - Basic description
                    medical_description = self.llm_processor.generate_medical_description(
                        clinical_data, "Unknown" 
                    )
                    llm_description = medical_description  
                    llm_features = self.llm_processor.encode_text(medical_description)
                    #Prepare LLM features for model input
                    llm_features = llm_features.view(-1).to(self.device)
                    combined_features = torch.cat([clinical_tensor, llm_features.unsqueeze(0)], dim=1)
                    
                    #Second LLM query - Get more detailed analysis
                    if generate_report:
                        detailed_prompt = f"Perform a detailed analysis for a mammogram with the following characteristics: {clinical_data.get('Side', 'Unknown')} breast, {clinical_data.get('View', 'Unknown')} view, BI-RADS {clinical_data.get('BI-RADS', 'Unknown')}. Include potential findings, differential diagnosis considerations, and recommended follow-up protocols based on ACR guidelines."
                        detailed_llm_analysis = detailed_prompt
                        
                except Exception as e:
                    print(f"Error in LLM processing: {e}")
                    combined_features = torch.cat([
                        clinical_tensor, 
                        torch.zeros((1, 768), dtype=torch.float32, device=self.device)
                    ], dim=1)
            else:
                #Create zero tensor for LLM features if LLM is not available
                combined_features = torch.cat([
                    clinical_tensor, 
                    torch.zeros((1, 768), dtype=torch.float32, device=self.device)
                ], dim=1)

            #Run model inference
            with torch.no_grad():
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    #Pass only image to model (as the model was trained with fusion_mode=0)
                    outputs = self.model(image, None)  
                    
                    #Apply temperature scaling for more confident predictions
                    temperature = 0.5
                    scaled_outputs = outputs / temperature
                    
                    #Convert to probabilities and get predicted class
                    probabilities = torch.softmax(scaled_outputs, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    predicted_class = self.class_names[predicted_idx]
                    confidence = float(probabilities[0][predicted_idx])
                    
            #Generate detailed LLM report if requested
            report = None
            if generate_report and self.llm_processor:
                try:
                    #Create comprehensive report with multiple sections
                    report = f"==== LLM Analysis Report ====\n"
                    report += f"Diagnosis: {predicted_class} (Confidence: {confidence*100:.2f}%)\n\n"
                    report += f"Clinical Data:\n"
                    for key, value in clinical_data.items():
                        report += f"  - {key}: {value}\n"
                    
                    #Add detailed LLM analysis with multiple query aspects
                    report += f"\nDetailed LLM Report:\n"
                    
                    #Primary diagnostic assessment
                    primary_prompt = f"Mammogram analysis: Case diagnosed as {predicted_class}, {clinical_data.get('Side', 'Unknown')} breast, {clinical_data.get('View', 'Unknown')} view, BI-RADS {clinical_data.get('BI-RADS', 'Unknown')}. What are the typical findings and recommended follow-up for this type of case?"
                    report += f"1. Primary Analysis: {primary_prompt}\n\n"
                    
                    #Risk assessment
                    risk_prompt = f"For a patient with BI-RADS {clinical_data.get('BI-RADS', 'Unknown')} and diagnosis of {predicted_class}, what are the risk factors and statistical likelihood of malignancy?"
                    report += f"2. Risk Assessment: {risk_prompt}\n\n"
                    
                    #Treatment recommendations
                    treatment_prompt = f"What are the standard treatment protocols and patient management strategies for a {predicted_class} finding in a {clinical_data.get('BI-RADS', 'Unknown')} categorized mammogram?"
                    report += f"3. Treatment Considerations: {treatment_prompt}\n\n"
                    
                except Exception as e:
                    print(f"Error generating report: {e}")
                    report = "Error generating detailed report."
            
            return predicted_class, confidence, report, llm_description, detailed_llm_analysis
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in analyze_mammogram: {str(e)}")
            return "Unknown", 0.0, None, None, None

#Tests a set number of images from each class and generates a comprehensive report
def test_selected_images(analyzer, excel_path="AISSLab/data2.xlsx", root_dir="AISSLab", samples_per_class=10):
    """
    Tests a specific number of images from each class and generates a report with LLM analysis
    
    Args:
        analyzer: MammogramAnalyzer instance
        excel_path: Path to Excel file containing clinical data
        root_dir: Root directory containing image folders
        samples_per_class: Number of samples to test from each class
    """
    #Load Excel data with clinical information
    if not os.path.exists(excel_path):
        print(f"Excel file not found: {excel_path}")
        return
    
    df = pd.read_excel(excel_path)
    print(f"Loaded {len(df)} records from {excel_path}")
    
    #Add image paths to dataframe
    df['image_path'] = df.apply(lambda row: get_image_path(row, root_dir), axis=1)
    
    #Filter to keep only existing images
    df = df[df['image_path'].apply(lambda x: x != "" and os.path.exists(x))]
    
    if len(df) == 0:
        print("No suitable images found for testing")
        return
    
    #Group images by class and select samples
    class_groups = {}
    for class_name in ['Normal', 'Benign', 'Malignant']:
        class_df = df[df['Class'] == class_name]
        if len(class_df) == 0:
            print(f"Warning: No suitable images found for class {class_name}")
            continue
            
        #Select specified number of samples or all available
        samples = min(samples_per_class, len(class_df))
        class_groups[class_name] = class_df.sample(n=samples).to_dict('records')
        print(f"Selected {samples} samples from {class_name} class")
    
    #Store test results
    results = []
    
    #Create timestamp and results file for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ablation_results_en_{timestamp}.txt"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("===== BREAST CANCER CLASSIFICATION AND LLM ABLATION STUDY =====\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of samples tested per class: {samples_per_class}\n\n")
        
        f.write("===== IMAGE TESTS AND LLM OUTPUTS =====\n\n")
        
        #Arrays to store true and predicted labels
        y_true = []
        y_pred = []
        
        #Test each class separately
        for class_name, examples in class_groups.items():
            f.write(f"\n\n==== {class_name.upper()} CLASS ====\n\n")
            
            for i, example in enumerate(examples, 1):
                image_path = example['image_path']
                image_name = os.path.basename(image_path)
                
                f.write(f"Sample {i}: {image_name}\n")
                f.write(f"True class: {class_name}\n")
                
                #Extract clinical data from example
                clinical_data = {}
                for key in ['Side', 'View', 'BI-RADS']:
                    if key in example and pd.notna(example[key]):
                        clinical_data[key] = example[key]
                    else:
                        clinical_data[key] = "Unknown"
                
                f.write(f"Clinical data: {clinical_data}\n\n")
                
                #Analyze image with model and LLM
                predicted_class, confidence, report, llm_description, detailed_llm_analysis = analyzer.analyze_mammogram(
                    image_path,
                    clinical_data,
                    generate_report=True
                )
                
                #Store results for statistics
                y_true.append(class_name)
                y_pred.append(predicted_class)
                results.append({
                    'image_path': image_path,
                    'true_class': class_name,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'llm_description': llm_description,
                    'detailed_analysis': detailed_llm_analysis,
                    'report': report
                })
                
                #Write model prediction results
                f.write(f"MODEL PREDICTION: {predicted_class} (Confidence: {confidence*100:.2f}%)\n")
                f.write(f"Correct: {'YES' if predicted_class == class_name else 'NO'}\n\n")
                
                #Write basic LLM output
                f.write("LLM OUTPUT:\n")
                if llm_description:
                    f.write(f"{llm_description}\n\n")
                else:
                    f.write("No LLM output could be generated.\n\n")
                
                #Write detailed LLM analysis if available
                if detailed_llm_analysis:
                    f.write("DETAILED LLM ANALYSIS:\n")
                    f.write(f"{detailed_llm_analysis}\n\n")
                
                #Write comprehensive report
                if report:
                    f.write("COMPREHENSIVE REPORT:\n")
                    f.write(f"{report}\n\n")
                
                f.write("-" * 80 + "\n\n")
        
        #Calculate and report overall statistics
        f.write("\n\n===== OVERALL STATISTICS =====\n\n")
        
        #Calculate class-based performance metrics
        class_counts = {}
        correct_counts = {}
        
        for true_class, pred_class in zip(y_true, y_pred):
            if true_class not in class_counts:
                class_counts[true_class] = 0
                correct_counts[true_class] = 0
                
            class_counts[true_class] += 1
            if true_class == pred_class:
                correct_counts[true_class] += 1
        
        f.write("Class-Based Performance:\n")
        f.write("-" * 30 + "\n")
        overall_correct = 0
        overall_total = 0
        
        #Report performance for each class
        for cls in ['Normal', 'Benign', 'Malignant']:
            if cls in class_counts and class_counts[cls] > 0:
                accuracy = (correct_counts.get(cls, 0) / class_counts[cls]) * 100
                f.write(f"{cls:<10}: {correct_counts.get(cls, 0)}/{class_counts[cls]} ({accuracy:.1f}%)\n")
                overall_correct += correct_counts.get(cls, 0)
                overall_total += class_counts[cls]
            else:
                f.write(f"{cls:<10}: No samples found\n")
        
        #Calculate and report overall accuracy
        overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
        f.write(f"\nOverall Accuracy: {overall_correct}/{overall_total} ({overall_accuracy:.1f}%)\n\n")
        
        #Calculate precision, recall, and F1 score
        from sklearn.metrics import precision_recall_fscore_support
        if len(y_true) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, 
                average='weighted', 
                zero_division=0,
                labels=['Normal', 'Benign', 'Malignant']
            )
            
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
        
        #Generate and report confusion matrix
        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=['Normal', 'Benign', 'Malignant'])
            f.write("Confusion Matrix:\n")
            f.write("                Normal  Benign  Malignant\n")
            for i, row_name in enumerate(['Normal', 'Benign', 'Malignant']):
                f.write(f"{row_name:<10}:  ")
                for j in range(cm.shape[1]):
                    f.write(f"{cm[i, j]:<8}")
                f.write("\n")
            
            f.write("\n")
            
            #Detailed classification report
            f.write("Classification Report:\n")
            cr = classification_report(
                y_true, y_pred, 
                target_names=['Normal', 'Benign', 'Malignant'],
                zero_division=0
            )
            f.write(cr)
        
        #Ablation study analysis and conclusions
        f.write("\n\n===== ABLATION STUDY ANALYSIS =====\n\n")
        f.write("This test evaluates a model trained with an image-only approach (fusion_mode=0)\n")
        f.write("while incorporating LLM analysis during the testing phase to assess its potential contribution.\n\n")
        
        f.write("Training: Image-only model (without LLM integration)\n")
        f.write("Testing: Image processing + LLM analysis (for reporting)\n\n")
        
        f.write("This approach allows us to evaluate both the base classification performance of the model\n")
        f.write("and the potential value of LLM as a clinical decision support mechanism.\n")
        
        f.write("\n===== TEST COMPLETED =====\n")
        
    #Generate and save confusion matrix visualization
    if len(y_true) > 0:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred, labels=['Normal', 'Benign', 'Malignant'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Benign', 'Malignant'],
                   yticklabels=['Normal', 'Benign', 'Malignant'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        plt.savefig(f"confusion_matrix_en_{timestamp}.png")
        plt.close()
    
    print(f"\nTest completed. Results saved to {results_file}")
    return results

#Utility function to find the image path from Excel data
def get_image_path(row, root_dir):
    """
    Creates an image path from Excel data
    
    Args:
        row: DataFrame row (containing Class, BI-RADS and ImageName information)
        root_dir: Root directory containing image folders
    
    Returns:
        Path to the image file
    """
    #Map class name to folder structure
    class_name = row['Class']
    
    #Determine base directory path based on class and BI-RADS
    if class_name == "Normal":
        #Normal class images are directly in JPG folder
        class_dir = os.path.join(root_dir, "Normal", "JPG")
    else:
        #Benign and Malignant are in "BI-RADS X/JPG" subfolders
        bi_rads = row['BI-RADS']
        #Convert "BI-RADS-X" format to "BI-RADS X" format
        if '-' in bi_rads:
            bi_rads_value = bi_rads.split('-')[-1]
            bi_rads_folder = f"BI-RADS {bi_rads_value}"
        else:
            bi_rads_folder = bi_rads
        
        class_dir = os.path.join(root_dir, class_name, bi_rads_folder, "JPG")
    
    #Check if directory exists
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return ""
    
    #Find image by exact name if provided
    if 'ImageName' in row and row['ImageName'] and not pd.isna(row['ImageName']):
        image_name = row['ImageName']
        
        #Handle different naming conventions
        if '_' in image_name:
            #Try to match patient ID and view
            parts = image_name.split('_')
            patient_id = parts[0]
            view = parts[-1].replace('.jpg', '')
            
            #Try different pattern variations
            patterns = [
                f"{patient_id}_{view}.jpg",  
                f"{patient_id}_{view.upper()}.jpg",
                f"{patient_id}_{view.lower()}.jpg",  
                f"{patient_id}_R_{view.upper()}.jpg",  
                f"{patient_id}_L_{view.upper()}.jpg",  
                f"{patient_id}_R{view}.jpg",  
                f"{patient_id}_L{view}.jpg"  
            ]
            
            for pattern in patterns:
                if os.path.exists(os.path.join(class_dir, pattern)):
                    return os.path.join(class_dir, pattern)
        
        #Direct match with the given ImageName
        image_path = os.path.join(class_dir, image_name)
        if os.path.exists(image_path):
            return image_path
    
    #Try to find by PatientID and Side if available
    if 'PatientID' in row and not pd.isna(row['PatientID']) and 'Side' in row and not pd.isna(row['Side']):
        patient_id = str(row['PatientID'])
        side = row['Side'].strip().upper()[0]  #Get first letter (L or R)
        view = row['View'].strip().upper() if 'View' in row and not pd.isna(row['View']) else None
        
        #Check all image files in directory
        for filename in os.listdir(class_dir):
            if not filename.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            if filename.startswith(patient_id) and filename.lower().endswith('.jpg'):
                #Check if side matches
                if f"_{side}_" in filename or f"_{side}" in filename:
                    #Check if view matches
                    if view and (f"_{view}" in filename or f"_{view}.jpg" in filename):
                        return os.path.join(class_dir, filename)
                    #Return if only patient ID and side match
                    return os.path.join(class_dir, filename)
    
    #Find by PatientID only if specific match fails
    if 'PatientID' in row and not pd.isna(row['PatientID']):
        patient_id = str(row['PatientID'])
        for filename in os.listdir(class_dir):
            if filename.startswith(patient_id) and filename.endswith(('.jpg', '.png', '.jpeg')):
                return os.path.join(class_dir, filename)
    
    for filename in os.listdir(class_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            return os.path.join(class_dir, filename)
    
    print(f"No suitable image found in {class_dir}")
    return ""

#Main function to run the test workflow
def main():
    """
    Run the main testing workflow
    """
    print("Starting Breast Cancer LLM Ablation Test...")
    
    #Initialize mammogram analyzer with trained model
    analyzer = MammogramAnalyzer(model_path='models/best_model_1.pth')
    
    #Test 10 samples from each class using LLM analysis
    print("\nTesting 10 images from each class using LLM analysis...")
    results = test_selected_images(
        analyzer, 
        excel_path="AISSLab/data2.xlsx", 
        root_dir="AISSLab", 
        samples_per_class=10
    )
    
    print("\nAblation test completed!")

if __name__ == "__main__":
    main()
