import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier

# Load the dataset
column_names = [
    'ID', 'Diagnosis', 'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean',
    'Compactness_mean', 'Concavity_mean', 'Concave_points_mean', 'Symmetry_mean', 'Fractal_dimension_mean',
    'Radius_se', 'Texture_se', 'Perimeter_se', 'Area_se', 'Smoothness_se', 'Compactness_se', 'Concavity_se',
    'Concave_points_se', 'Symmetry_se', 'Fractal_dimension_se', 'Radius_worst', 'Texture_worst',
    'Perimeter_worst', 'Area_worst', 'Smoothness_worst', 'Compactness_worst', 'Concavity_worst',
    'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst'
]

data = pd.read_csv('wdbc.data', header=None, names=column_names)

# Drop the ID column (not useful for model training)
data.drop(columns=['ID'], inplace=True)

# Encode the Diagnosis column (M = 1, B = 0)
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Check class distribution
class_counts = data['Diagnosis'].value_counts()
majority_class = data[data['Diagnosis'] == class_counts.idxmax()]
minority_class = data[data['Diagnosis'] == class_counts.idxmin()]

# Balance the dataset using oversampling
minority_upsampled = resample(minority_class, 
                              replace=True,  # Sample with replacement
                              n_samples=len(majority_class),  # Match majority class size
                              random_state=42)  # Reproducibility

# Combine the majority and upsampled minority class
balanced_data = pd.concat([majority_class, minority_upsampled])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X = balanced_data.drop(['Diagnosis'], axis=1)
y = balanced_data['Diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression model
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# MLP Neural Network model
mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

# Evaluation
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))
print("Classification Report:")
print(classification_report(y_test, y_pred_logistic))

print("\nMLP Neural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp))
print("Classification Report:")
print(classification_report(y_test, y_pred_mlp))
