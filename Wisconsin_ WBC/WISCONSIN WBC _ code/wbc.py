import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
# Assuming the WBC dataset is in CSV format and named 'wbc.data'
wbc_data = pd.read_csv('wbc.data', header=None)

# Add appropriate column names (as per the WBC dataset description)
wbc_data.columns = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 
                     'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 
                     'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 
                     'Normal_Nucleoli', 'Mitoses', 'Class']

# Replace missing values (if any) indicated by '?'
wbc_data.replace('?', pd.NA, inplace=True)
wbc_data.dropna(inplace=True)

# Convert Bare_Nuclei to numeric
wbc_data['Bare_Nuclei'] = wbc_data['Bare_Nuclei'].astype(int)

# Features and target
X = wbc_data.drop(['Sample_code_number', 'Class'], axis=1)
y = wbc_data['Class'].apply(lambda x: 1 if x == 4 else 0)  # 4 = malignant (1), 2 = benign (0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
