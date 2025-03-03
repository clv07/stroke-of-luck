import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
data_dir = "C:\\Users\\gaogr\\school\\senior_spring\\UW-GEHC Collaboration\\PTBXL\\"
meas_file = os.path.join(data_dir, 'meas.csv')
statements_file = os.path.join(data_dir, '12slv24_stt.csv')

# Load the datasets
df_meas = pd.read_csv(meas_file)
df_statements = pd.read_csv(statements_file)

# Ensure 'TestID' column exists in both datasets for merging
if 'TestID' not in df_meas.columns or 'TestID' not in df_statements.columns:
    raise ValueError("Missing 'TestID' column in one of the datasets.")

# Merge datasets on TestID
df = df_meas.merge(df_statements[['TestID', 'MI_Phys']], on='TestID', how='inner')

# Drop non-numeric and non-relevant columns
df = df.select_dtypes(include=[np.number])

# Check for missing values and handle them (e.g., fill with median)
df.fillna(df.median(), inplace=True)

# Split into features and target
X = df.drop(columns=['MI_Phys'])
y = df['MI_Phys']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No MI", "MI"], yticklabels=["No MI", "MI"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
