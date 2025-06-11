import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import zipfile
import os
import time

# Load dataset
# df = pd.read_csv("creditcard.csv")

# Download dataset (zipped)
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
# Extract zip file
zip_path = os.path.join(path, "creditcardfraud.zip")
extract_path = os.path.join(path, "extracted")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
# Load the dataset
csv_path = os.path.join(extract_path, "creditcard.csv")
df = pd.read_csv(csv_path)

print(f"Loaded dataset with shape: {df.shape}")
print(f"Dataset shape: {df.shape}")
print(f"Number of fraud cases: {df[df['Class'] == 1].shape[0]}")
print(f"Number of normal cases: {df[df['Class'] == 0].shape[0]}")

# === Descriptive Statistics === #
print("\nDescriptive Statistics:")
print(df.describe())

# === Class Distribution Visualization === #
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Class")
plt.title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
plt.show()

# === Correlation Matrix === #
plt.figure(figsize=(20, 12))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# === Scale 'Time' and 'Amount' === #
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
df.drop(["Time", "Amount"], axis=1, inplace=True)

# Reorder columns
scaled_features = ["scaled_amount", "scaled_time"]
remaining_features = [col for col in df.columns if col not in scaled_features + ["Class"]]
df = df[scaled_features + remaining_features + ["Class"]]

# === Train-test split === #
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Apply SMOTE === #
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
print(f"After SMOTE - Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)}")

# === Train Random Forest === #
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
start_time = time.time()
clf.fit(X_resampled, y_resampled)
end_time = time.time()
print(f"\n✅ Model trained in {end_time - start_time:.2f} seconds.")

# === Predictions === #
y_pred = clf.predict(X_test)

# === Confusion Matrix === #
print("Unique predictions:", np.unique(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === Classification Report === #
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# === ROC-AUC Score === #
probas = clf.predict_proba(X_test)
if probas.shape[1] == 2:
    auc_score = roc_auc_score(y_test, probas[:, 1])
    print(f"ROC-AUC Score: {auc_score:.4f}")
else:
    print("ROC-AUC Score: Model predicted only one class.")

# === Feature Importance Plot === #
importances = clf.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feat_df.head(15), x="Importance", y="Feature")
plt.title("Top 15 Feature Importances from Random Forest")
plt.tight_layout()
plt.show()
