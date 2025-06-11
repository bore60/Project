# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time

# === Load dataset from Kaggle ===
file_path = "creditcard.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    file_path
)

print(f"✅ Loaded dataset with shape: {df.shape}")
print(f"Fraud cases: {df[df['Class'] == 1].shape[0]}, Non-fraud cases: {df[df['Class'] == 0].shape[0]}")

# === Descriptive statistics ===
print("\n🔍 Descriptive Statistics:\n", df.describe())

# === EDA: Class distribution ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Class")
plt.title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
plt.show()

# === EDA: Amount distribution ===
plt.figure(figsize=(8, 4))
sns.histplot(df["Amount"], bins=100, kde=True)
plt.title("Transaction Amount Distribution")
plt.show()

# === EDA: Time by Class ===
plt.figure(figsize=(10, 4))
sns.kdeplot(data=df, x="Time", hue="Class", fill=True, common_norm=False)
plt.title("Time of Transaction by Class")
plt.show()

# === Correlation matrix ===
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# === Feature scaling ===
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
df.drop(["Time", "Amount"], axis=1, inplace=True)

# Reorder columns
scaled_features = ["scaled_amount", "scaled_time"]
remaining = [col for col in df.columns if col not in scaled_features + ["Class"]]
df = df[scaled_features + remaining + ["Class"]]

# === Split data ===
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Check & clean before SMOTE ===
if X_train.isnull().values.any() or y_train.isnull().values.any():
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

# === Apply SMOTE ===
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)}")

# === Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
start_time = time.time()
clf.fit(X_resampled, y_resampled)
print(f"\n✅ Model trained in {time.time() - start_time:.2f} seconds.")

# === Predictions ===
y_pred = clf.predict(X_test)

# === Confusion Matrix ===
print("Unique predictions:", np.unique(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === Classification Report ===
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# === ROC-AUC Score ===
probas = clf.predict_proba(X_test)
if probas.shape[1] == 2:
    auc = roc_auc_score(y_test, probas[:, 1])
    print(f"✅ ROC-AUC Score: {auc:.4f}")
else:
    print("⚠️ ROC-AUC not available: Model predicted only one class.")

# === Feature Importances ===
importances = clf.feature_importances_
feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feat_df.head(15), x="Importance", y="Feature")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
