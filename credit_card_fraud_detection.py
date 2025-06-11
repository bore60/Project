import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time

# === File Upload ===
st.title("💳 Credit Card Fraud Detection")
uploaded_file = st.file_uploader("Upload the creditcard.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # === Basic info ===
    st.write(f"✅ Loaded dataset with shape: {df.shape}")
    st.write(f"Fraud cases: {df[df['Class'] == 1].shape[0]}, Non-fraud cases: {df[df['Class'] == 0].shape[0]}")

    # === Visuals ===
    st.subheader("📊 Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Class", ax=ax1)
    st.pyplot(fig1)

    # === Scale features ===
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    # Reorder columns
    scaled_features = ["scaled_amount", "scaled_time"]
    remaining = [col for col in df.columns if col not in scaled_features + ["Class"]]
    df = df[scaled_features + remaining + ["Class"]]

    # === Split ===
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # === Clean & SMOTE ===
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    st.write(f"✅ After SMOTE: Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)}")

    # === Train ===
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    start_time = time.time()
    clf.fit(X_resampled, y_resampled)
    st.success(f"✅ Model trained in {time.time() - start_time:.2f} seconds.")

    # === Predictions ===
    y_pred = clf.predict(X_test)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1], ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # === Classification Report ===
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred, digits=4))

    # === ROC-AUC ===
    probas = clf.predict_proba(X_test)
    if probas.shape[1] == 2:
        auc = roc_auc_score(y_test, probas[:, 1])
        st.success(f"✅ ROC-AUC Score: {auc:.4f}")
    else:
        st.warning("⚠️ ROC-AUC not available: Only one class predicted.")

    # === Feature Importances ===
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df.head(15), x="Importance", y="Feature", ax=ax3)
    ax3.set_title("Top 15 Feature Importances")
    st.pyplot(fig3)

else:
    st.warning("📂 Please upload the `creditcard.csv` file to continue.")
