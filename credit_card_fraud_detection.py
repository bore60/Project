# =============================================
# 💳 Credit Card Fraud Detection Web App
# 🧑‍💻 Owner: Sylvia Chelangat Bore
# =============================================

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

# === Streamlit App Header === #
st.title("💳 Credit Card Fraud Detection \n Sylvia Chelangat Bore")
st.markdown("**App by Sylvia Chelangat Bore**")
st.markdown("Upload the **creditcard.csv** file to begin analysis.")

# === File Uploader === #
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded! Shape: {df.shape}")

    st.write(f"Number of fraud cases: {df[df['Class'] == 1].shape[0]}")
    st.write(f"Number of normal cases: {df[df['Class'] == 0].shape[0]}")

    # === Descriptive Statistics === #
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(df.describe())

    # === Class Distribution Visualization === #
    st.subheader("📌 Class Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Class", ax=ax1)
    ax1.set_title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
    st.pyplot(fig1)

    # === Correlation Matrix === #
    st.subheader("🔗 Correlation Matrix")
    fig2, ax2 = plt.subplots(figsize=(20, 12))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax2)
    ax2.set_title("Correlation Matrix")
    st.pyplot(fig2)

    # === Feature Scaling === #
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    # Reorder columns
    scaled_features = ["scaled_amount", "scaled_time"]
    remaining_features = [col for col in df.columns if col not in scaled_features + ["Class"]]
    df = df[scaled_features + remaining_features + ["Class"]]

    # Clean any missing or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # === Train-test split === #
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Make sure all features are numeric
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        st.error(f"❌ Non-numeric columns found: {list(non_numeric_cols)}")
        X = X.select_dtypes(include=[np.number])
        st.warning("⚠️ Non-numeric columns removed before resampling.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Drop any remaining NaNs or infs
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.dropna(inplace=True)
    y_train = y_train.loc[X_train.index]  # Realign y with cleaned X

    # === Apply SMOTE === #
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    st.success(f"✅ After SMOTE - Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)}")

    # === Train Random Forest === #
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    start_time = time.time()
    clf.fit(X_resampled, y_resampled)
    end_time = time.time()
    st.success(f"✅ Model trained in {end_time - start_time:.2f} seconds.")

    # === Predictions === #
    y_pred = clf.predict(X_test)

    # === Confusion Matrix === #
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1], ax=ax3)
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

    # === Classification Report === #
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred, digits=4))

    # === ROC-AUC Score === #
    probas = clf.predict_proba(X_test)
    if probas.shape[1] == 2:
        auc_score = roc_auc_score(y_test, probas[:, 1])
        st.success(f"✅ ROC-AUC Score: {auc_score:.4f}")
    else:
        st.warning("⚠️ ROC-AUC Score not available: Model predicted only one class.")

    # === Feature Importance Plot === #
    importances = clf.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=feat_df.head(15), x="Importance", y="Feature", ax=ax4)
    ax4.set_title("Top 15 Feature Importances from Random Forest")
    st.pyplot(fig4)

else:
    st.info("👆 Please upload the `creditcard.csv` file to start.")
