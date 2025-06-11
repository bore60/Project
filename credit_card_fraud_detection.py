import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

import time

# === Streamlit App Header === #
st.title("💳 Credit Card Fraud Detection")
st.markdown("**👩‍💻 Created by Sylvia Chelangat Bore**")
st.markdown("Upload the **creditcard.csv** file to begin analysis.")

# === File Uploader === #
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded! Shape: {df.shape}")

    st.write(f"Number of fraud cases: {df[df['Class'] == 1].shape[0]}")
    st.write(f"Number of normal cases: {df[df['Class'] == 0].shape[0]}")

    # === Data Quality Check === #
    st.subheader("🔍 Data Quality Check")
    st.write(f"Missing values: {df.isnull().sum().sum()}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        st.write(f"Infinite values: {inf_count}")

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

    # === Train-test split === #
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === Enhanced SMOTE Implementation === #
    st.subheader("⚖️ Applying SMOTE for Class Balance")
    
    # Step 1: Data cleaning and preparation
    st.write("🔄 Preparing data for SMOTE...")
    
    # Create copies to avoid modifying original data
    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()
    
    # Handle any potential infinite values
    X_train_processed = X_train_processed.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values if any exist
    if X_train_processed.isnull().sum().sum() > 0:
        st.warning("⚠️ Found missing values, filling with median...")
        imputer = SimpleImputer(strategy='median')
        X_train_processed = pd.DataFrame(
            imputer.fit_transform(X_train_processed),
            columns=X_train_processed.columns,
            index=X_train_processed.index
        )
    
    # Ensure target variable is integer
    y_train_processed = y_train_processed.astype(int)
    
    # Step 2: Convert to numpy arrays with explicit data types
    X_train_array = X_train_processed.values.astype(np.float64)
    y_train_array = y_train_processed.values.astype(np.int32)
    
    # Step 3: Data validation
    st.write(f"✅ Data shape: {X_train_array.shape}")
    st.write(f"✅ Data types - X: {X_train_array.dtype}, y: {y_train_array.dtype}")
    
    nan_count = np.isnan(X_train_array).sum()
    inf_count = np.isinf(X_train_array).sum()
    
    if nan_count > 0:
        st.error(f"❌ Found {nan_count} NaN values in training data")
    if inf_count > 0:
        st.error(f"❌ Found {inf_count} infinite values in training data")
    
    # Step 4: Apply SMOTE with error handling
    try:
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_train_array, y_train_array)
        st.success(f"✅ SMOTE completed successfully!")
        st.success(f"After SMOTE - Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)}")
        
        # Convert back to DataFrame for consistency (optional)
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        
    except Exception as e:
        st.error(f"❌ SMOTE failed: {str(e)}")
        st.info("🔄 Proceeding without SMOTE...")
        X_resampled, y_resampled = X_train_array, y_train_array

    # === Train Random Forest === #
    st.subheader("🌲 Training Random Forest Model")
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    start_time = time.time()
    clf.fit(X_resampled, y_resampled)
    end_time = time.time()
    st.success(f"✅ Model trained in {end_time - start_time:.2f} seconds.")

    # === Predictions === #
    # Ensure test data has same preprocessing as training data
    X_test_processed = X_test.copy()
    X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan)
    
    # Fill any missing values in test set using same strategy
    if X_test_processed.isnull().sum().sum() > 0:
        if 'imputer' in locals():
            X_test_processed = pd.DataFrame(
                imputer.transform(X_test_processed),
                columns=X_test_processed.columns,
                index=X_test_processed.index
            )
        else:
            X_test_processed = X_test_processed.fillna(X_test_processed.median())
    
    # Convert test data to same format
    X_test_array = X_test_processed.values.astype(np.float64)
    
    # Make predictions
    y_pred = clf.predict(X_test_array)

    # === Model Evaluation === #
    st.subheader("📊 Model Performance")

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
    try:
        probas = clf.predict_proba(X_test_array)
        if probas.shape[1] == 2:
            auc_score = roc_auc_score(y_test, probas[:, 1])
            st.success(f"✅ ROC-AUC Score: {auc_score:.4f}")
        else:
            st.warning("⚠️ ROC-AUC Score not available: Model predicted only one class.")
    except Exception as e:
        st.warning(f"⚠️ Could not calculate ROC-AUC: {str(e)}")

    # === Feature Importance Plot === #
    st.subheader("🎯 Feature Importance")
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
