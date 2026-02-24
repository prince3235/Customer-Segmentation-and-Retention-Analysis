import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os

def load_data(filepath):
    print(f"Loading data for modeling from {filepath}...")
    return pd.read_csv(filepath)

def prepare_data(df, churn_days_threshold=180):
    print(f"Defining Churn: Customers who haven't purchased in {churn_days_threshold} days.")
    df['Is_Churn'] = np.where(df['Recency'] > churn_days_threshold, 1, 0)
    
    print(f"Original Churn Distribution:\n{df['Is_Churn'].value_counts(normalize=True) * 100}")
    
    features = ['Frequency', 'Monetary'] 
    X = df[features]
    y = df['Is_Churn']
    
    return X, y, features

def train_evaluate_model(X, y, features, report_dir="../reports"):
    # 1. Split Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nOriginal Training data shape: {X_train.shape}")
    
    # 2. Apply SMOTE to balance the training data
    print("\nApplying SMOTE to balance the training classes...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"New Training data shape after SMOTE: {X_train_smote.shape}")
    
    # 3. Initialize and Train XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        eval_metric='logloss' # Avoids a common warning in newer XGBoost versions
    )
    xgb_model.fit(X_train_smote, y_train_smote)
    
    # 4. Predict on Test Data
    y_pred = xgb_model.predict(X_test)
    
    # 5. Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Optimized Model Performance (XGBoost + SMOTE) ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Confusion Matrix Visualization
    os.makedirs(report_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix - XGBoost Churn Prediction')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to: {cm_path}")
    plt.close()

    # 7. Feature Importance Visualization (Warning Fixed!)
    plt.figure(figsize=(8, 5))
    importances = xgb_model.feature_importances_
    sns.barplot(x=importances, y=features, hue=features, palette='viridis', legend=False)
    plt.title('Feature Importance for Churn Prediction (XGBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    
    fi_path = os.path.join(report_dir, "feature_importance.png")
    plt.savefig(fi_path)
    print(f"Feature Importance plot saved to: {fi_path}")
    plt.close()
    
    # NEW: Save the trained model for our Web App later
    import joblib
    model_path = "../reports/churn_xgb_model.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"Model saved to {model_path} for Web App deployment.")
    
    return xgb_model, accuracy

if __name__ == "__main__":
    INPUT_PATH = "../data/processed/rfm_clusters.csv"
    
    try:
        df = load_data(INPUT_PATH)
        X, y, feature_names = prepare_data(df, churn_days_threshold=180)
        model, acc = train_evaluate_model(X, y, feature_names)
        print(f"\nSUCCESS: Phase 4 Optimized! Update your resume with the new accuracy of {acc * 100:.2f}%.")
    except Exception as e:
        print(f"\nError: {e}")