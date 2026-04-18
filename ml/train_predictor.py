import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model(data_path, model_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Map categorical variables
    binary_columns = ['ExtracurricularActivities', 'PlacementTraining']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        
    df['PlacementStatus'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})
    
    # Drop StudentID
    df = df.drop('StudentID', axis=1)
    
    # Separate features and target
    X = df.drop('PlacementStatus', axis=1)
    y = df['PlacementStatus']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest Classifier (Optimized for Confidence)...")
    model = RandomForestClassifier(
        n_estimators=100,        # Slightly fewer trees
        max_depth=6,             # Severely reduced depth to prevent overfitting
        min_samples_split=15,    # Require more samples to split a node
        min_samples_leaf=10,     # Require more samples at leaf nodes
        class_weight='balanced', # Restore 'balanced' class weights for fairness
        oob_score=True,          # Use out-of-bag scoring
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    train_acc = model.score(X_train_scaled, y_train) * 100
    test_acc = model.score(X_test_scaled, y_test) * 100
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Accuracy: {test_acc:.2f}%")
    
    # Save model and scaler
    print(f"Saving model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': list(X.columns)}, f)
    print("Optimization Complete.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_file = os.path.join(project_root, 'placementdata.csv')
    model_file = os.path.join(current_dir, 'model.pkl')
    
    train_and_save_model(data_file, model_file)
