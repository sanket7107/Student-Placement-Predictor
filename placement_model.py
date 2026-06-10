import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from placement_data import generate_synthetic_placement_data

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / 'artifacts'
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / 'placement_rf_model.joblib'
SCALER_PATH = ARTIFACT_DIR / 'placement_scaler.joblib'
ENCODERS_PATH = ARTIFACT_DIR / 'placement_encoders.joblib'

FEATURE_COLUMNS = [
    'CGPA',
    'MockPerformance',
    'AcademicPerformance',
    'Communication',
    'InternshipDone',
]
COMMUNICATION_OPTIONS = ['Excellent', 'Good', 'Average', 'Poor']
INTERNSHIP_OPTIONS = ['Yes', 'No']


def preprocess_data(df):
    df_processed = df.copy()

    le_communication = LabelEncoder()
    le_internship = LabelEncoder()
    le_placed = LabelEncoder()

    df_processed['Communication'] = le_communication.fit_transform(df_processed['Communication'])
    df_processed['InternshipDone'] = le_internship.fit_transform(df_processed['InternshipDone'])
    df_processed['Placed'] = le_placed.fit_transform(df_processed['Placed'])

    X = df_processed[FEATURE_COLUMNS]
    y = df_processed['Placed']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoders = {
        'communication': le_communication,
        'internship': le_internship,
        'placed': le_placed,
    }

    return X, y, scaler, encoders


def train_models(n_samples=200, random_state=42):
    df = generate_synthetic_placement_data(n_samples=n_samples, random_state=random_state)
    X, y, scaler, encoders = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
        'classification_report': classification_report(y_test, y_pred_rf, target_names=['Not Placed', 'Placed']),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
    }

    artifacts = {
        'model': rf_model,
        'scaler': scaler,
        'encoders': encoders,
        'metrics': metrics,
    }
    return artifacts


def save_artifacts(model, scaler, encoders):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)


def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not ENCODERS_PATH.exists():
        artifacts = train_models()
        save_artifacts(artifacts['model'], artifacts['scaler'], artifacts['encoders'])
        return artifacts['model'], artifacts['scaler'], artifacts['encoders']

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, encoders


def prepare_input(cgpa, mock_perf, academic_perf, communication, internship, scaler):
    comm_encoded = int(LabelEncoder().fit(COMMUNICATION_OPTIONS).transform([communication])[0])
    intern_encoded = int(LabelEncoder().fit(INTERNSHIP_OPTIONS).transform([internship])[0])
    features = np.array([[cgpa, mock_perf, academic_perf, comm_encoded, intern_encoded]])
    return features


def predict_placement(cgpa, mock_perf, academic_perf, communication, internship):
    model, scaler, encoders = load_artifacts()
    comm_encoded = encoders['communication'].transform([communication])[0]
    intern_encoded = encoders['internship'].transform([internship])[0]

    features = np.array([[cgpa, mock_perf, academic_perf, comm_encoded, intern_encoded]])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()
    placement_label = encoders['placed'].inverse_transform([prediction])[0]
    return placement_label, float(confidence)


def get_model_info():
    model, scaler, encoders = load_artifacts()
    return {
        'features': FEATURE_COLUMNS,
        'communication_options': COMMUNICATION_OPTIONS,
        'internship_options': INTERNSHIP_OPTIONS,
    }
