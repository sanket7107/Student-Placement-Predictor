import numpy as np
import pandas as pd


def generate_synthetic_placement_data(n_samples=200, random_state=42):
    """Generate synthetic student placement data."""
    np.random.seed(random_state)

    data = {
        'CGPA': np.random.uniform(5.5, 10, n_samples),
        'MockPerformance': np.random.uniform(40, 100, n_samples),
        'AcademicPerformance': np.random.uniform(50, 95, n_samples),
        'Communication': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], n_samples),
        'InternshipDone': np.random.choice(['Yes', 'No'], n_samples),
    }

    df = pd.DataFrame(data)

    # Calculate placement probability from feature signals
    cgpa_score = (df['CGPA'] - 5.5) / (10 - 5.5)
    mock_score = (df['MockPerformance'] - 40) / (100 - 40)
    academic_score = (df['AcademicPerformance'] - 50) / (95 - 50)
    communication_score = df['Communication'].map(
        {'Excellent': 0.9, 'Good': 0.7, 'Average': 0.4, 'Poor': 0.1}
    )
    internship_score = df['InternshipDone'].map({'Yes': 0.8, 'No': 0.3})

    placement_probability = (
        cgpa_score * 0.25
        + mock_score * 0.2
        + academic_score * 0.2
        + communication_score * 0.2
        + internship_score * 0.15
    )

    df['Placed'] = np.where(placement_probability > 0.5, 'Yes', 'No')
    return df
