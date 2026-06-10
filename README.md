# Student Placement Prediction ML Model

A machine learning project that predicts whether a student will be placed or not using key features such as CGPA, mock performance, academic performance, communication skills, and internship experience.


## 📁 Project Structure

```
Student_Placement_Pridiction/
├── app.py                               # Flask web application entry point
├── placement_data.py                    # Synthetic dataset generation
├── placement_model.py                   # Model training, saving, and prediction utilities
├── train_model.py                       # Train and save the model artifacts
├── artifacts/                           # Saved model, scaler, and encoders
├── templates/ 
│   └── index.html                       # Web page template
├── static/
│   └── style.css                        # Web page styling
├── student_placement_prediction.ipynb   # Original notebook
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7 or higher
- `pip` package manager

### Install dependencies

```bash
cd "c:\Users\sanke\Desktop\ML Models\Student_Placement_Pridiction"
pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

This creates the `artifacts/` folder and saves the trained Random Forest model, scaler, and label encoders.

### Run the web app

```bash
python app.py
```

Then open the browser at `http://127.0.0.1:5000` to use the prediction page.

## 🌐 Web Interface

The Flask app provides a user-friendly web page where you can enter:

- CGPA
- Mock Performance
- Academic Performance
- Communication skill level
- Internship status

Then the app returns whether the student is predicted to be placed and the confidence score.

## 🧠 Model Pipeline

1. Generate a synthetic dataset for student placement
2. Preprocess categorical fields
3. Train a Random Forest classifier
4. Serialize the model artifacts
5. Use the web interface to make predictions

## 🧪 Useful Commands

- `python train_model.py` — train and save the model artifacts
- `python app.py` — start the Flask web server

## 📌 Notes

- The web app is built using Flask and a simple HTML/CSS form.
- The model uses a synthetic dataset for demonstration purposes.
- You can replace `placement_data.py` with your own dataset to make predictions on real data.
