from flask import Flask, render_template, request, jsonify

from placement_model import get_model_info, predict_placement

app = Flask(__name__)

model_info = get_model_info()


@app.route('/')
def index():
    return render_template(
        'index.html',
        communication_options=model_info['communication_options'],
        internship_options=model_info['internship_options'],
    )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        cgpa = float(data.get('cgpa', 0))
        mock_perf = float(data.get('mock_performance', 0))
        academic_perf = float(data.get('academic_performance', 0))
        communication = data.get('communication', 'Good')
        internship = data.get('internship', 'No')

        prediction, confidence = predict_placement(
            cgpa, mock_perf, academic_perf, communication, internship
        )

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'confidence': f'{confidence:.2%}',
        })
    except Exception as error:
        return jsonify({
            'status': 'error',
            'message': str(error),
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
