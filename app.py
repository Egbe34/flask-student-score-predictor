from flask import Flask, request, render_template
import joblib
import numpy as np

# Load model
model = joblib.load('student_score_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    prediction = model.predict(np.array([[hours]]))
    return render_template('index.html', prediction_text=f'Predicted Score: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
