from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Set the template folder path correctly
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            rooms = int(request.form['rooms'])
            if rooms <= 0:
                raise ValueError('Number of rooms must be positive.')
            prediction = model.predict(np.array([[rooms]]))[0]
        except Exception as e:
            error = f"Invalid input: {e}"
    return render_template('index.html', prediction=prediction, error=error)

# For Vercel serverless
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 