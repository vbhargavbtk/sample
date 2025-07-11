from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
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

# For local development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 