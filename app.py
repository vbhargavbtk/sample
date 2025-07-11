from flask import Flask, render_template, request
import pickle
import numpy as np
import os

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

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 