from http.server import BaseHTTPRequestHandler
import pickle
import numpy as np
import os
import json

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>House Cost Predictor</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
        <div class="container mt-5">
            <h2 class="mb-4">Predict House Cost by Number of Rooms</h2>
            <form method="post">
                <div class="mb-3">
                    <label for="rooms" class="form-label">Number of Rooms</label>
                    <input type="number" class="form-control" id="rooms" name="rooms" min="1" required>
                </div>
                <button type="submit" class="btn btn-primary">Predict</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
        document.querySelector('form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const rooms = document.getElementById('rooms').value;
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rooms: parseInt(rooms)})
            });
            const data = await response.json();
            document.getElementById('result').innerHTML = 
                '<div class="alert alert-success mt-4"><strong>Predicted House Cost:</strong> $' + data.prediction.toFixed(2) + '</div>';
        });
        </script>
        </body>
        </html>
        """
        self.wfile.write(html.encode())
    
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            try:
                rooms = int(data['rooms'])
                if rooms <= 0:
                    raise ValueError('Number of rooms must be positive.')
                prediction = model.predict(np.array([[rooms]]))[0]
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'prediction': float(prediction)}).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers() 