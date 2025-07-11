import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# 1. Create a small dataset
data = {
    'number_of_rooms': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'house_cost': [15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
}
df = pd.DataFrame(data)

# 2. Split the data
X = df[['number_of_rooms']]
y = df['house_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse:.2f}")

# 5. Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f) 