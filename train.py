import joblib, os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

# Load and split data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save artifact
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/regression_model.pkl')
print("Model trained and saved.")
