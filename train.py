import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load dataset
data = pd.read_csv("data/heatwave_data.csv")

# Separate features and target
X = data[
    [
        "max_temp",
        "min_temp",
        "humidity",
        "wind_speed",
        "previous_temp"
    ]
]

y = data["heatwave"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/heatwave_model.pkl")

print("Model trained successfully.")
print("Model saved as model/heatwave_model.pkl")