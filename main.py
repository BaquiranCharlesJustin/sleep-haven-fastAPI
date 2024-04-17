from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = FastAPI()

# Load LabelEncoder
le = joblib.load("label_encoder.pkl")

# Load the model and scaler
model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define BPM ranges
bpm_ranges = [(50, 60, 'Asleep'), (61, 90.9, 'Awake')]

# Maintain a history of states
state_history = []

# Define maximum history length
MAX_HISTORY_LENGTH = 5

# Define minimum confidence threshold for prediction
MIN_CONFIDENCE_THRESHOLD = 0.3

class Item(BaseModel):
    BPM: float

def predict_state(bpm):
    # Check BPM categorization
    state = None
    for lower, upper, label in bpm_ranges:
        if lower <= bpm <= upper:
            state = label
            break
    if state is None:
        raise HTTPException(status_code=400, detail="BPM value out of range")
    return state

def predict_with_threshold(prediction_proba):
    if prediction_proba[1] > MIN_CONFIDENCE_THRESHOLD:
        return 1
    else:
        return 0

@app.post("/predict/")
def predict(item: Item):
    global state_history

    # Preprocess the input data
    bpm = item.BPM

    # Predict state using a simple threshold
    state = predict_state(bpm)

    # Make prediction
    normalized_bpm = scaler.transform([[bpm]])[0][0]
    prediction_proba = model.predict_proba([[normalized_bpm]])[0]
    prediction = predict_with_threshold(prediction_proba)

    # Add predicted state to history
    state_history.append(prediction)

    # Keep history length within maximum limit
    if len(state_history) > MAX_HISTORY_LENGTH:
        state_history = state_history[-MAX_HISTORY_LENGTH:]

    # Determine the majority state in history
    majority_state = max(set(state_history), key=state_history.count)

    # Print the predicted state
    print(f"Predicted state: {majority_state}")

    # Return the majority state
    return {"BPM": bpm, "predicted_state": majority_state}
