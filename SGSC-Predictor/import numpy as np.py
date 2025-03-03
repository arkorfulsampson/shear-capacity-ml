import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the original scaler
scaler = joblib.load('scaler.pkl')

# Save it again with an updated format
joblib.dump(scaler, 'scaler_fixed.pkl')
