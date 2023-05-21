import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import pickle

# Load the saved model
model = load_model('Model/model-v1.2')

# Load the fitted scaler and encoders
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder_light = pickle.load(open('encoder_light.pkl', 'rb'))
encoder_outlet = pickle.load(open('encoder_outlet.pkl', 'rb'))

# Load your test data
test_data = {
    'Indoor_Temperature': [20.0],
    'Outdoor_Temperature': [10.0],
    'Light_Level': [500]
}

df_test = pd.DataFrame(test_data)

# Normalize the test data using the same StandardScaler used in training
df_test = scaler.transform(df_test)

# Make predictions
predictions = model.predict(df_test)

# Print the shape of predictions
print('Shape of predictions:', np.shape(predictions))

# The model gives you the predicted light switch statuses, RGB LED values, and outlet statuses
predicted_light_status = predictions[0]
predicted_led_rgb = predictions[1]
predicted_outlet_status = predictions[2]

print('Shape of predicted_light_status:', np.shape(predicted_light_status))
print('Shape of predicted_led_rgb:', np.shape(predicted_led_rgb))
print('Shape of predicted_outlet_status:', np.shape(predicted_outlet_status))
