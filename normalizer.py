import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('smart_home_data.csv')

# Select features
features = df[['Indoor_Temperature', 'Outdoor_Temperature', 'Light_Level']]

# Light switches
light_status = df[['Light1_Status', 'Light2_Status']]
encoder_light = LabelEncoder()
light_status = light_status.apply(encoder_light.fit_transform)

# Outlets
outlet_status = df[['Outlet1_Status', 'Outlet2_Status']]
encoder_outlet = LabelEncoder()
outlet_status = outlet_status.apply(encoder_outlet.fit_transform)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features
features = scaler.fit_transform(features)

# Save the fitted scaler and encoders for later use
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(encoder_light, open('encoder_light.pkl', 'wb'))
pickle.dump(encoder_outlet, open('encoder_outlet.pkl', 'wb'))
