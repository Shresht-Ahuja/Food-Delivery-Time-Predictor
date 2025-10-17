import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model("delivery_time_model.h5", compile=False)

drivers = pd.DataFrame({
    "DriverID": ["D1", "D2", "D3"],
    "Age": [25, 30, 22],
    "Rating": [4.5, 4.2, 4.8],
    "Distance": [5, 2, 7]
})

user_estimated_time = float(input("Enter your estimated delivery time (in minutes): "))

features = drivers[["Age", "Rating", "Distance"]].values
predicted_times = model.predict(features).flatten()

drivers["PredictedTime"] = predicted_times
drivers["Difference"] = abs(drivers["PredictedTime"] - user_estimated_time)

alpha = 0.6 

age_min, age_max = drivers['Age'].min(), drivers['Age'].max()
drivers['Age_Score'] = 1 - (drivers['Age'] - age_min) / (age_max - age_min)
diff_max = drivers['Difference'].max()
drivers['Diff_Score'] = 1 - drivers['Difference'] / diff_max

drivers['Final_Score'] = alpha * drivers['Age_Score'] + (1 - alpha) * drivers['Diff_Score']

best_driver = drivers.sort_values(by='Final_Score', ascending=False).iloc[0]

print(f"Recommended Driver: {best_driver['DriverID']}")
print(f"Predicted Delivery Time: {best_driver['PredictedTime']:.2f} minutes")
print(f"Difference from input estimate: {best_driver['Difference']:.2f} minutes")
print(f"Driver Age: {best_driver['Age']}")
