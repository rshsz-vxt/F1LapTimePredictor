import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# STEP 1: LOAD DATA & FILTER SPECIFIC RACE
# ==========================================
print("Loading data...")

# This magic line finds the EXACT folder where your f1.py file is saved
current_folder = os.path.dirname(os.path.abspath(__file__))

# Now we force Pandas to look inside that specific folder for the CSVs
laps = pd.read_csv(os.path.join(current_folder, 'lap_times.csv'))
pit_stops = pd.read_csv(os.path.join(current_folder, 'pit_stops.csv'))
results = pd.read_csv(os.path.join(current_folder, 'results.csv'))

# Select 2019 Italian GP (raceId = 1025)
TARGET_RACE_ID = 1025

# Get top 6 drivers who finished this race
race_results = results[(results['raceId'] == TARGET_RACE_ID) & (results['positionOrder'] <= 6)]
driver_ids = race_results['driverId'].tolist()

# Filter laps and pit stops for our specific race and drivers
race_laps = laps[(laps['raceId'] == TARGET_RACE_ID) & (laps['driverId'].isin(driver_ids))].copy()
race_pits = pit_stops[(pit_stops['raceId'] == TARGET_RACE_ID) & (pit_stops['driverId'].isin(driver_ids))].copy()

# ==========================================
# STEP 2: CALCULATE TIRE AGE & STINTS
# ==========================================
# Merge pit stop info with lap times
race_laps = pd.merge(race_laps, race_pits[['raceId', 'driverId', 'lap']], 
                     on=['raceId', 'driverId', 'lap'], 
                     how='left', indicator='is_pit')

# is_pit will be True if it was a pit stop lap
race_laps['is_pit'] = race_laps['is_pit'] == 'both'

# Calculate Stint number: increases by 1 every time a driver pits
race_laps['stint'] = race_laps.groupby('driverId')['is_pit'].cumsum() + 1

# Calculate Tire Age: cumulative count of laps within each stint
race_laps['tire_age'] = race_laps.groupby(['driverId', 'stint']).cumcount() + 1

# Convert lap time from milliseconds to seconds for easier reading
race_laps['lap_time_sec'] = race_laps['milliseconds'] / 1000

# ==========================================
# STEP 3: DATA CLEANING (Documented)
# ==========================================
initial_lap_count = len(race_laps)

# Remove lap 1 and pit stop laps. Cap lap time at 150s to avoid deleting valid data.
cleaned_laps = race_laps[(race_laps['lap'] > 1) & 
                         (race_laps['is_pit'] == False) & 
                         (race_laps['lap_time_sec'] < 150)].copy()

removed_laps = initial_lap_count - len(cleaned_laps)
print(f"\n--- CLEANING STEP ---")
print(f"Total laps initially: {initial_lap_count}")
print(f"Laps removed (Start, Pit, Traffic/VSC): {removed_laps}")
print(f"Laps remaining for modeling: {len(cleaned_laps)}")

# ==========================================
# STEP 4: STINT-BASED SPLIT
# ==========================================
# Train on Stint 1, Test on Stint 2
train_data = cleaned_laps[cleaned_laps['stint'] == 1].copy()
test_data = cleaned_laps[cleaned_laps['stint'] == 2].copy()

# Features for Model 1 (Baseline) - Just knows which driver it is
X_train_base = train_data[['driverId']]
X_test_base = test_data[['driverId']]

# Features for Model 2 (Tire Age Model) - Knows driver AND tire age
X_train_adv = train_data[['driverId', 'tire_age']]
X_test_adv = test_data[['driverId', 'tire_age']]

y_train = train_data['lap_time_sec']
y_test = test_data['lap_time_sec']

# ==========================================
# STEP 5: MODELING & COMPARISON
# ==========================================
# Train Baseline Model (Linear Regression without tire age)
baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train)
base_preds = baseline_model.predict(X_test_base)

# Train Tire Age Model (Random Forest captures degradation well)
tire_model = RandomForestRegressor(random_state=42, n_estimators=100)
tire_model.fit(X_train_adv, y_train)
tire_preds = tire_model.predict(X_test_adv)

# Calculate Metrics (RMSE & MAE)
print("\n--- MODEL COMPARISON ---")
print("Baseline Model (Ignores Tire Age):")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, base_preds)):.3f} seconds")
print(f"MAE:  {mean_absolute_error(y_test, base_preds):.3f} seconds")

print("\nTire-Age Model (Considers Tire Degradation):")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, tire_preds)):.3f} seconds")
print(f"MAE:  {mean_absolute_error(y_test, tire_preds):.3f} seconds")

# ==========================================
# STEP 6: PLOTTING (Predicted vs Actual)
# ==========================================
# USING driver_ids[1] SO YOUR GRAPH IS UNIQUE FROM YOUR PARTNER'S
plot_driver_id = driver_ids[1] 
driver_test_data = test_data[test_data['driverId'] == plot_driver_id].copy()

# Get predictions just for this driver
driver_test_data['pred_base'] = baseline_model.predict(driver_test_data[['driverId']])
driver_test_data['pred_tire'] = tire_model.predict(driver_test_data[['driverId', 'tire_age']])

# Sort by tire age so the line graph looks correct
driver_test_data = driver_test_data.sort_values('tire_age')

plt.figure(figsize=(10, 6))
plt.plot(driver_test_data['tire_age'], driver_test_data['lap_time_sec'], marker='o', label='Actual Lap Time', color='black')
plt.plot(driver_test_data['tire_age'], driver_test_data['pred_base'], linestyle='--', label='Baseline Model (No Tire Age)', color='red')
plt.plot(driver_test_data['tire_age'], driver_test_data['pred_tire'], marker='x', linestyle='-', label='Model with Tire Age', color='green')

plt.title(f'Tire Degradation Track: Actual vs Predicted (Driver ID {plot_driver_id}, Stint 2)')
plt.xlabel('Tire Age (Laps since pit stop)')
plt.ylabel('Lap Time (Seconds)')
plt.legend()
plt.grid(True)
plt.show()