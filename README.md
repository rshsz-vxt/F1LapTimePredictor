# F1 Lap Time Predictor & Tire Degradation Analysis 🏎️

This data science project models physical tire wear in Formula 1 racing. Using real-world race telemetry from the 2019 Italian GP (a dry race with no red flags), this project engineers a custom 'Tire Age' feature to track lap time performance. 

## 🛠️ Methodology & Data Cleaning
Real racing data is highly noisy. To ensure model accuracy and avoid data leakage, the following strict parameters were applied:
* **Feature Engineering:** Calculated dynamic 'Stints' and 'Tire Age' (laps elapsed since last pit stop) from raw telemetry.
* **Outlier Removal:** Filtered out standing starts, pit-stop laps, and the immediate out-laps. Furthermore, any lap slower than 1.5× the driver's median lap time was removed to account for VSC/traffic anomalies.
* **Chronological Splitting:** Avoided random train-test splitting (which causes data leakage). Models were trained on all early stints and tested strictly on each driver's **final stint**.

## 📊 Model Comparison Table
The baseline Linear Regression (trained purely on grid position and lap number) was compared against a Random Forest Regressor enhanced with our engineered `tire_age` feature.

| Model | Features Used | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| **Baseline (Linear)** | Grid Position, Lap Number | 1.095s | 0.902s |
| **Enhanced (Random Forest)** | Grid Position, Lap Number, Tire Age | 0.732s | 0.584s |
*(Note: Replace the numbers above with the exact outputs from your VS Code terminal).*

## 📈 Visualization
![Final Stint Visualization](stint_plot.png)
The baseline model fails to account for tire wear, predicting a relatively flat pace. The Random Forest model successfully captures the physical degradation curve.

## 💻 Tech Stack
Python, Pandas, Scikit-Learn, and Matplotlib.

## 📁 Dataset
We used the awesome [F1 World Championship dataset on Kaggle](https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020).
