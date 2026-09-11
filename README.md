# F1LapTimePredictor
A Machine Learning project predicting F1 lap times by analyzing tire degradation.
# F1 Lap Time Predictor & Tire Degradation Analysis 🏎️

## 🛠️ What we actually did
* **Built custom features:** The raw dataset didn't have a "Tire Age" column, so we had to code it ourselves by combining lap times with pit-stop data.
* **No random splits:** Instead of a basic `train_test_split`, we trained the models on Stint 1 and tested them on Stint 2. This stops data leakage and makes it act like a real live-race forecast.
* **Cleaned up the mess:** Real racing data is super noisy. We filtered out Lap 1 (standing starts), pit-lane laps, and crazy slow laps caused by traffic or VSCs.
* **Model showdown:** We compared a basic Linear Regression against a Random Forest Regressor using RMSE and MAE to see which one caught the degradation curve better.

## 📊 What we found
* **The Baseline Model:** Completely ignores tire wear. It just plays it safe and predicts a flat, average speed for the whole race.
* **The Random Forest Model:** This one actually got it! It successfully tracked the non-linear curve, showing exactly how lap times drop as the tires get older.

*(Note: We trained the models on the top 6 finishers, and plotted the final graphs for specific driver stints to make the degradation super clear.)*

## 💻 Tech Stack
Python, Pandas, Scikit-Learn, and Matplotlib.

## 📁 Dataset
We used the awesome [F1 World Championship dataset on Kaggle](https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020).
