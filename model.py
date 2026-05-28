import os
import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# ==============================
# CREATE CACHE FOLDER
# ==============================

os.makedirs("cache", exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache("cache")

# ==============================
# RACES TO LOAD
# ==============================

races = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Japan",
    "China",
    "Miami",
    "Monaco",
    "Canada",
    "Silverstone",
    "Belgium",
    "Monza",
    "Singapore",
    "Austin",
    "Mexico",
    "Las Vegas",
    "Abu Dhabi"
]

# ==============================
# LOAD ALL RACE DATA
# ==============================

all_data = []

for race in races:

    print(f"\nLoading {race} GP data...")

    try:
        # Load race session
        session = fastf1.get_session(2024, race, "R")
        session.load()

        # Get race results
        results = session.results

        # Create dataframe
        race_data = pd.DataFrame({
            'Driver': results['Abbreviation'],
            'Team': results['TeamName'],
            'GridPosition': results['GridPosition'],
            'FinalPosition': results['Position']
        })

        # Add circuit name
        race_data['Circuit'] = race

        # Winner column
        race_data['Winner'] = race_data['FinalPosition'].apply(
            lambda x: 1 if x == 1 else 0
        )

        # Append data
        all_data.append(race_data)

        print(f"{race} loaded successfully!")

    except Exception as e:
        print(f"Error loading {race}: {e}")

# ==============================
# COMBINE DATA
# ==============================

data = pd.concat(all_data, ignore_index=True)

# Remove missing values
data = data.dropna()

print("\nDataset Preview:")
print(data.head())

# Save CSV
data.to_csv("f1_data.csv", index=False)

print("\nCSV dataset saved!")

# ==============================
# ENCODE CATEGORICAL DATA
# ==============================

driver_encoder = LabelEncoder()
team_encoder = LabelEncoder()
circuit_encoder = LabelEncoder()

data['DriverEncoded'] = driver_encoder.fit_transform(data['Driver'])
data['TeamEncoded'] = team_encoder.fit_transform(data['Team'])
data['CircuitEncoded'] = circuit_encoder.fit_transform(data['Circuit'])

# ==============================
# FEATURES & TARGET
# ==============================

X = data[[
    'DriverEncoded',
    'TeamEncoded',
    'GridPosition',
    'CircuitEncoded'
]]

Y = data['Winner']

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ==============================
# TRAIN MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, Y_train)

# ==============================
# PREDICTIONS
# ==============================

predictions = model.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ==============================
# SAVE MODEL & ENCODERS
# ==============================

pickle.dump(model, open("f1_model.pkl", "wb"))

pickle.dump(driver_encoder, open("driver_encoder.pkl", "wb"))
pickle.dump(team_encoder, open("team_encoder.pkl", "wb"))
pickle.dump(circuit_encoder, open("circuit_encoder.pkl", "wb"))

print("\nF1 AI Model Saved Successfully!")