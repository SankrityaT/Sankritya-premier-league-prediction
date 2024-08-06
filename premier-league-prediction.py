import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the CSV files
historical_data = pd.read_csv('premier-league-matches.csv')
fixtures_data = pd.read_csv('epl-fixtures-2025.csv')

# Filter relevant teams for the upcoming season
teams = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brighton", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Ipswich Town", "Leicester City", "Liverpool", "Manchester City",
    "Manchester Utd", "Newcastle", "Nottingham", "Southampton", "Tottenham", "West Ham", "Wolves"
]

# Filter historical data for relevant teams
historical_data = historical_data[historical_data['Home'].isin(teams) & historical_data['Away'].isin(teams)]

# One-hot encode the categorical features
historical_data = pd.get_dummies(historical_data, columns=['Home', 'Away'])

# Extract features and target variable
X = historical_data.drop(columns=['FTR', 'Date'])
y = historical_data['FTR']

# Convert target variable to binary: Home Win (H) vs Not Home Win (NH)
y = y.apply(lambda x: 1 if x == 'H' else 0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
logistic_model = LogisticRegression(max_iter=1000)
decision_tree_model = DecisionTreeClassifier()

# Train the models
logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)

# Make predictions
y_pred_logistic = logistic_model.predict(X_test)
y_pred_tree = decision_tree_model.predict(X_test)

# Calculate accuracy
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
tree_accuracy = accuracy_score(y_test, y_pred_tree)

print(f"Revised Logistic Regression Model Test Accuracy: {logistic_accuracy}")
print(f"Revised Decision Tree Model Test Accuracy: {tree_accuracy}")

# User input for prediction
home_team = input("Enter the home team: ")
away_team = input("Enter the away team: ")

# Create a new match dataframe for prediction
new_match = pd.DataFrame(columns=X.columns)
new_match.loc[0] = 0
new_match[f'Home_{home_team}'] = 1
new_match[f'Away_{away_team}'] = 1

# Make a prediction
logistic_pred = logistic_model.predict(new_match)[0]
tree_pred = decision_tree_model.predict(new_match)[0]

# Predict probabilities
logistic_prob = logistic_model.predict_proba(new_match)[0]
tree_prob = decision_tree_model.predict_proba(new_match)[0]

# Output prediction results
result = "Home Win" if logistic_pred == 1 else "Not Home Win (Draw or Away Win)"
print(f"The predicted outcome for {home_team} (home) vs {away_team} (away) is: {result}")
print(f"Win Probabilities (Logistic Regression):")
print(f"{home_team} (Home Win): {logistic_prob[1] * 100:.2f}%")
print(f"Draw or {away_team} (Away Win): {logistic_prob[0] * 100:.2f}%")
