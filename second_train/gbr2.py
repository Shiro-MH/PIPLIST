import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor  #梯度提升機
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import numpy as np


data = pd.read_excel("SUM-CrCoNi.xlsx")

# Preparing the data
X = data[['x', 'y', 'z']]  # Input features
y = data['stress']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating the GradientBoostingRegressor model
gbm_model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [300, 500, 700],  # Increased number of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Fine-tuning learning rate
    'max_depth': [3, 5, 7],  # Adjusting tree depth
    'min_samples_split': [4, 6],  # Adjusting minimum samples for a split
    'min_samples_leaf': [2, 3]  # Adjusting minimum samples in a leaf
}

# Applying GridSearchCV
grid_search = GridSearchCV(gbm_model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
# Predicting using the model
y_pred = best_model.predict(X_test_scaled)

# Calculating the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

# Printing the evaluation metrics
print("Best Parameters:", grid_search.best_params_)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)
print("Explained Variance Score (EVS):", evs)

# Displaying predictions and actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())
