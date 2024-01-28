import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR #支持向量回歸
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

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

# Creating the SVR model
svr_model = SVR()

# Parameters for GridSearchCV
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'poly'],
    'degree': [2, 3,]
}

# Applying GridSearchCV
grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Best model after grid search
best_model = grid_search.best_estimator_

# Generating learning curve data
train_sizes, train_scores, validation_scores = learning_curve(
    best_model, X_train_scaled, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='neg_mean_squared_error')

# Calculating mean and standard deviation of training and validation scores
train_scores_mean = -train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)
validation_scores_std = validation_scores.std(axis=1)

#Plotting the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score (Negative MSE)")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# # 特徵分布圖
# plt.figure(figsize=(12, 4))
# for i, col in enumerate(['x', 'y', 'z']):
#     plt.subplot(1, 3, i+1)
#     sns.histplot(X[col], kde=True)
#     plt.title(f'Distribution of {col}')
# plt.tight_layout()
# plt.show()

# # 特徵與目標變量關係圖
# plt.figure(figsize=(12, 4))
# for i, col in enumerate(['x', 'y', 'z']):
#     plt.subplot(1, 3, i+1)
#     sns.scatterplot(x=X[col], y=y)
#     plt.title(f'Relationship between {col} and Stress')
# plt.tight_layout()
# plt.show()

# 實際值與預測值比較圖
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs Predicted Values')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  # Diagonal line
# plt.grid(True)
# plt.show()

# # 誤差分布圖
# errors = y_test - y_pred
# plt.figure(figsize=(8, 6))
# sns.histplot(errors, kde=True)
# plt.title('Distribution of Prediction Errors')
# plt.xlabel('Error')
# plt.ylabel('Frequency')
# plt.show()

# Predicting using the model
y_pred = best_model.predict(X_test_scaled)

# Calculating the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
evs = explained_variance_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Explained Variance Score (EVS):", evs)

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())  # Print the first few actual and predicted values