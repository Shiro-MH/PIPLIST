import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR #支持向量回歸
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import numpy as np


tmp = pd.read_excel("SUM-CrCoNi.xlsx")

# 將資料分為特徵 (X) 和目標 (y)
X = tmp[["x", "y", "z"]]
y = tmp["stress"]

# 將資料分為訓練集和測試集，這裡我們用最後兩個資料作為測試集，其餘作為訓練集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立一個支持向量迴歸 (SVR) 的模型，這裡我們使用預設的參數，您可以根據您的需求來調整它們
model = SVR()

# 用訓練集來訓練模型
model.fit(X_train, y_train)

# 用測試集來預測目標
y_pred = model.predict(X_test)

# Calculating the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
evs = explained_variance_score(y_test, y_pred)

# Printing the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Explained Variance Score (EVS):", evs)

# Displaying predictions and actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())  # Print the first few actual and predicted values