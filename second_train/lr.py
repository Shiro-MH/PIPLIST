import pandas as pd
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from math import sqrt
# "improve"
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

tmp = pd.read_excel("SUM-CrCoNi.xlsx")
print(tmp)

# "improve" Create a pipeline with polynomial features and linear regression
degree = 2  # Degree of polynomial features
pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=degree)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])

# 將資料分為特徵 (X) 和目標 (y)
X = tmp[["x", "y", "z"]]
y = tmp["stress"]

# 建立一個線性迴歸的模型
model = LinearRegression()

# 用交叉驗證來評估模型
#scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
#print("Cross-validated scores:", scores)

# 將資料分為訓練集和測試集， -2 = 用最後 2 個資料作為測試集，其餘作為訓練集
X_train = X[:-2]
X_test = X[-2:]
y_train = y[:-2]
y_test = y[-2:]

# 用訓練集來訓練模型
#model.fit(X_train, y_train)

# 用測試集來預測目標
#y_pred = model.predict(X_test)

# "improve" Train the model using the pipeline
pipeline.fit(X_train, y_train)

# "improve" Predict using the pipeline
y_pred_pipeline = pipeline.predict(X_test)

# Calculate model accuracy metrics for the new model
mse_pipeline = mean_squared_error(y_test, y_pred_pipeline)
r2_pipeline = r2_score(y_test, y_pred_pipeline)
mae_pipeline = mean_absolute_error(y_test, y_pred_pipeline)
rmse_pipeline = sqrt(mse_pipeline)
evs_pipeline = explained_variance_score(y_test, y_pred_pipeline)

# Preparing the improved model output
output = {
    "predict": y_pred_pipeline[0],
    "actual": y_test.iloc[0],
    "mse": mse_pipeline,
    "r2": r2_pipeline,
    "mae": mae_pipeline,
    "rmse": rmse_pipeline,
    "evs": evs_pipeline
}

print(output)