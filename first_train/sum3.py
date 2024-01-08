import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


tmp = pd.read_excel("CrCoNi.xlsx", sheet_name="SUM")
print(tmp)

# 將資料分為特徵 (X) 和目標 (y)
X = tmp[["x", "y", "z"]]
y = tmp["stress"]

# 將資料分為訓練集和測試集，這裡我們用最後兩個資料作為測試集，其餘作為訓練集
X_train = X[:-2]
X_test = X[-2:]
y_train = y[:-2]
y_test = y[-2:]

# 建立一個支持向量迴歸 (SVR) 的模型，這裡我們使用預設的參數，您可以根據您的需求來調整它們
model = SVR()

# 用訓練集來訓練模型
model.fit(X_train, y_train)

# 用測試集來預測目標
y_pred = model.predict(X_test)

# 印出預測結果和實際結果的比較
print("pred:", y_pred)
print("exper:", y_test.values)

# 計算並印出均方誤差和決定係數
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("mse:", mse)
print("coeff:", r2)



