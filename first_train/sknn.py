import pandas as pd
#from sklearn import svm
from sklearn.neural_network import MLPRegressor
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

seed = 42
df = pd.read_excel("CrCoNi.xlsx",sheet_name="SUM")
data, label = [], []

for y, z in zip(range(20, 47, 2), range(50, 23, -2)):
    sheet_name = f'30{y}{z}'
    #print(sheet_name)
    try:
        data_label = pd.read_excel("CrCoNi.xlsx",header=None, sheet_name=sheet_name, usecols="A:B")
    except ValueError:
        print(f'sheet_name:{sheet_name} not exist')
        continue
    #print(data_label.columns)
    data_tmp, label_tmp = data_label[0].to_numpy(), data_label[1].to_numpy()
    cr_co_ni = np.array([30, y, z])
    size = len(data_tmp)
    cr_co_ni = np.repeat(cr_co_ni, size, axis=0).reshape((size,3))
    data_tmp = data_tmp.reshape((size, -1))
    #print(data_tmp.shape, cr_co_ni.shape)
    data.append(np.concatenate([data_tmp, cr_co_ni], axis=1))
    label.append(label_tmp)

data = np.concatenate(data, axis=0)
label = np.concatenate(label, axis=0)
print(data.shape)
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=.2, random_state=seed)
# X = df[['x', 'y', 'z']].iloc[:-2].to_numpy()
# y = df['strain'].iloc[:-2]
regr = MLPRegressor(random_state=seed)
regr.fit(train_x, train_y)
predy = regr.predict(test_x)

def mse_loss(test_y, predy):
    diff = test_y - predy
    square = diff ** 2
    mean = np.mean(square)
    return mean ** .5
print(mse_loss(test_y, predy))

joblib.dump(regr, 'ok.joblib')

#SVR()
#regr.predict([[1, 1]])
#array([1.5])



