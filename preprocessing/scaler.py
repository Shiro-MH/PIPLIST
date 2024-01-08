from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
scaler
StandardScaler()

scaler.mean_
array([1. ..., 0. ..., 0.33...])

scaler.scale_
array([0.81..., 0.81..., 1.24...])

X_scaled = scaler.transform(X_train)
X_scaled
array([[ 0.  ..., -1.22...,  1.33...],
       [ 1.22...,  0.  ..., -0.26...],
       [-1.22...,  1.22..., -1.06...]])