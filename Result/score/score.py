import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as MSE

def R2(y_true, y_pred):
    r2(y_pred, y_true)

def RMSE(y_true, y_pred):
    np.sqrt(MSE(y_pred, y_true))

pred = pd.read_csv(sys.argv[2]).to_numpy()[:,1:]
gt = pd.read_csv(sys.argv[1]).to_numpy()[:,1:]

r2_score = R2(gt, pred)
RMSE_score = RMSE(gt, pred)

print("r2 score : {0}\nRMSE score : {1}".format(r2_score, RMSE_score))
