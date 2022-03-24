import os
import sys
import pandas as pd
import numpy as np

def RMSE(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def R2(y_true, y_pred):
    err_2_sum = ((y_true-y_pred)**2).sum()
    dev_2_sum = ((y_true-np.mean(y_true))**2).sum()
    return 1-(err_2_sum/dev_2_sum)
    
pred = pd.read_csv(sys.argv[2]).to_numpy()[:,:]
gt = pd.read_csv(sys.argv[1]).to_numpy()[:,:]

RMSE_score = round(RMSE(gt, pred),4)
R2_score = round(R2(gt, pred),4)
score = round(((1-RMSE_score)/2 + R2_score/2),4)

print(f"score:{score}")
