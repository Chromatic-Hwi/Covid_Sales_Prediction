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

RMSE_score = RMSE(gt, pred)
R2_score = R2(gt, pred)
score = 1

print(RMSE_score, R2_score, score)
print(f"score:{score}")
