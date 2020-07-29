import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import math
filePath = 'train_data_Peking.dat'

df = pd.read_csv(filePath)
df = df.sample(frac=1).reset_index(drop=True)  #frac随机打乱 选取100%
RF = RandomForestRegressor(
    n_estimators=400,
    max_depth=40,
    min_samples_split=50,
    min_samples_leaf=20,
    oob_score=True,
    verbose=2,
    max_features='log2'
    )
train_count = int(0.9 * len(df))
RF.fit(df.ix[:train_count, :-1], df.ix[:train_count, 'allRSRP'])
from sklearn import metrics
print(metrics.mean_squared_error(df.ix[train_count:, 'allRSRP'], RF.predict(df.ix[train_count:, :-1])))

np.savetxt('./toTestData/'+'V7_TreeNum200_depth40_log2_X.dat',df.ix[train_count:, :-1])
np.savetxt('./toTestData/'+'V7_TreeNum200_depth40_log2_Y.dat',df.ix[train_count:, 'allRSRP'])

modelName = './RandomForests_Models/'+'V7_TreeNum200_depth40_log2.pkl'
from sklearn.externals import joblib
joblib.dump(RF,modelName,compress=9)


