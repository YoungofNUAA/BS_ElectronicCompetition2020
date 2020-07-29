import numpy as np
from keras.models import load_model
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.externals import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
myFont = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc",size=7)
myFont_Euclid = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\Euclid.ttf",size=7)

RF = joblib.load('./RandomForests_Models/'+'V2_TreeNum200_depth40_log2.pkl')
X_test = np.loadtxt('./toTestData/'+'V2_TreeNum200_depth40_log2_X.dat')
Y_true = np.loadtxt('./toTestData/'+'V2_TreeNum200_depth40_log2_Y.dat')

predict = RF.predict(X_test[1:200])
print(metrics.mean_squared_error(Y_true, predict))

# f = plt.figure()
# plt.plot(predict,'b-o')
# plt.plot(Y_true[1:200],'r-*')
# plt.legend(['随机森林算法预测RSRP','真实测量RSRP'],prop=myFont,edgecolor='white')
# plt.xticks(fontproperties=myFont)
# plt.yticks(fontproperties=myFont_Euclid)
# plt.xlabel('随机测试点',fontproperties=myFont)
# plt.ylabel('测试点平均接收功率RSRP(dBm)',fontproperties=myFont)
# plt.grid(linestyle='-.',linewidth=0.5)
# f.savefig('V2_TreeNum200_depth40_log2.jpeg', dpi=600)