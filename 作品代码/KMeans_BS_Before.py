#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：huawei5G -> K_means_BS
@Author ：Mr. Young
@Date   ：2020/4/28 10:47
@Desc   ：网络规划模块的KMeans算法前的数据处理与显示
=================================================='''
import numpy as np
from keras.models import load_model
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt\

#定义字体

cm1 = plt.cm.get_cmap('jet')
cm2 = plt.cm.get_cmap('hot')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
myFont = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc",size=7)
myFont_Euclid = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\Euclid.ttf",size=7)

header_24 = ['Distance', 'Signal Line Height', '3Axis Distance', 'Antenna Distance',
       'Antenna Downtilt', 'Antenna Azimuth', 'Cell Index', 'Cell X', 'Cell Y',
       'Height', 'Azimuth', 'Electrical Downtilt', 'Mechanical Downtilt',
       'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height',
       'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
       'Clutter Index', 'RSRP']

def lowPower(RSRP,low=-110):
    allCounts = len(RSRP)
    lowCounts = 0
    for i in RSRP:
        if(i<=low):
            lowCounts = lowCounts+1
    ratio = lowCounts/allCounts
    return ratio,lowCounts

#选择基站个数进行聚簇算法分析

df_cellInfo = pd.read_csv('./DataUtil/CellIndexAndDistance.csv')
df_cellInfo.sort_values('CellX',inplace=True) #按照基站X距离升序排序

# from sklearn.externals import joblib
# model = joblib.load('./ModelsToAnalysis/0514_opti.pkl')

CellIndex = df_cellInfo['CellIndex']
CellIndex = CellIndex[100:180:2] # python中最后一个数代表切片步长

allCellX = []
allCellY = []

allTestRSRP = []
allTestX = []
allTestY = []
allTestCluster = []

fig = plt.figure(1)
for index in CellIndex:
    print(index)
    df_cell = pd.read_csv('./EachCellData/'+str(index)+'.csv')
    df_cell_Peking = pd.read_csv('./EachCellData_24/' + str(index) + '.csv')
    CellX = np.unique(df_cell['allCell_X'])
    CellY = np.unique(df_cell['allCell_Y'])

    allCellX.append(CellX[0])
    allCellY.append(CellY[0])

    TestX = np.array(df_cell['all_X'])
    TestY = np.array(df_cell['all_Y'])
    TestClutter = np.array(df_cell['allClutter'])

    #保证初始化的时候接收点RSRP为模型预测结果
    # df_cell_Peking = df_cell_Peking.drop(['RSRP'],axis=1)
    # data = df_cell_Peking[[col for col in df_cell_Peking.columns]]
    # RSRP = np.array(model.predict(data))
    RSRP = df_cell_Peking['RSRP']
    allTestX = np.append(allTestX,TestX)
    allTestY = np.append(allTestY,TestY)
    allTestCluster = np.append(allTestCluster,TestClutter)
    allTestRSRP = np.append(allTestRSRP,RSRP)
    # plt.figure(1)
    # plt.scatter(TestX, TestY,
    #             c=(Index),
    #             s=np.pi*0.1, alpha=0.5, cmap=cm2)
    # plt.xlabel('X位置/m')
    # plt.ylabel('Y位置/m')
    # plt.title('原始基站地理类型展示')
    # plt.grid(True)
    #
    # plt.figure(2)
    # plt.scatter(TestX, TestY,
    #             c=(RSRP),
    #             s=np.pi*0.1, alpha=0.5, cmap=cm2)
    # plt.xlabel('X位置/m')
    # plt.ylabel('Y位置/m')
    # plt.title('站址规划前基站信号覆盖展示')
    # plt.grid(True)
# plt.figure(1)
# plt.plot(allCellX,allCellY,'b*')
# plt.show()
#
# plt.figure(2)
# plt.plot(allCellX,allCellY,'b*')
# plt.show()
#
# #k-means聚类算法数据准备
# pd_cell = pd.DataFrame(data=list(zip(CellIndex,allCellX,allCellY)),columns=['CellIndex','allCellX','allCellY'])
# pd_cell.to_csv('./KMeans_BS_Before/KMeans_BS_40_Rand2/CellInfo.csv',index=False)
#
# pd_test = pd.DataFrame(data=list(zip(allTestX,allTestY,allTestRSRP,allTestCluster)),columns=['allTestX','allTestY','allTestRSRP','allTestCluster'])
# pd_test.to_csv('./KMeans_BS_Before/KMeans_BS_40_Rand2/TestInfo.csv',index=False)

ratio = np.array(lowPower(allTestRSRP,-105)).reshape(1,-1)
np.savetxt('./KMeans_BS_Before/KMeans_BS_40_Rand2/LowRSRP_Ratio_105.dat',ratio)
#统计弱信号覆盖比例<-110dBm
# for index in CellIndex:
#     # print(index)
#     df_cell = pd.read_csv('../TrainData/EachCellData/'+str(index)+'.csv')
#     RSRP = df_cell['allRSRP']
#     print('{:.2%}'.format(lowPower(RSRP)))
