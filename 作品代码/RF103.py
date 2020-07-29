#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：huawei5G -> loc_clustering
@Author ：Mr. Young
@Date   ：2020/5/10 14:04
@Desc   ：南京邮电大学论文实现（北大20指标+随机森林+加权KMeans）
fightting Young
=================================================='''
import pandas as pd
import numpy as np
from numpy import *

#calcuDistance距离影响度计算
def calcuDistance(curTest,centroids,n):
    k = np.shape(centroids)[0]
    sumDis = 0
    for i in range(k):
        sumDis += sum(power(curTest-centroids[i,1:3],2))
    return sum(power(curTest-centroids[n,1:3],2))/sumDis

#划分数据到最近的簇中
def minDistance(apTest_dis,centroidList_dis):
    m = shape(apTest_dis)[0]
    k = shape(centroidList_dis)[0]
    clusterAssment = mat(zeros((m,3))) #第一列存放所属簇  第二列存放到该簇的距离度量  第三列存放该点到簇中心的距离的平方
    for i in range(m):
        minDist = inf
        minIndex = -1
        for j in range(k):
            distJI = calcuDistance(apTest_dis[i,0:2],centroidList_dis,j)
            if distJI < minDist:
                minDist = distJI
                minIndex = j
        distance = sum(power(apTest_dis[i,0:2]-centroidList_dis[minIndex,1:3],2))
        clusterAssment[i,:] = minIndex, minDist, distance
    return clusterAssment

#根据簇中所有点的均值选择基站位置
def getCentroids_young(apTest_getloc,centroidList_getloc,clusterDict):
    m = np.shape(apTest_getloc)[0]  # m个测试点数据
    k = np.shape(centroidList_getloc)[0]
    isChanged = False
    newCentroids = centroidList_getloc
    for j in range(k):
        sum_x = 0
        sum_y = 0
        count = 1
        for i in range(m):
            if(clusterDict[i,0] == j):
                count += 1
                sum_x += apTest_getloc[i,0]
                sum_y += apTest_getloc[i,1]
        newBS_X = (sum_x)/count
        newBS_Y = (sum_y)/count
        if(newCentroids[j,1:3]!=newBS_X,newBS_Y):
            isChanged = True
            newCentroids[j,1:3] = newBS_X, newBS_Y
    return newCentroids,isChanged

def sum_RSRPForCell(W_RSRP,clusterDict,k,lowPower=-103):
    m = shape(clusterDict)[0]
    RSRP_EachCell = np.zeros((k,1))
    for j in range(k):
        temp = 0
        for i in range(m):
            if(clusterDict[i,0] == j):
                temp += (W_RSRP[i])
        RSRP_EachCell[j,0] = temp
    return RSRP_EachCell
#更新簇的位置 根据每个簇中的所有点进行更新
def getCentroids(apTest_getloc,centroidList_getloc,clusterDict):
    m = np.shape(apTest_getloc)[0]  # m个测试点数据
    k = np.shape(centroidList_getloc)[0]
    W_RSRP = abs(apTest_getloc[:,2])
    RSRP_EachCell = sum_RSRPForCell(W_RSRP,clusterDict,k)
    # temp = abs(apTest_getloc[:,2])
    # W_RSRP = (temp-mean(temp))/var(temp)
    isChanged = False
    newCentroids = centroidList_getloc
    # for j in range(k):
    #     sumDis_up_x = 0
    #     sumDis_below = 0
    #     sumDis_up_y = 0
    #     for i in range(m):
    #         if(clusterDict[i,0] == j):
    #             temp = calcuDistance(apTest_getloc[i, 0:2],centroidList_getloc, j)*(W_RSRP[i]/RSRP_EachCell[j,0])
    #             # temp = calcuDistance(apTest_getloc[i, 0:2], centroidList_getloc, j) * (W_RSRP[i])
    #             # sumDis_up_x += temp * np.sqrt(sum(np.power(apTest_getloc[i, 0]-centroidList_getloc[j,1],2)))
    #             # sumDis_up_y += temp * np.sqrt(sum(np.power(apTest_getloc[i, 1]-centroidList_getloc[j,2],2)))
    #             sumDis_up_x += temp * apTest_getloc[i, 0]
    #             sumDis_up_y += temp * apTest_getloc[i, 1]
    #             sumDis_below += temp
    #     newBS_X = (sumDis_up_x / sumDis_below)
    #     newBS_Y = (sumDis_up_y / sumDis_below)
    #     if(newCentroids[j,1:3]!=newBS_X,newBS_Y):
    #         isChanged = True
    #         newCentroids[j,1:3] = newBS_X, newBS_Y

    #young提出的加权Means实现(同时考虑基站周围相同衰落点的空间位置对基站位置x,y调整的影响幅度)
    for j in range(k):
        shift_x = []
        shift_y = []
        for i in range(m):
            if(clusterDict[i,0] == j):
                temp = calcuDistance(apTest_getloc[i, 0:2], centroidList_getloc, j) * (W_RSRP[i]/RSRP_EachCell[j,0])
                shift_x += temp*(apTest_getloc[i,0]-centroidList_getloc[j,1])
                shift_y += temp*(apTest_getloc[i,0]-centroidList_getloc[j,2])
        newBS_X = centroidList_getloc[j,1] + shift_x
        newBS_Y = centroidList_getloc[j,2] + shift_y
        if(newCentroids[j,1:3]!=newBS_X,newBS_Y):
            isChanged = True
            newCentroids[j,1:3] = newBS_X, newBS_Y
    return newCentroids,isChanged

#计算簇集合间的均方误差
def getVar(clusterAssment):
    a = np.array(clusterAssment[:,1])
    b = np.array(clusterAssment[:,2])
    sse_sum = sum(a*b)/shape(clusterAssment)[0]
    return sse_sum

#计算总的目标函数
def calcuObjective(preRSRP,low = -103):
    sum = 0
    count = 0
    for i in preRSRP:
        if i < low:
            count += 1
            diff = np.square(i-low)
            sum += diff
    return sum,count,mean(preRSRP)

#重新刷新预测RSRP指标
def refreshRSRP(centroidList_re,apTest_re):
    print('------更新RSRP开始------')
    cellIndex = centroidList_re[:,0]
    allRSRP = np.array([])
    count = 0
    k = shape(centroidList_re)[0]
    for index in cellIndex:
        df_cell_Peking = pd.read_csv('./EachCellData_Peking/'+str(index)+'.csv')
        df_cell_Origin = pd.read_csv('./EachCellData/'+str(index)+'.csv')

        df_cell_Peking = df_cell_Peking.drop(['allRSRP','CellIndex'],axis=1)

        df_cell_Origin.loc[:,['allCell_X']] =  centroidList_re[count,1]
        df_cell_Origin.loc[:,['allCell_Y']] = centroidList_re[count, 2]

        count += 1

        allX = np.array(df_cell_Origin['all_X'])
        allY = np.array(df_cell_Origin['all_Y'])
        allCellX = np.array(df_cell_Origin['allCell_X'])
        allCellY = np.array(df_cell_Origin['allCell_Y'])
        allAzimuth = np.array(df_cell_Origin['allAzimuth'])
        allED = np.array(df_cell_Origin['all_ED'])
        allMD = np.array(df_cell_Origin['all_MD'])
        allH = np.array(df_cell_Origin['allH'])
        allCellAltitude = np.array(df_cell_Origin['allCellAltitude'])
        allAltitude = np.array(df_cell_Origin['allAltitude'])
        allF = np.array(df_cell_Origin['allF'])



        len = shape(allX)[0]

        Angle_North = zeros((len,1))
        Angle_Horizontal = zeros((len,1))
        SignalOnHeight = zeros((len,1))
        Distance_horizontal = zeros((len,1))
        LinkDistance = zeros((len,1))
        LinkLoss = zeros((len,1))

        # for i in range(len):
        #     for j in range(k):
        #         if(clusterDict_re[i,0] == j):
        #             if (allX[i] >= centroidList_re[j,0]):
        #                 Angle_North[i,0] = arccos((allY[i])/(sqrt(power(allX[i],2)+power(allY[i],2))))*(180/pi)
        #             else:
        #                 Angle_North[i,0] = 360 - arccos((allY[i]) / (sqrt(power(allX[i], 2) + power(allY[i], 2)))) * (
        #                             180 / pi)
        #             SignalOnHeight[i, 0] = allH[i] + allCellAltitude[i] - allAltitude[i] - tan(allED[i] + allMD[i]) * sqrt(
        #                 power(centroidList_re[j,0] - allX[i], 2) + power(centroidList_re[j,1] - allY[i], 2))
        #             Distance_horizontal[i, 0] = sqrt(power(centroidList_re[j,0] - allX[i], 2) + power(centroidList_re[j,1] - allY[i], 2))
        #             LinkDistance[i, 0] = sqrt(power(centroidList_re[j,0] - allX[i], 2) + power(centroidList_re[j,1] - allY[i], 2) + power(
        #                 allH[i] + allCellAltitude[i] - allAltitude[i], 2))
        #             LinkLoss[:, 0] = 32.44 + 20 * np.log10(allF[i]) + 20 * log10(LinkDistance[i,0])
        #     if((Angle_North[i]-allAzimuth[i])<=180):
        #         Angle_Horizontal[i,0] = Angle_North[i] - allAzimuth[i]
        #     else:
        #         Angle_Horizontal[i,0] = 360 - (Angle_North[i]-allAzimuth[i])

        for i in range(len):
            if (allX[i] >= allCellX[i]):
                Angle_North[i, 0] = arccos((allY[i]) / (sqrt(power(allX[i], 2) + power(allY[i], 2)))) * (180 / pi)
            else:
                Angle_North[i, 0] = 360 - arccos((allY[i]) / (sqrt(power(allX[i], 2) + power(allY[i], 2)))) * (
                        180 / pi)
        for i in range(len):
            if ((Angle_North[i] - allAzimuth[i]) <= 180):
                Angle_Horizontal[i, 0] = Angle_North[i] - allAzimuth[i]
            else:
                Angle_Horizontal[i, 0] = 360 - (Angle_North[i] - allAzimuth[i])

        SignalOnHeight[:, 0] = allH + allCellAltitude - allAltitude - tan(allED + allMD) * sqrt(
            power(allCellX - allX, 2) + power(allCellY - allY, 2))
        Distance_horizontal[:, 0] = sqrt(power(allCellX - allX, 2) + power(allCellY - allY, 2))
        LinkDistance[:, 0] = sqrt(power(allCellX - allX, 2) + power(allCellY - allY, 2) + power(allH + allCellAltitude - allAltitude, 2))

        SignalOnHeight[:,0] = allH+allCellAltitude-allAltitude-tan(allED+allMD)*sqrt(power(allCellX-allX,2)+power(allCellY-allY,2))
        Distance_horizontal[:,0] = sqrt(power(allCellX-allX,2)+power(allCellY-allY,2))
        LinkDistance[:,0] = sqrt(power(allCellX-allX,2)+power(allCellY-allY,2)+power(allH+allCellAltitude-allAltitude,2))
        # LinkDistance[:,np.newaxis]
        # LinkLoss[:,0] = 32.44 + 20*np.log10(allF)+20*log10(LinkDistance)

        df_cell_Peking['Angle_North'] = Angle_North
        df_cell_Peking['Angle_horizontal'] = Angle_Horizontal
        df_cell_Peking['SignalOnHeight'] = SignalOnHeight
        df_cell_Peking['Distance_horizontal'] = Distance_horizontal
        df_cell_Peking['LinkDistance'] = LinkDistance
        # df_cell_Peking['LinkLoss'] = LinkLoss

        data = df_cell_Peking[[col for col in df_cell_Peking.columns]]
        RF_RSRP = model.predict(data)
        allRSRP = np.append(allRSRP,RF_RSRP)
    apTest_re[:,2] = allRSRP
    print('------更新RSRP结束------')
    return apTest_re
def myloc_clustering(centroidList,apTest):
    print('开始迭代计算基站位置')
    clusterDict = minDistance(apTest, centroidList)
    newVar = getVar(clusterDict)
    print('***第1次迭代***')
    oldVar = -0.1

    k = 2
    isChanged = True

    while (abs(newVar - oldVar) >= 0.1 and isChanged):
        centroidList, isChanged = getCentroids_young(apTest, centroidList, clusterDict)
        clusterDict = minDistance(apTest, centroidList)
        # np.savetxt('./results/clusterDict_test.dat',clusterDict)
        # np.savetxt('./results/centroids_test.dat', centroidList[:, 1:3])

        oldVar = newVar
        newVar = getVar(clusterDict)
        print('***第%d次迭代***' % k)
        k += 1
        if k>15:
            break

    np.savetxt('./KMeans_BS_Result_Peking/RF_KMeans_BS_40_103_Rand2/Centroid.dat', centroidList)
    print('迭代结束')
    #返回新的基站位置  新的测试点数据 （centroidList，apTest）
    return centroidList,clusterDict

baseStation = pd.read_csv('./KMeans_BS_Before/KMeans_BS_40_Rand2/CellInfo.csv')
bsLoc = baseStation.values
centroidList = bsLoc

apTest = pd.read_csv('./KMeans_BS_Before/KMeans_BS_40_Rand2/TestInfo.csv')
apTest = apTest.values

# myloc_clustering(centroidList,apTest)

from keras.models import load_model
from sklearn.externals import joblib

#加入总目标函数优化 两层while循环
model = joblib.load('./ModelsToAnalysis_Peking/N10_TreeNum200_depth40_sqrt.pkl')

allObjectValue = []
alllowPowerRatio = []
allmeanRSRP = []

newObj,lowRatio,meanRSRP = calcuObjective(apTest[:,2])

allObjectValue.append(newObj)
alllowPowerRatio.append(lowRatio)

oldObj = -1
k=2
print(newObj-oldObj)

while(abs(newObj-oldObj)>=1):
    print('********总的迭代开始*********')
    newCentroids_loc, newCluster = myloc_clustering(centroidList,apTest)
    refreshRSRP_apTest = refreshRSRP(newCentroids_loc,apTest)
    apTest = refreshRSRP_apTest
    centroidList = newCentroids_loc
    oldObj = newObj
    newObj,lowRatio,meanRSRP = calcuObjective(refreshRSRP_apTest[:,2])
    print(abs(newObj-oldObj))
    allObjectValue.append(newObj)
    alllowPowerRatio.append(lowRatio)
    allmeanRSRP.append(meanRSRP)
    print('********第%d次总迭代结束********' %k)
    k += 1
    if k>15:
        break
np.savetxt('./KMeans_BS_Result_Peking/RF_KMeans_BS_40_103_Rand2/RefreshRSRP.dat',refreshRSRP_apTest)
np.savetxt('./KMeans_BS_Result_Peking/RF_KMeans_BS_40_103_Rand2/LowPowerCount.dat',alllowPowerRatio)


