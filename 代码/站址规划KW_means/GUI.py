#%% 导入
import PySimpleGUI as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams
import joblib
import random
from keras.models import load_model
config = {
    "font.family":'serif',
    "font.size": 9,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    'axes.unicode_minus':False}
rcParams.update(config)
#%% ZOOM画图
def partial(data, tag_X, tag_Y, area):
    x_lower, x_upper, y_lower, y_upper = area
    data = data[data[tag_X] > x_lower]
    data = data[data[tag_X] < x_upper]
    data = data[data[tag_Y] > y_lower]
    data = data[data[tag_Y] < y_upper]
    return data

def zoomplot(data,celldata,label,filename,minn):
    # if label == 'RSRP':
    #     cnlabel, dw , zm= '信号接收强度(规划前)', '/dBm', 4 / 9
    # elif label == 'Clutter Index':
    #     cnlabel, dw, zm = '地物类型索引', '', 4 / 9

    if filename == 'map_RSRP':
        cnlabel, dw , zm= '信号接收强度(规划前)', '/dBm', 4 / 9
    elif filename == 'map_Clutter':
        cnlabel, dw, zm = '地物类型索引', '', 4 / 9
    else:
        cnlabel, dw, zm = '信号接收强度(规划后)', '/dBm', 4 / 9
    maxx, minx, maxy, miny = max(data['X']), min(data['X']), max(data['Y']), min(data['Y'])
    detx, dety = maxx-minx, maxy-miny
    area = np.array([minx+detx*zm, maxx-detx*zm, miny+dety*zm, maxy-dety*zm])
    datap = partial(data, 'X', 'Y', area)
    Xp, Yp, Zp = datap['X'], datap['Y'], datap[label]
    cX, cY, cZ = celldata['Cell X'], celldata['Cell Y'], celldata['Cell X']*0
    X, Y, Z = data['X'], data['Y'], data[label]
    c = 'Spectral'#'gist_earth' 'winter' 'Spectral' 'jet' 'hot'
    cm = plt.cm.get_cmap(c)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax0 = ax.scatter(X/1000, Y/1000, c=Z, cmap=cm, s=np.pi*0.25, alpha=1, marker='.',label=cnlabel+dw,vmin=minn)
    ax.scatter(cX / 1000, cY / 1000, c='#FF0000', s=np.pi * 2, alpha=1, label='Cell', marker='*')
    ax.set_xlabel('X坐标/km')
    ax.set_ylabel('Y坐标/km')
    ax.legend(loc=4)
    #ax.axis('equal')
    fig.colorbar(ax0, ax=ax)
    ax.set_title(cnlabel+'分布', fontsize=20)
    # 嵌入绘制局部放大图的坐标系
    axins = inset_axes(ax, width="35%", height="35%",loc='upper left',
                       bbox_to_anchor=(0.01, -0.01, 1, 1),
                       bbox_transform=ax.transAxes)
    # 在子坐标系中绘制原始数据
    axins.scatter(Xp/1000, Yp/1000, c=Zp, cmap=cm, s=np.pi*0.25, alpha=1, marker='.',label=cnlabel+dw,vmin=minn)
    axins.scatter(cX/1000, cY/1000, c='#FF0000', s=np.pi*2, alpha=1, label='Cell', marker='*')
    #axins.legend(loc='upper left',fontsize=8)
    xlim0, xlim1, ylim0, ylim1 = area/1000
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    #axins.axis('equal')
    #plt.tight_layout()
    plt.savefig('Images/'+filename+'.png',dpi=100)

#%% 衰减图:1-6:开阔；7-9植被；10-14：城区；15-20：城郊
def COST(freq, distance, height, terrain, hmobile):
    # f=MHz,d=km,h=m
    k1, k2, k3, k4, k5 = 46.3, 33.9, -13.82, 44.9, -6.55
    if 1:
        k6, k7, k8, k9 = 3.2, 11.75, -4.97, 3
        a = (1.1 * np.log10(freq) - 0.7) * hmobile - (1.56 * np.log10(freq) - 0.8)
        loss = k1 + k2 * np.log10(freq) + k3 * np.log10(height) + (k4 + k5 * np.log10(height)) * np.log10(distance) \
               - (k6 * np.log10(k7 * hmobile)) ** 2 + k8 + k9 + a
    elif 0:
        k6, k7, k8, k9 = 1.11, -0.7, 1.56, -0.8
        loss = k1 + k2 * np.log10(freq) + k3 * np.log10(height) + (k4 + k5 * np.log10(height)) * np.log10(distance) \
               - k6 * (np.log10(freq) + k7) * hmobile + k8 * np.log10(freq) + k9
    else:
        loss = []
    return loss
def ECC(freq, distance, height, terrain, hmobile):
    # f=GHz,d=km,h=m
    if 0:
        loss = 137.71345 + 29.83 * np.log10(distance) + 35.9085 * np.log10(freq)\
               +9.56 * (np.log10(freq)) ** 2 - 13.958 * np.log10(height / 200)\
               -5.8 * np.log10(height / 200) * (np.log10(distance)) ** 2\
               -42.57 * np.log10(hmobile) - 13.7 * np.log10(freq) * np.log10(hmobile)
    elif 1:
        loss = 114.672 + 29.83 * np.log10(distance) - 13.958 * np.log10(height / 200)\
               +27.894 * np.log10(freq) + 9.56 * (np.log10(freq)) ** 2\
               -5.8 * np.log10(height / 200) * np.log10(distance) - 0.795 * hmobile
    return loss
def lplot(ax, x, Op, loss, label, sub, lw, ls, mk):
    MSE = ((np.array([loss]) - np.array(Op)) ** 2) / len(Op)
    RMSE = np.sqrt(MSE.sum().round(3)).round(3)
    ax.plot(x, loss, label=label + '\nRMSE=' + str(RMSE) + 'dB', lw=lw, linestyle=ls, marker=mk, alpha=0.5)
def standardize(train):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    if 1:
        mean = np.loadtxt('ModelsToAnalysis/mean.txt')
        std = np.loadtxt('ModelsToAnalysis/std.txt')
    train = (train-mean)/std
    return train
def DrawModel(data,model):
    if 1:
        dataa = data[(data['Distance'] > 0) & (data['Distance'] < 1000)].sample(n=300, random_state=123)
        datab = data[(data['Distance'] > 1000) & (data['Distance'] < 3000)].sample(n=300, random_state=123)
        datac = data[(data['Distance'] > 3000)].sample(n=300, random_state=123)
        data = pd.concat([dataa, datab])
        # data = pd.concat([dataa])
    data = data[(data['Distance'] > 0)].sample(n=600, random_state=123)
    data.sort_values(by=['Distance'], inplace=True)
    sub, fileroot = 0, 'ModelsToAnalysis/'
    modelname = ['COST', 'ECC33', 'RF', 'CNN', 'DNN']
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    dloss = data['RS Power'] - data['RSRP']
    ax.plot(data['Distance'], dloss, label='真实值$RSRP$', marker='.', alpha=0.5)
    closs = COST(data['Frequency Band'], data['3Axis Distance'] / 1000, data['Height'], data['Clutter Index'],
                data['Height'] * 0 + 3)
    lplot(ax,data['Distance'], dloss, closs, 'COST231-Hata模型', sub, 2, '--', None)
    eloss = ECC(data['Frequency Band']/1000, data['3Axis Distance'] / 1000, data['Height'], data['Clutter Index'],
               data['Height'] * 0 + 3)
    lplot(ax,data['Distance'], dloss, eloss, 'ECC33模型', sub, 2, '-.', None)
    X = standardize(data.iloc[:,:-1])
    if model is 'RF':
        # model = joblib.load(fileroot + '0514_opti.pkl')
        RSRP = RFmodel.predict(data.iloc[:,:-1])
        label = '随机森林模型'
    elif model is 'CNN':
        # model = load_model(fileroot + 'CNN_Model.h5')
        X = X.values.reshape((X.shape[0], X.shape[1], 1))
        RSRP = CNNmodel.predict(X)
        label = 'CNN模型'
    elif model is 'DNN':
        # model = load_model(fileroot + 'DNN_Model.h5')
        X = X.values.reshape((X.shape[0], X.shape[1]))
        RSRP = DNNmodel.predict(X)
        label =  'DNN模型'
    RSRP = RSRP.reshape((RSRP.shape[0]))
    loss = data['RS Power'] - RSRP
    lplot(ax,data['Distance'], dloss, loss, label, sub, 0.5, '-', '^')
    # ax.set_title('预测值$RSRP(\mathrm{\mathbf{x}})$与真实值$RSRP$(' + label + ')', fontsize=20)
    ax.set_title('预测值$\\bar{RSRP}$与真实值$RSRP$(' + label + ')', fontsize=20)
    ax.set_xlabel('距离$d$/m')
    ax.set_ylabel('无线传播损耗$PL$/dB')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig('Images/model.png', dpi=100)

#%% 更新
def graph1(data):
    celldata = data[['Cell X','Cell Y','Cell Index']].drop_duplicates()
    zoomplot(data,celldata,'Clutter Index','map_Clutter',0)
    print('plot graph1')
    window.FindElement("graph1").DrawImage('./Images/map_Clutter.png', color='black', location=(-100, 100))

def graph2(data):
    celldata = data[['Cell X', 'Cell Y', 'Cell Index']].drop_duplicates()
    zoomplot(data,celldata,'RSRP','map_RSRP',-140)
    print('plot graph2')
    window.FindElement("graph2").DrawImage('./Images/map_RSRP.png', color='black', location=(-100, 100))

def graph3(data,model):
    DrawModel(data,model)
    print('plot graph3')
    window.FindElement("graph3").DrawImage('./Images/model.png', color='black', location=(-100, 100))

def graph4(data,celldata):
    zoomplot(data, celldata, 'RSRP', 'map_update',-140)
    print('plot graph4')
    window.FindElement("graph4").DrawImage('./Images/map_update.png', color='black', location=(-100, 100))

def update(value):
    rootPath = 'KMeans_BS_Result_Peking/'
    valueData = int(value['LIMIT'])
    # if valueData==-100:
    #     randInt = random.sample(range(0,3),1)  # 0,1,2随机数
    #     if randInt[0] == 0:
    #         filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'RefreshRSRP.dat'
    #     elif randInt[0] == 1:
    #         filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData)  +'_Rand2'+ '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '_Rand2' + '/' + 'RefreshRSRP.dat'
    #     else:
    #         filePath = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'RefreshRSRP.dat'
    # elif valueData == -103:
    #     randInt = random.sample(range(0, 3), 1)  # 0,1,2随机数
    #     print(randInt)
    #     if randInt[0] == 0:
    #         filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'RefreshRSRP.dat'
    #     elif randInt[0] == 1:
    #         filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData)  +'_Rand2'+ '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '_Rand2' + '/' + 'RefreshRSRP.dat'
    #     else:
    #         filePath = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'RefreshRSRP.dat'
    # else:  # -105
    #     randInt = random.sample(range(0, 3), 1)  # 0,1,2随机数
    #     if randInt[0] == 0:
    #         filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'RefreshRSRP.dat'
    #     elif randInt[0] == 1:
    #         filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData)  +'_Rand2'+ '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '_Rand2' + '/' + 'RefreshRSRP.dat'
    #     else:
    #         filePath = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'Centroid.dat'
    #         refreshRSRP_Path = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'RefreshRSRP.dat'

    randInt = random.sample(range(0, 3), 1)  # 0,1,2随机数
    if randInt[0] == 0:
        filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'Centroid.dat'
        refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '/' + 'RefreshRSRP.dat'
    elif randInt[0] == 1:
        filePath = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '_Rand2' + '/' + 'Centroid.dat'
        refreshRSRP_Path = rootPath + 'RF_KMeans_BS_20_' + str(-valueData) + '_Rand2' + '/' + 'RefreshRSRP.dat'
    else:
        filePath = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'Centroid.dat'
        refreshRSRP_Path = rootPath + 'RF_KMeans_BS_40_' + str(-valueData) + '/' + 'RefreshRSRP.dat'

    CellIndex = np.loadtxt(filePath)
    RSRP_update = np.loadtxt(refreshRSRP_Path)
    data_update,celldata = pd.DataFrame([]),pd.DataFrame([])
    data_update['X'],data_update['Y'] = RSRP_update[:, 0],RSRP_update[:, 1]
    data_update['RSRP'],data_update['Clutter Index'] = RSRP_update[:, 2],RSRP_update[:, 3]
    celldata['Cell Index'],celldata['Cell X'],celldata['Cell Y'] = CellIndex[:, 0],CellIndex[:, 1],CellIndex[:, 2]
    data = pd.DataFrame([])
    for index in (CellIndex[:, 0]):
        df_cell = pd.read_csv('EachCellData_24/' + str(int(index)) + '.csv')
        data = pd.concat([data, df_cell])
    graph1(data)
    graph2(data)
    modelname = ['RF', 'CNN', 'DNN']
    modelname_CN = ['随机森林','CNN','DNN']
    for mname in modelname:
        if value[mname]:
            graph3(data,mname)
            window.FindElement("MODEL").Update('预测模型：'+modelname_CN[modelname.index(mname)]+'模型')
    graph4(data_update,celldata)
    before = str(data[data['RSRP']<float(value['LIMIT'])].shape[0])
    after = str(data_update[data_update['RSRP']<float(value['LIMIT'])].shape[0])
    print(before,after)
    window.FindElement("PLAN").Update('       弱信号点统计：规划前'+before+'个；规划后'+after+'个')

#%% 主界面
menu_def = [['预测模型', ['随机森林模型', 'CNN模型', 'DNN模型']]]
layout = [[sg.Menu(menu_def, tearoff=False, pad=(20,1), key='MENU')],
          [sg.Text('链路预测模块',size=(24,1),justification='left',font=("微软雅黑",20),auto_size_text=True),
           sg.Text(' 网络规划模块',size=(15,1),justification='left',font=("微软雅黑",20),auto_size_text=True)],
          [sg.Text('预测模型：随机森林模型', key='MODEL',size=(35, 1), justification='left', font=("Terminal",15), auto_size_text=True),
           sg.Text('       弱信号点统计：规划前(●—●)个，规划后(●—●)个', key='PLAN',size=(60, 1), justification='left', font=("Terminal", 14), auto_size_text=True)],
          [sg.Radio('CNN模型','MODEL',key='CNN',default=True),sg.Radio('DNN模型','MODEL',key='DNN'),
           sg.Radio('随机森林模型','MODEL',key='RF'),sg.Text("                 弱信号覆盖阈值"),
           sg.InputText('-100',size=(10,1),key='LIMIT',justification='left',font=("微软雅黑",12)),sg.Text('dBm'),
           sg.Button('更新', key='submit'), sg.Quit('退出', key='q')],
          [sg.Graph(canvas_size=(500, 400), graph_bottom_left=(-100, -100), background_color='white',
                    graph_top_right=(100, 100), key='graph1'),
           sg.Graph(canvas_size=(500, 400), graph_bottom_left=(-100, -100), background_color='white',
                    graph_top_right=(100, 100), key='graph2')],
          [sg.Graph(canvas_size=(500, 400), graph_bottom_left=(-100, -100), background_color='white',
                    graph_top_right=(100, 100), key='graph3'),
           sg.Graph(canvas_size=(500, 400), graph_bottom_left=(-100, -100), background_color='white',
                    graph_top_right=(100, 100), key='graph4')]]
window = sg.Window('基站规划客户端', layout, resizable=True, size=(1050,950), location=(400,10), font='微软雅黑').Finalize()
print('模型加载中......')
RFmodel = joblib.load('ModelsToAnalysis/' + '0514_opti.pkl')
print('随机森林模型加载完毕!')
CNNmodel = load_model('ModelsToAnalysis/' + 'CNN_Model.h5')
print('CNN模型加载完毕!')
DNNmodel = load_model('ModelsToAnalysis/' + 'DNN_Model.h5')
print('DNN模型加载完毕!')
while True:
    event, value = window.Read()
    print(event)
    if event is 'submit':
        update(value)
    elif event is None or 'q':
        break
window.Close()
