#%% 导入
import PySimpleGUI as sg
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams
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

def zoomplot(data,celldata,label,filename,minn,area):
    if label == 'Altitude':
        cnlabel, dw , zm= '海拔高度', '/m', 6 / 9
    elif label == 'Clutter Index':
        cnlabel, dw, zm = '地物类型索引', '', 6 / 9
    maxx, minx, maxy, miny = max(data['X']), min(data['X']), max(data['Y']), min(data['Y'])
    detx, dety = maxx-minx, maxy-miny
    #area = np.array([minx+detx*zm, maxx-detx*zm, miny+dety*zm, maxy-dety*zm])
    datap = partial(data, 'X', 'Y', area)
    Xp, Yp, Zp = datap['X'], datap['Y'], datap[label]
    cX, cY, cZ = celldata['Cell X'], celldata['Cell Y'], celldata['Cell X']*0
    data = data.sample(n=50000, random_state=123)
    X, Y, Z = data['X'], data['Y'], data[label]
    c = 'Spectral'#'gist_earth' 'winter' 'Spectral' 'jet' 'hot'
    cm = plt.cm.get_cmap(c)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax0 = ax.scatter(X/1000, Y/1000, c=Z, cmap=cm, s=np.pi*0.25, alpha=1, marker='.',label=cnlabel+dw,vmin=minn)
    #ax.scatter(cX / 1000, cY / 1000, c='#FF0000', s=np.pi * 2, alpha=1, label='Cell', marker='*')
    ax.set_xlabel('X坐标/km')
    ax.set_ylabel('Y坐标/km')
    ax.legend(loc=4)
    #ax.axis('equal')
    fig.colorbar(ax0, ax=ax)
    ax.set_title(cnlabel+'分布', fontsize=20)
    # 嵌入绘制局部放大图的坐标系
    axins = inset_axes(ax, width="45%", height="45%",loc='upper left',
                       bbox_to_anchor=(0.01, -0.01, 1, 1),
                       bbox_transform=ax.transAxes)
    # 在子坐标系中绘制原始数据
    axins.scatter(Xp/1000, Yp/1000, c=Zp, cmap=cm, s=np.pi*0.25, alpha=1, marker='.',label=cnlabel+dw,vmin=minn)
    #axins.scatter(cX/1000, cY/1000, c='#FF0000', s=np.pi*2, alpha=1, label='Cell', marker='*')
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
        loss = 137.71345 + 29.83 * np.log10(distance) + 35.9085 * np.log10(freq) \
               +9.56 * (np.log10(freq)) ** 2 - 13.958 * np.log10(height / 200) \
               -5.8 * np.log10(height / 200) * (np.log10(distance)) ** 2 \
               -42.57 * np.log10(hmobile) - 13.7 * np.log10(freq) * np.log10(hmobile)
    elif 1:
        loss = 114.672 + 29.83 * np.log10(distance) - 13.958 * np.log10(height / 200) \
               +27.894 * np.log10(freq) + 9.56 * (np.log10(freq)) ** 2 \
               -5.8 * np.log10(height / 200) * np.log10(distance) - 0.795 * hmobile
    return loss
def lplot(ax, x, Op, loss, label, sub, lw, ls, mk):
    MSE = ((np.array([loss]) - np.array(Op)) ** 2) / len(Op)
    RMSE = np.sqrt(MSE.sum().round(3)).round(3)
    ax.plot(x, loss, label=label + '\nRMSE=' + str(RMSE) + 'dB', lw=lw, linestyle=ls, marker=mk, alpha=0.5)
def standardize(train):
    if 1:
        mean = np.loadtxt('ModelsToAnalysis/mean.txt')
        std = np.loadtxt('ModelsToAnalysis/std.txt')
    train = (train-mean)/std
    return train
def DrawModel(data,model):
    if 0:
        dataa = data[(data['Distance'] > 0) & (data['Distance'] < 1000)].sample(n=200, random_state=123)
        datab = data[(data['Distance'] > 1000) & (data['Distance'] < 3000)].sample(n=200, random_state=123)
        datac = data[(data['Distance'] > 3000)].sample(n=200, random_state=123)
        data = pd.concat([dataa, datab, datac])
    data = data[(data['Distance'] > 0)].sample(n=600, random_state=123)
    data.sort_values(by=['Distance'], inplace=True)
    sub, fileroot = 0, 'ModelsToAnalysis/'
    modelname = ['COST', 'ECC33', 'RF', 'CNN', 'DNN']
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    dloss = data['RS Power'] - data['RSRP']
    ax.plot(data['Distance'], dloss, label='真实值$y$', marker='.', alpha=0.5)
    closs = COST(data['Frequency Band'], data['3Axis Distance'] / 1000, data['Height'], data['Clutter Index'],
                 data['Height'] * 0 + 3)
    lplot(ax,data['Distance'], dloss, closs, 'COST231-Hata模型', sub, 2, '--', None)
    eloss = ECC(data['Frequency Band']/1000, data['3Axis Distance'] / 1000, data['Height'], data['Clutter Index'],
                data['Height'] * 0 + 3)
    lplot(ax,data['Distance'], dloss, eloss, 'ECC33模型', sub, 2, '-.', None)
    X = standardize(data.iloc[:,:-1])
    if model is 'RF':
        model = joblib.load(fileroot + '0514_opti.pkl')
        RSRP = model.predict(data.iloc[:,:-1])
        label = '随机森林模型'
    elif model is 'CNN':
        model = load_model(fileroot + 'CNN_Model.h5')
        X = X.values.reshape((X.shape[0], X.shape[1], 1))
        RSRP = model.predict(X)
        label = 'CNN模型'
    elif model is 'DNN':
        model = load_model(fileroot + 'DNN_Model.h5')
        X = X.values.reshape((X.shape[0], X.shape[1]))
        RSRP = model.predict(X)
        label =  'DNN模型'
    RSRP = RSRP.reshape((RSRP.shape[0]))
    loss = data['RS Power'] - RSRP
    lplot(ax,data['Distance'], dloss, loss, label, sub, 0.5, '-', '^')
    ax.set_title('预测值$H(\mathrm{\mathbf{x}})$与真实值$y$(' + label + ')', fontsize=20)
    ax.set_xlabel('距离$d$/m')
    ax.set_ylabel('无线传播损耗$PL$/dB')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig('Images/model.png', dpi=100)

#%% 更新
def graph1(data,area):
    celldata = data[['Cell X','Cell Y','Cell Index']].drop_duplicates()
    zoomplot(data,celldata,'Clutter Index','map2_Clutter',1,area)
    print('plot graph1')
    window.FindElement("graph1").DrawImage('./Images/map2_Clutter.png', color='black', location=(-100, 100))

def graph2(data,area):
    celldata = data[['Cell X', 'Cell Y', 'Cell Altitude']].drop_duplicates()
    zoomplot(data,celldata,'Altitude','map2_Altitude',470,area)
    print('plot graph2')
    window.FindElement("graph2").DrawImage('./Images/map2_Altitude.png', color='black', location=(-100, 100))

def update(data,value):
    if data.shape[0]<1000:
        data = pd.read_hdf('alldata_RSRP.h5',key='data')
        print('data loaded')
    area = np.array([float(value['Xmin']), float(value['Xmax']), float(value['Ymin']), float(value['Ymax'])])*1000
    graph1(data,area)
    graph2(data,area)
    return data

#%% 主界面
menu_def = []
layout = [[sg.Menu(menu_def, tearoff=False, pad=(20,1), key='MENU')],
          [sg.Text('实测数据可视化模块',size=(31,1),justification='left',font=("微软雅黑",20),auto_size_text=True)],
          [sg.Text('水平距离范围',size=(6,1),justification='left',font=("微软雅黑",12),auto_size_text=True),
           sg.InputText('409',size=(5,1),key='Xmin',justification='left',font=("微软雅黑",12)),sg.Text('km To',font=("微软雅黑",12)),
           sg.InputText('412',size=(5,1),key='Xmax',justification='left',font=("微软雅黑",12)),sg.Text('km (范围：385km-435km)',font=("微软雅黑",12))],
          [sg.Text('垂直距离范围',size=(6,1),justification='left',font=("微软雅黑",12),auto_size_text=True),
           sg.InputText('3391',size=(5,1),key='Ymin',justification='left',font=("微软雅黑",12)),sg.Text('km To',font=("微软雅黑",12)),
           sg.InputText('3394',size=(5,1),key='Ymax',justification='left',font=("微软雅黑",12)),sg.Text('km (范围：3375km-3420km)',font=("微软雅黑",12)),
           sg.Button('更新', key='submit',font=("微软雅黑",12)), sg.Quit('退出', key='q',font=("微软雅黑",12))],
          [sg.Text('地物类型索引数字编号说明：',font=("微软雅黑",8))],
          [sg.Text('1-海洋\t2-内陆湖泊\t3-湿地\t4-城郊开阔区域\t5-市区开阔区域\t6-道路开阔区域\t7-植被区\t8-灌木植被\t9-森林植被',font=("微软雅黑",8))],
          [sg.Text('10-城区超高层建筑(>60m)\t11-城市高层建筑(40m-60m)\t12-城市中高层建筑(20m-40m)\t13-城区<20m高密度建筑群',font=("微软雅黑",8))],
          [sg.Text('14-城区<20m多层建筑\t15-低密度工业建筑区域\t16-高密度工业建筑区域\t17-城郊\t18-发达城郊区域\t19-农村\t20-CBD商务圈',font=("微软雅黑",8))],
          [sg.Graph(canvas_size=(500, 400), graph_bottom_left=(-100, -100), background_color='white',
                    graph_top_right=(100, 100), key='graph1'),
           sg.Graph(canvas_size=(500, 400), graph_bottom_left=(-100, -100), background_color='white',
                    graph_top_right=(100, 100), key='graph2')]]
window = sg.Window('基站规划客户端', layout, resizable=True, size=(1050,650), location=(100,100), font='微软雅黑').Finalize()
try:
    window.FindElement("graph1").DrawImage('./Images/initmap2_Clutter.png', color='black', location=(-100, 100))
    window.FindElement("graph2").DrawImage('./Images/initmap2_Altitude.png', color='black', location=(-100, 100))
except Exception as e:
    print('程序报错：'+str(e))
data = pd.DataFrame([])
while True:
    event, value = window.Read()
    print(event)
    if event is 'submit':
        if data.shape[0]<1000:
            data = update(data,value)
        else:
            update(data,value)
    elif event is None or 'q':
        break
window.Close()
