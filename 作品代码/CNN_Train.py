from keras import Input,Model
from keras import layers
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv1D
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import models
from keras.optimizers import RMSprop
from numpy import random as rd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
import datetime
from keras.callbacks import ReduceLROnPlateau
from keras.losses import mean_squared_error as MSE
from keras.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_model(train_data):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    #*********************CNN****************************
    model = models.Sequential()
    model.add(layers.Conv1D(64,5,activation='relu',padding='same',input_shape=(train_data.shape[1],1)))
    model.add(layers.Conv1D(64, 5, activation='relu',padding='same'))
    model.add(layers.Conv1D(64, 5, activation='relu',padding='same'))
    model.add(layers.Flatten())
    model.add(Dropout(0.5))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(1))
    return model

def standardize(train,test):
    mean = np.mean(train,axis=0)
    std = np.std(train,axis=0)
    np.savetxt('ModelsToAnalysis/mean.txt',mean)
    np.savetxt('ModelsToAnalysis/std.txt', std)
    X_train = (train-mean)/std
    X_test = (test-mean)/std
    return X_train,X_test

df = pd.read_hdf('alldata_RSRP.h5',key='data')
y = df['RSRP']
df = df.drop(['RSRP'],axis=1)
X = df[[col for col in df.columns]]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=23)
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)
[X_train,X_test] = standardize(X_train,X_test)

# X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

X_train = X_train.values.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.values.reshape((X_test.shape[0],X_test.shape[1],1))
y_train = y_train.values.reshape((y_train.shape[0],1))
y_test = y_test.values.reshape((y_test.shape[0],1))
train_targets = y_train
train_data = X_train
test_data = X_test
test_targets = y_test

num_epochs = 100
# Build the Keras model (already compiled)
model = build_model(train_data)
rmsprop = RMSprop(lr=0.0001, decay=1e-6)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
# Train the model (in silent mode, verbose=0)
stopping = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
start = datetime.datetime.now()
history = model.fit(train_data, train_targets,
                    validation_data=(test_data, test_targets),
            epochs=num_epochs, batch_size=128, verbose=2,shuffle=True,callbacks=[stopping])
end = datetime.datetime.now()
print(end-start)
model.summary()

modelName = './ModelsToAnalysis' + '/' + 'CNN_Model' + '.h5'

historyName =  './ModelsToAnalysis' + '/' + 'CNN_Model' + '.pkl'
model.save(modelName)
file_to_save_history = open(historyName, 'wb')
pickle.dump(history.history, file_to_save_history)
file_to_save_history.close()