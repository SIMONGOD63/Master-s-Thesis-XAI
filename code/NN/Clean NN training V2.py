#%% Import

#---- Packages
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras # Idk if required
#from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input,Dense, Dropout
from keras_tuner import Hyperband # Not working anymore

#---- Path and dataset

path = r"C:\Users\simeo\A. Data Science\Master thesis\01. Data"

X_train = pd.read_csv(path + "/00. FINALS/X_train_aligned.csv",index_col=0)
y_train = pd.read_csv(path + "/00. FINALS/y_train_aligned.csv",index_col=0)
X_test = pd.read_csv(path + "/00. FINALS/X_test_aligned.csv",index_col=0)
y_test= pd.read_csv(path + "/00. FINALS/y_test_aligned.csv",index_col=0)

#--- Remove the index
X_train.drop(columns = ["absolute_idx"],inplace= True);
X_test.drop(columns = ["absolute_idx"],inplace= True);
y_train.drop(columns = ["absolute_idx2"],inplace= True);
y_test.drop(columns = ["absolute_idx2"],inplace= True);

#--- Remove the raw response variable because 

#--- import a function 

def get_met(test, pred):
    
    # get the clean y_test
    test_number = test.filter(like ="op_intent_")
    c_test =np.array(test_number).argmax(axis = 1)
    
    # get the clean y_pred
    pred_number = pred.filter(like ="op_intent_")
    c_pred = np.array(pred_number).argmax(axis = 1)
    
    acc = accuracy_score(c_test, c_pred)
    #f1 = f1_score(c_test, c_pred)
    #f1 = 1
    return acc

#%% Test if the NN model works

#--- load the tuner
def build_model2(hp):
    inpt = Input(shape = (X_train.shape[1],))
    # 0
    l = Dense(256, activation = "tanh")(inpt)
    l = Dropout(0.2)(l)
    
    #1
    l = Dense(64,activation ="selu")(l)
    
    # 2
    l = Dense(256, activation = "softplus")(l)
    l= Dropout(0.5)(l)
    
    # 3
    l = Dense(192, activation = "selu")(l)
    l = Dropout(0.4)(l)
    
    # 4
    l = Dense(224,activation = "softplus")(l)
    l = Dropout(0.4)(l)
    
    # 5
    l = Dense(128, activation = "relu")(l)
    #○l = Dropout()()
    
    # 6 
    l = Dense(96, activation = "selu")(l)
    l = Dropout(0.3)(l)
    
    # 7 
    l = Dense(96, activation = "selu")(l)
    
    output = Dense(5,activation = "softmax", name = "output")(l)
    mod = keras.Model(inputs = [inpt], outputs = [output], name = "full_mod")
    
    lr = hp.Float("lr", min_value = 1e-4, max_value = 0.1, sampling = "log")
    
    mod.compile(optimizer =keras.optimizers.Adam(
        learning_rate = lr),
                loss = "categorical_crossentropy",
                metrics = ["accuracy"])
    return mod


HB_tuner2 = Hyperband(
    build_model2,
    objective = "val_loss",
    max_epochs= 100,
    hyperband_iterations=3,
    directory = path + "/hyperband/models",
    project_name = "model1-07-M1",
    max_consecutive_failed_trials=3,
    seed = 4243
    )

#-- Callbacks
#♣from tensorflow.keras.optimizers.schedules import ExponentialDecay

log_dir2 = path + "/hyperband/logs/logs01-07-M1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, histogram_freq=1)

reduce_lr = ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 10,
    min_lr = 0.001,
    cooldown = 5)

early_s = keras.callbacks.EarlyStopping('val_loss', patience=10,restore_best_weights = True)


#---- load the results of the tuner
HB_tuner2.search(
    X_train,y_train,
    validation_data =(X_test,y_test),
    #class_weight = c_dic2,
    epochs=100, 
    batch_size= 512, #arbitraly chosen
    callbacks=[reduce_lr,early_s,]) #tensorboard_callback2
best_hps2 = HB_tuner2.get_best_hyperparameters(num_trials=1)[0]
print(best_hps2.values)

model2 = HB_tuner2.hypermodel.build(best_hps2)

#%% Test performance oversampling


# check the dtypes
(X_train.dtypes == float).all()
(X_test.dtypes == float).all()

(y_test.iloc[:,1:].dtypes == (int | float)).all()
y_test.dtypes
y_train.dtypes


n_iter = 5
m2_acc = []


for i in range(n_iter):
    model2.fit(x = X_train.to_numpy(), y = y_train.iloc[:,1:].to_numpy(),
               batch_size = 512,
               epochs = 200,
               #class_weight= c_dic2,
               validation_data = (X_test.to_numpy(),y_test.iloc[:,1:].to_numpy()),
               callbacks = [early_s, reduce_lr] )
    
    
    y_pred = model2.predict(X_test)
    y_pred = y_pred.argmax(axis =1)
    y_true = y_test.iloc[:,1:].to_numpy().argmax(axis = 1)
    m2_acc.append(accuracy_score(y_true, y_pred) )
    

    
print("Model 2 LR :", round(np.mean(m2_acc)*100,4)) # 60.0 / 59.86 /// SS2 /// 59.91 /// 5 class SS ///  62.64 / 62.18 /// SS2 61.95 // 61.385
#%% download the model

model2.save(path + "/00. FINALS/last_model.keras")


