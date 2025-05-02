# Combined and edited from Sam's code FalsePositiveModel.ipynb and FalseNegativeModel.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def prepare_data(file):

    # read data
    data = pd.read_csv(file, na_values=['NULL'])
    data['AcquisitionDateTime_DT'] = pd.to_datetime(data['AcquisitionDateTime_DT'])
    prediction = data['MI_Phys'] 

    drop_cols =  ['PatientID', 
                  '12SL_Codes', 
                  'Phys_Codes',
                  'TestID', 
                  'Source', 
                  'Gender',
                  'PatientAge', 
                  'AcquisitionDateTime_DT',
                   'MI_Phys']   
    
    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=drop_cols), 
            prediction, 
            test_size=0.2, 
            random_state=42)
    
    # drop the 12SL column
    X_test = X_test.drop(columns=['MI_12SL'])
    X_train = X_train.drop(columns=['MI_12SL'])

    return X_train, X_test, y_train, y_test

def train_lgb(X_train, y_train, type):

    # train model
    scale_pos_weight = (5419/4003) if type == "falpos" else (63927/3968)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 200,
        'learning_rate': 0.15,
        'random_state': 42,
        'verbose': -1,
        'num_leaves': 127,
        'scale_pos_weight': scale_pos_weight,
        }
    model = (lgb.LGBMClassifier(**lgb_params))
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):

    # get f1 score
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')

    return score

# false positive model 
falpos_X_train, falpos_X_test, falpos_y_train, falpos_ytest = prepare_data('positive.csv')
falpos_model = train_lgb(falpos_X_train, falpos_y_train, "falpos")
falpos_model.booster_.save_model("falpos.pkl") # save false positive model
falpos_score = evaluate_model(falpos_model, falpos_X_test, falpos_ytest)
print("False positive model score: ", falpos_score)

# false negative model
falneg_X_train, falneg_X_test, falneg_y_train, falneg_ytest = prepare_data('negative.csv')
falneg_model = train_lgb(falneg_X_train, falneg_y_train, "falneg")
falneg_model.booster_.save_model('falneg.pkl') # save false negative model
falneg_score = evaluate_model(falneg_model, falneg_X_test, falneg_ytest)
print("False negative model score: ", falneg_score)