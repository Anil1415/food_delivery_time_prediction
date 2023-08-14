import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

# u can code here - read db, common functions


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


  
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)


class handle_column_name:

    def __init__(self) -> None:
        pass

    def lower_column_names(self):
        for features in df.columns:
            df.columns = df.columns.str.lower()
        return df.columns
    def fill_column_names(self):
        for feature in df.columns:
            df.columns = df.columns.str.replace(" ","_")
        return df.columns
    
    def haversine_np(lon1, lat1, lon2, lat2):
        """ Calculate the great circle distance between two points on the earth (specified in decimal degrees) All args must be of equal length"""
        
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dist_lon = lon2-lon1
        dist_lat = lat2-lat1
        
        a = np.sin(dist_lat/2)**2 + np.cos(lat1) *np.cos(lat2) * np.sin(dist_lon/2)**2
        
        c = 2 * np.arcsin(np.sqrt(a))
        
        km = 6378 * c
        
        return km
    

