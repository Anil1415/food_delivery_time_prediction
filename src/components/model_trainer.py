# Baic import 
import numpy as np
import pandas as dp
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from src.exception import CustomException
from src.logger import logging


from src.utils import evaluate_model
from src.utils import save_object

from dataclasses import dataclass
import sys
import os

@dataclass
class Model_Trainer_Config:
    try:
        # assigning the training model path and file name as model.pkl

        trained_model_file_path = os.path.join('artifacts','model.pkl')
        logging.info('Model pickle file has been created')

    except Exception as e:
        
        logging.info('An occured while creating model file')

class Model_Trainer:
    def __init__(self) -> None:
        
        # Initialising the class which can be used for training the model
        self.model_trainer_config = Model_Trainer_Config()

    def initiate_model_training(self,train_array, test_array):

        try:

            logging.info('Splitting the dependent and independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                    train_array[:, :-1], 
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )

            # models to evaluate

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'SupportVectorRegressor': SVR(kernel='linear')
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n===========================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            print(f"Best Model score : {best_model_score}")
            print('\n===========================================================================\n')


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] #returns the key value
            print(f'Best Model Name : {best_model_name}')
            print('\n===========================================================================\n')

            best_model = models[best_model_name] # returns value of the key

            print(f'Best Model found, Model Name : {best_model_name} ,R2 score : {best_model_score}')
            print('\n===================================================\n')
            logging.info(f'Best Model found, Model Name : {best_model_name}, R2 score {best_model_score}')

            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        obj=best_model )

            

        except Exception as e:

            logging.info('exception occured at Model Training')
            raise CustomException(e,sys)


