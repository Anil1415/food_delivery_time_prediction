import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
# from src.utils import save_
from src.utils import handle_column_name

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_objeect(self):
        try:
            logging.info('data transform initiated')
            # define which columns should be ordinal_encoded and which should be scaled

            categorical_columns = X.select_dtypes(include = 'object').columns
            numerical_columns = X.select_dtypes(exclude = 'object').columns

            # define custom ranking for each ordinal variable

            weather_category = ['Sunny','Fog','Cloudy','Windy','Sandstorms','Stormy']
            road_traffic_category = ['Low','Medium','High', 'Jam']
            vehilcle_type = ['electric_scooter','scooter','motorcycle']
            festival_category = ['No', 'Yes']
            city_category = ['Urban','Semi-Urban','Metropolitian']

            logging.info('pipeline initiated')

            ## Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                       ('imputer', SimpleImputer(strategy='median')),
                       ('scalar', StandardScaler())
                       ]
                    )
            ##  categorical_pipeline

            categorical_pipeline = Pipeline(
                steps= [
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ordinalencoder', OrdinalEncoder(categories=[weather_category,road_traffic_category,vehilcle_type,festival_category,city_category])),
                        ('scaler', StandardScaler())
                        ]
                    )
            
            ## preprocessor

            preprocessor = ColumnTransformer(
                            [
                             ('numerical_pipeline', numerical_pipeline, [i for i in numerical_columns]),
                             ('categorical_pipeline', categorical_pipeline, [i for i in categorical_columns])
                            ]
                        )   
            return preprocessor
            logging.info('Pipeline completed')

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,raw_data_path):
        try:
            # Read train and test data

            data  = pd.read_csv(raw_data_path)

            logging.info('Reading raw data completed')
            logging.info(f"Raw Data Frame Head: \n {data.head().to_string()}")

            logging.info("Data Cleaning, EDA and Feature Engineering started")
            # Handling column name
            obj_handling = handle_column_name()
            obj_handling.lower_column_names()
            obj_handling.fill_column_names()

            if data.duplicated().any():
                 # drop duplicates
                 data = data.drop_duplicates()
            else:
                 data

            # pattern match time format are kept and others are dropped


            data = data[~((~data.time_orderd.str.contains(r'[0-9]{2}:[0-9]{2}', na = False)) | (~data.time_order_picked.str.contains(r'[0-9]{2}:[0-9]{2}', na=False)))]
            data['time_order_picked'] = data.time_order_picked.str.replace('24','00')   

            


            


        except Exception as e:
            logging.info('Exception occured in the initiate data transformation')
            raise CustomException(e, sys)
        
            
