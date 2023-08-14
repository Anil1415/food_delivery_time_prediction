import sys
import pandas as pd
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_Trainer





class PredictPipeline:

    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info("Exception has occured in prediction")
            raise CustomException(e,sys)
    

class CustomData:

    def __init__(self,
                 delivery_person_age:float,
                 delivery_person_ratings:float,
                 weather_conditions:str,
                 road_traffic_density:str,
                 vehicle_condition:int,
                 type_of_vehicle:str,
                 multiple_deliveries:str,
                 festival:str,
                 city:str,
                 preparation_time_min:float,
                 distance_to_delivery_km:float
                 ):
        
        self.delivery_person_age=delivery_person_age
        self.delivery_person_ratings=delivery_person_ratings
        self.weather_conditions=weather_conditions
        self.road_traffic_density= road_traffic_density
        self.vehicle_condition=vehicle_condition
        self.type_of_vehicle=type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.festival = festival
        self.city = city
        self.preparation_time_min = preparation_time_min
        self.distance_to_delivery_km = distance_to_delivery_km



    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'delivery_person_age':[self.delivery_person_age],
                'delivery_person_ratings':[self.delivery_person_ratings],
                'weather_conditions':[self.weather_conditions],
                'road_traffic_density':[self.road_traffic_density],
                'vehicle_condition':[self.vehicle_condition],
                'type_of_vehicle':[self.type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'festival':[self.festival],
                'city':[self.city],
                'preparation_time_min':[self.preparation_time_min],
                'distance_to_delivery_km':[self.distance_to_delivery_km]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame gathered')

            return df
           
        except Exception as e:
            logging.info('Exception occured in Prediction pipeline')
            raise CustomException(e,sys)
    

if __name__ == '__main__':
    obj = DataIngestion()

    train_data_path, test_data_path = obj.initiate_data_ingestion() 
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = Model_Trainer() # initilise the class
    model_trainer.initiate_model_training(train_arr, test_arr)
    data = CustomData(35,4,'Fog','High',1,'motorcycle',1.0,'No','Metropolitian', 15, 10)
    final_new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_new_data)
    results = round(pred[0],2)
    logging.info(f"Prediction value is {results}")
    print(results)


