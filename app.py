from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logging


application  = Flask(__name__)

app = application

@app.route('/anil', methods = ['POST'])
def home_page1():
    return "Anil"
    

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            delivery_person_age=float(request.form.get('delivery_person_age')),
            delivery_person_ratings = float(request.form.get('delivery_person_ratings')),
            weather_conditions = request.form.get('weather_conditions'),
            road_traffic_density = request.form.get('road_traffic_density'),
            vehicle_condition = int(request.form.get('vehicle_condition')),
            type_of_vehicle = request.form.get('type_of_vehicle'),
            multiple_deliveries = request.form.get('multiple_deliveries'),
            festival= request.form.get('festival'),
            city = request.form.get('city'),
            preparation_time_min = float(request.form.get('preparation_time_min')),
            distance_to_delivery_km = float(request.form.get('distance_to_delivery_km'))

        ) # initialise the CustomData class with object 'data'




        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)
        print(results)
        logging.info(f"Output value is : {results}")

        return render_template('results.html', final_result = results)
    





if __name__ == '__main__':
    app.run(host = '127.0.0.1', port= 8000, debug=True)