import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify
from datetime import datetime
from config import config
from pipeline.modelinference import ModelInference
import warnings
warnings.simplefilter('ignore', UserWarning)

mi = ModelInference()
os.environ['FLASK_ENV']="development"
app = Flask(__name__)

data = pd.read_csv(os.path.join(config.DATAPATH, 'test.csv'))
data = data.iloc[:50]

# Read predictions from database
@app.route('/v1/users/predictions', methods=['GET'])
def many():
    data = pd.read_csv(os.path.join(config.DATAPATH, 'test.csv'))
    data = data.iloc[:50]
    data = data.rename(columns={'tweet':'text'})
    if (data.text is not None):
        # Parse the data through the predicted_output_category & predicted_probability, 
        # which scales and makes predictions
        preds = mi.predicted_output_category(data)
        prob = mi.predicted_probability(data)
        data['predictions'] = preds
        data['probability'] = prob
        data = data.rename(columns={'text':'raw_text'})
        data = data[['id', 'raw_text','predictions', 'probability']]
    return jsonify(outputs=data.to_json(orient="records")), 200


# Store in database
@app.route('/v1/batch/predictions', methods=['POST'])
def add_pred():
    data = pd.read_json(request.get_json())
    data = data.rename(columns={'tweet':'text'})
    print(data.shape)
    if (data.text is not None):
        # Parse the data through the predicted_output_category & predicted_probability, 
        # which scales and makes predictions
        preds = mi.predicted_output_category(data)
        prob = mi.predicted_probability(data)
        data['predictions'] = preds
        data['probability'] = prob
        data = data.rename(columns={'text':'raw_text'})
        data = data[['id', 'raw_text','predictions', 'probability']]
        res = {
            'id':data['id'].tolist(), 'tweet':data['raw_text'].tolist(), 
            'predictedLabel': data['predictions'].tolist(), 
            'probabilityScore': data['probability'].tolist()
        }
        content = [dict(zip(res.keys(), i)) for i in zip(*res.values())]
        # predictions=data.to_json(orient="records")
    return jsonify({"outputs":content}), 200

# Real-time prediction
@app.route('/v1/single-entry/predict', methods=['POST'])
def single():
    # data = pd.read_json(request.get_json())
    data = pd.DataFrame(request.get_json())
    data = data.rename(columns={'tweet':'text'})
    if (data.text is not None):
        # Parse the data through the predicted_output_category & predicted_probability, 
        # which scales and makes predictions
        preds = mi.predicted_output_category(data)
        prob = mi.predicted_probability(data)
        data['predictions'] = preds
        data['probability'] = prob
        data = data.rename(columns={'text':'raw_text'})
        data = data[['id', 'raw_text','predictions', 'probability']]
        result = {
            "creation_time":datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
            "id": data['id'].to_json(orient="records"),
            "tweet": data['raw_text'].to_json(orient="records"),
            "Prediction":{
                "Details":{
                "PredictiveModelType":"Binary:0-non-hate, 1-hate",
                "Algorithm":"XGBoost"
            },
            "PredictedLabel": data['predictions'].to_json(orient="records"),
            "PredictedScore": data['probability'].to_json(orient="records")
            },
        }
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True, port=2626)
