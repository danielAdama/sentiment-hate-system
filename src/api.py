import pandas as pd
import numpy as np
import os
import logging
from functools import wraps
import argparse
import json
from quart import Quart, request, jsonify
from quart_jwt_extended import (
    JWTManager,
    jwt_required,
    create_access_token,
    get_jwt_identity
)
from datetime import datetime
from config import config
from utilities import encoders
from pipeline.modelinference import ModelInference
import warnings
warnings.simplefilter('ignore', UserWarning)


app = Quart(__name__)
jwt = JWTManager(app)
logger = logging.getLogger('asyncio')
logger.setLevel(level=logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(level=logging.DEBUG)
logger.addHandler(consoleHandler)

app.config['modelInference'] = ModelInference(experiment_id=2, run_id="3bff3d4e681b4574872d26ecb645173a")
app.config['SECRET_KEY'] = config.keys['SECRET_KEY']
app.config['USERNAME'] = config.keys["USERNAME"]
app.config['PASSWORD'] = config.keys["PASSWORD"]


@app.route('/api/v1/login', methods = ["POST"])
async def login():
    if not request.is_json:
        return jsonify({
                "Message": "Missing JSON in request",
                "Data": None,
                "Error": "Bad request"
            }), config.HTTP_400_BAD_REQUEST
    try:
        username = (await request.get_json()).get("username", None)
        password = (await request.get_json()).get("password", None)
        if not await request.get_json():
            return jsonify({
                "Message": "Please provide login credentials",
                "Data": None,
                "Error": "Bad request"
            }), config.HTTP_400_BAD_REQUEST
        
        if username == app.config['USERNAME'] and password == app.config['PASSWORD']:
            try:
                token = create_access_token(identity=username)

                return jsonify({
                    "Message": "Successfully fetched authentication token",
                    "User": app.config['USERNAME'],
                    "Token":token
                }), config.HTTP_200_OK

            except Exception as ex:
                return {
                    "Message":str(ex),
                    "Error":"Something went wrong"
                }, config.HTTP_500_INTERNAL_SERVER_ERROR
        
        return jsonify({
            "Message": "Error fetching authentication token! invalid username or password",
            "Data": None,
            "Error": "Unauthorized"
        }), config.HTTP_404_NOT_FOUND
    except Exception as ex:
        return {
            "Message":str(ex),
            "Error":"Something went wrong",
            "Data":None
        }, config.HTTP_500_INTERNAL_SERVER_ERROR


@app.route('/api/v1/predict', methods = ["POST"])
@jwt_required
async def predict():
    try:
        query = await request.get_json() or {
            "id": 21,
            "text": "Hello World"
        }
        
        if not query:
                logger.debug("API PREDICT > please enter a text")
                return jsonify({
                    "BaseResponse":{
                            "Status":False,
                            "Message": "Please enter a text",
                            "Data": None,
                        }, 
                    "Error": "Bad request"
                }), config.HTTP_400_BAD_REQUEST
    
        data = pd.DataFrame(query, index=[0])
        payload = {}
        if (data.text is not None):
            payload['prediction'] = []
            detection = {}
            try:
                pred, model_version = app.config['modelInference'].predicted_output_category(data)
                prob = app.config['modelInference'].predicted_probability(data)
                if pred == 0:
                    label = 'noHate'
                else:
                    label = 'Hate'
                
                detection['label'] = label
                detection['confidence'] = prob
                detection['model_version'] = model_version
                payload['prediction'].append(detection)

            except Exception as ex:
                logger.debug(f"API PREDICT > {ex}")
                return jsonify({
                    "BaseResponse":{
                        "Status":False,
                        "Message":str(ex)
                        },
                    "Error":"Something went wrong"
                }), config.HTTP_404_NOT_FOUND

        return jsonify({
            "BaseResponse":{
                "Status":True,
                "Messsage":"Operation successfully",
                "User": get_jwt_identity()
            },
            "query":query.get('text'),
            "results": encoders.encode_to_json(payload, as_py=True)
        }), config.HTTP_200_OK

    except Exception as ex:
            logger.debug(f"API PREDICT > APPLICATION ERROR while predicting text - {ex}")
            return jsonify({
                "BaseResponse":{
                    "Status":False,
                    "Message":str(ex)
                },
                "Error":"Something went wrong",
            }), config.HTTP_500_INTERNAL_SERVER_ERROR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hate Detector Api exposing sentiment model")
    parser.add_argument("-p", "--port", default=8080, type=int, help="port number")
    args = parser.parse_args()
    app.run(host='0.0.0.0', debug=False, port=args.port)
