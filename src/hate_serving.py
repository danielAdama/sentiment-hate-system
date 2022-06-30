import pandas as pd
import numpy as np
import os
from config import config
from pipeline.modelinference import ModelInference
import warnings
warnings.simplefilter('ignore', UserWarning)

# Script for performing inference on new data

mi = ModelInference()
# def main():
#     data = pd.read_csv(os.path.join(config.DATAPATH, 'test.csv'))
#     data = data.iloc[:500]
#     data = data.rename(columns={'tweet':'text'})
#     print(data.shape)
#     if (data is not None):
#         # Parse the data through the predicted_output_category & predicted_probability, 
#         # which scales and makes predictions
#         preds = mi.predicted_output_category(data)
#         prob = mi.predicted_probability(data)
#         data['predictions'] = preds
#         data['probability'] = prob
#         data = data.rename(columns={'text':'raw_text'})
#         data = data[['id', 'raw_text','predictions', 'probability']]
#         print(data.head())
#         data.to_csv('test_with_predictions.csv')


# if __name__ == '__main__':
#     main()