import pandas as pd
import numpy as np
import os
from config import config
from pipeline.modelinference import ModelInference
import warnings
warnings.simplefilter('ignore', UserWarning)

# Script to label new data
mi = ModelInference()
def main():
    data = pd.read_csv(os.path.join(config.DATAPATH, 'test.csv'))
    data = data.iloc[:500]
    data = data.rename(columns={'tweet':'text'})
    print(data.shape)
    if (data is not None):
        # Parse the data through the predicted_output_category & predicted_probability, 
        # which scales and makes predictions
        preds = mi.predicted_output_category(data)
        data['label'] = preds
        data = data.rename(columns={'text':'raw_text'})
        data = data[['id', 'raw_text','label']]
        print(data.head())
        file = 'labelled_data/'
        if os.path.exists(file):
            data.to_csv('labelled_data/test_labelled_based_on_predictions.csv')
            print('Success')
        else:
            os.mkdir('labelled_data/')
            data.to_csv('labelled_data/test_labelled_based_on_predictions.csv')
            print('Success')


if __name__ == '__main__':
    main()