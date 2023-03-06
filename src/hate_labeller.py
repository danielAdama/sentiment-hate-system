import pandas as pd
import numpy as np
import os
from config import config
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score,f1_score
from sklearn.metrics import auc, average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from time import time
from pipeline.modelinference import ModelInference
import warnings
warnings.simplefilter('ignore', UserWarning)


mi=ModelInference(experiment_id=2)
t1=time()
data = pd.read_csv(os.path.join(config.DATAPATH, 'test.csv'))
data = data.rename(columns={"tweet":"text"})
data = data.iloc[:1000]
print(data.shape)
print(data.head())
mi.predicted_output_category(data)
t2=time()
print(f"{(t2-t1):.4f} secs")

# evaluate model with new test data 'test.csv'
def test_model(X_test, y_test):
  y_pred = mi.predicted_output_category(X_test)
  test_prob = mi.predicted_probability(X_test)
  misclass = np.mean(y_pred != y_test)*100
  return {"auc": roc_auc_score(y_test, test_prob), "misclass":misclass}



#     data = data.rename(columns={'tweet':'text'})
#     print(data.shape)
#     if (data is not None):
#         # Parse the data through the predicted_output_category & predicted_probability, 
#         # which scales and makes predictions
#         data  = mi.predicted_output_category(data)[0]
#         preds = mi.predicted_output_category(data)[1]
#         data['label'] = preds
#         data = data.rename(columns={'text':'raw_text'})
#         data = data[['id', 'raw_text','label']]
#         print(data)
#         file = 'labelled_data/'
#         if os.path.exists(file):
#             data.to_csv('labelled_data/model4_test_labelled_based_on_predictions.csv')
#             print('Success')
#         else:
#             os.mkdir('labelled_data/')
#             data.to_csv('labelled_data/model4_test_labelled_based_on_predictions.csv')
#             print('Success')


# if __name__ == '__main__':
#     main()