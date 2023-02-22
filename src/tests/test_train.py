import pytest
import numpy as np
import pandas as pd
from train import train
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
import sys
sys.path.append('/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src')
from pipeline.modelinference import ModelInference
from config import config


@pytest.mark.skip(reason="Test this script locally")
def test_train_pipeline(get_train_metrics_dict):
    expected_dict = {
        'semi_auc':0.98316,
        'semi_misclass':5.29562,
        'train_auc': 0.99719, 
        'test_auc':0.98303, 
        'misclass':5.28681
    }
    
    assert get_train_metrics_dict == expected_dict


