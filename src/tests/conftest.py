"""
conftest.py
"""

import pytest
import pandas as pd
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
from train import train
from pipeline.modelinference import ModelInference

mi=ModelInference(experiment_id=2)

@pytest.fixture
def dummy_data():
    data = pd.DataFrame({
    "id":[0, 1, 2],
    "text":
    ["Cause cause because YOU",
    "cause me @danieltovia1 and @user get to live together for a whole week!   #cantwaittocook ðð",
    "half way through the website now and #allgoingwell very  "
    ]
    })
    return data

@pytest.fixture
def get_make_features_data_type(dummy_data):
    actual_dtype = mi.make_features(dummy_data).dtypes.to_dict()
    return actual_dtype


@pytest.mark.skip(reason="Test this script locally")
@pytest.fixture
def get_train_metrics_dict():
    actual_dict = train.prepare_and_train()
    return actual_dict

@pytest.fixture
def get_make_features_columns():
    data = pd.DataFrame({
    "id":[0, 1, 2],
    "text":["Cause cause because YOU",
    "cause me @danieltovia1 and @user get to live together for a whole week!   #cantwaittocook ðð",
    "half way through the website now and #allgoingwell very  "]
    })
    actual_columns = mi.make_features(data).columns.tolist()
    return actual_columns

@pytest.fixture
def expect_df():
    return pd.DataFrame({
        'id': [0, 1, 2],
        'text': [
        "Cause cause because YOU",
        "cause me @danieltovia1 and @user get to live together for a whole week!   #cantwaittocook ðð",
        "half way through the website now and #allgoingwell very  "],
        'mention_count': [0, 2, 0],
        'char_count': [23, 80, 57],
        'word_count': [4, 13, 9],
        'uniq_word_count': [4, 13, 9],
        'htag_count': [0, 1, 1],
        'stopword_count': [1, 5, 5],
        'sent_count': [1, 2, 1],
        'avg_word_len': [4.6, 5.7, 5.7],
        'avg_sent_len': [2.0, 4.3, 4.5],
        'uniq_vs_words': [1.0, 1.0, 1.0],
        'stopwords_vs_words': [0.2, 0.4, 0.6],
        'title_word_count': [1, 0, 0],
        'uppercase_count': [1, 0, 0],
        'noun_count': [0, 3, 3],
        'verb_count': [0, 2, 0],
        'adj_count': [0, 1, 0],
        'adv_count': [1, 1, 3],
        'pron_count': [1, 1, 0],
        'cleaned_text': ["", "live week   canotwaittocook ", "half website  allgoingwell"]
    })

@pytest.fixture
def model_test_dummy_data():
    data = pd.DataFrame({
    "id":[0, 1, 2, 3, 4, 5, 6],
    "text":
    ["Cause cause because YOU",
    "suppo the #taiji fisherman! no bullying! no racism! #tweet4taiji #thecove #seashepherd",
    "interview feat grandmaster flash - ze lovely message â«âªâ«â«âºâº #nurap #nudisco #music #paris   â«âªâ«  via @user",
    "cause me @danieltovia1 and @user get to live together for a whole week!   #cantwaittocook ðð",
    "  i can't believe how much i used to care what people thought of me now i'm just like ""lol fuck u, fuck you and  wtf fuck youâ¦",
    "it could be worse. #embarrassed #unfounate #traumatized #bitches black lives should matter to other black people.",
    "half way through the website now and #allgoingwell very  "
    ]
    })
    return data

@pytest.fixture
def train_dummy_data():
    data = pd.DataFrame({
    "id":[0, 1, 2, 3, 4, 5, 6],
    "date":[
        "2022-07-15 15:47:23",
        "2022-07-15 15:58:10",
        "2022-07-15 20:18:10",
        "2022-07-16 20:18:10",
        "2022-07-16 15:47:23",
        "2022-07-16 15:58:10",
        "2022-07-16 20:18:10"
        ],
    "text":
    ["Cause cause because YOU",
    "suppo the #taiji fisherman! no bullying! no racism! #tweet4taiji #thecove #seashepherd",
    "interview feat grandmaster flash - ze lovely message â«âªâ«â«âºâº #nurap #nudisco #music #paris   â«âªâ«  via @user",
    "cause me @danieltovia1 and @user get to live together for a whole week!   #cantwaittocook ðð",
    "  i can't believe how much i used to care what people thought of me now i'm just like ""lol fuck u, fuck you and  wtf fuck youâ¦",
    "it could be worse. #embarrassed #unfounate #traumatized #bitches black lives should matter to other black people.",
    "half way through the website now and #allgoingwell very  "
    ]})
    return data
