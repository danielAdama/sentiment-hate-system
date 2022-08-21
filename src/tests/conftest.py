import pytest
import pandas as pd
from nltk import sent_tokenize
from pipeline.modelinference import ModelInference

mi = ModelInference('vectorizerV3.bin', 'modelV3.bin')

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
    df = pd.DataFrame([])
    df['id'] = [0, 1, 2]
    df['text'] = [
        "Cause cause because YOU",
        "cause me @danieltovia1 and @user get to live together for a whole week!   #cantwaittocook ðð",
        "half way through the website now and #allgoingwell very  "
    ]
    df['mention_count'] = [0, 2, 0]
    df['char_count'] = [23, 80, 57]
    df['word_count'] = [4, 13, 9]
    df['uniq_word_count'] = [4, 13, 9]
    df['htag_count'] = [0, 1, 1]
    df['stopword_count'] = [1, 5, 5]
    df['sent_count'] = [1, 2, 1]
    df['avg_word_len'] = [4.6, 5.6, 5.7]
    df['avg_sent_len'] = [2.0, 4.7, 4.5]
    df['uniq_vs_words'] = [1.0, 1.0, 1.0]
    df['stopwords_vs_words'] = [0.25, 0.35, 0.56]
    df['title_word_count'] = [1, 0, 0]
    df['uppercase_count'] = [1, 0, 0]
    df['noun_count'] = [0, 3, 3]
    df['verb_count'] = [0, 2, 0]
    df['adj_count'] = [0, 1, 0]
    df['adv_count'] = [1, 1, 3]
    df['pron_count'] = [1, 1, 0]
    df['cleaned_text'] = ["", "live week   canotwaittocook ", "half website  allgoingwell"]

    return df