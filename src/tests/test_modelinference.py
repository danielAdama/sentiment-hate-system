import pytest
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from pipeline.modelinference import ModelInference
from config import config

mi = ModelInference('vectorizerV3.bin', 'modelV3.bin')

def test_get_pos_tag():
    pos = mi.get_pos_tag("half way through the website now")
    expect = [('half', 'NN'), ('way', 'NN'), ('through', 'IN'), ('the', 'DT'), ('website', 'NN'), ('now', 'RB')]
    assert pos == expect

def test_preprocess_text():
    fun = mi.preprocess_text("We need to be testing our code because it boost confidence")
    expect = "test code boost confidence"
    assert fun == expect

def test_count_pos_tag():
    fun = mi.count_pos_tag("half way through the website now and #allgoingwell very  ", 'noun')
    expect = 3
    
    assert fun == expect

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

def test_make_feature_data_type(get_make_features_data_type):
    expected_dtype = {
        'mention_count':np.dtype('int64'),
        'char_count':np.dtype('int64'),'word_count':np.dtype('int64'),
        'uniq_word_count':np.dtype('int64'), 'htag_count':np.dtype('int64'), 
        'stopword_count':np.dtype('int64'), 'sent_count':np.dtype('int64'), 
        'avg_word_len':np.dtype('float64') ,'avg_sent_len':np.dtype('float64'),
        'uniq_vs_words':np.dtype('float64'), 'stopwords_vs_words':np.dtype('float64'),
        'title_word_count':np.dtype('int64'), 'uppercase_count':np.dtype('int64'),
        'noun_count':np.dtype('int64'), 'verb_count':np.dtype('int64'),
        'adj_count':np.dtype('int64'), 'adv_count':np.dtype('int64'),
        'pron_count':np.dtype('int64'), 'cleaned_text':np.dtype('O')
    }
    assert get_make_features_data_type == expected_dtype

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

def test_make_feature_schema(get_make_features_columns):
    expected_schema = ['mention_count', 'char_count', 'word_count', 'uniq_word_count', 'htag_count', 
    'stopword_count', 'sent_count', 'avg_word_len','avg_sent_len', 'uniq_vs_words', 'stopwords_vs_words',
    'title_word_count', 'uppercase_count','noun_count', 'verb_count', 'adj_count', 
    'adv_count', 'pron_count', 'cleaned_text']
    assert get_make_features_columns == expected_schema

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

def test_make_features_integer(dummy_data, expect_df):
    expect_df_int = expect_df[['mention_count', 'char_count', 'word_count', 'uniq_word_count', 'htag_count', 
    'title_word_count', 'uppercase_count', 'stopword_count', 'sent_count', 'adj_count', 'adv_count',
    'noun_count', 'verb_count', 'pron_count']]
    
    feats_df = mi.make_features(dummy_data)
    actual_df_int = feats_df[['mention_count', 'char_count', 'word_count', 'uniq_word_count', 'htag_count', 
    'title_word_count', 'uppercase_count', 'stopword_count', 'sent_count', 'adj_count', 'adv_count',
    'noun_count', 'verb_count', 'pron_count']]
    actual_df_int.equals(expect_df_int)

def test_make_features_avg_word_len_float(dummy_data, expect_df):
    expect_df_float = expect_df['avg_word_len']
    feats_df = mi.make_features(dummy_data)
    actual_df_float = feats_df['avg_word_len']
    assert pytest.approx(actual_df_float, 0.1) == expect_df_float

def test_make_features_avg_sent_len_float(dummy_data, expect_df):
    expect_df_float = expect_df['avg_sent_len']
    feats_df = mi.make_features(dummy_data)
    actual_df_float = feats_df['avg_sent_len']
    assert pytest.approx(actual_df_float, 0.1) == expect_df_float

def test_make_features_uniq_vs_words_float(dummy_data, expect_df):
    expect_df_float = expect_df['uniq_vs_words']
    feats_df = mi.make_features(dummy_data)
    actual_df_float = feats_df['uniq_vs_words']
    assert pytest.approx(actual_df_float, 0.1) == expect_df_float

def test_make_features_stopwords_vs_words_float(dummy_data, expect_df):
    expect_df_float = expect_df['stopwords_vs_words']
    feats_df = mi.make_features(dummy_data)
    actual_df_float = feats_df['stopwords_vs_words']
    assert pytest.approx(actual_df_float, 0.1) == expect_df_float

def test_make_features_string(dummy_data, expect_df):
    expect_df_str = expect_df['cleaned_text']
    feats_df = mi.make_features(dummy_data)
    actual_df_str = feats_df['cleaned_text']
    actual_df_str.equals(expect_df_str)

def test_transform_process_shape_is_1000(dummy_data):
    expect_shape = 1000
    vec_shape = mi.transform(dummy_data, 'cleaned_text').shape[1]
    assert vec_shape == expect_shape

def test_merged_shape_is_1018(dummy_data):
    expect_shape = 1018
    merge_shape = mi.merge(dummy_data).shape[1]
    assert merge_shape == expect_shape

def test_predicted_probability(dummy_data):
    expect = [0.1861, 0.0726, 0.0782]
    actual = mi.predicted_probability(dummy_data)
    assert pytest.approx(actual, 0.1) == expect

def test_predicted_category_of_dummy_data_result_are_all_0(dummy_data):
    expect = np.array([0, 0, 0]).reshape(-1, 1)
    actual = mi.predicted_output_category(dummy_data)
    np.array_equal(actual, expect)