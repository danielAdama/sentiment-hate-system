import pytest
import pandas as pd
import re
import sys
import json
import numpy as np
from nltk import sent_tokenize
sys.path.append('/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src')
from pipeline.modelinference import ModelInference
from config import config

stopwords = set(json.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/stopWords/custome_nltk_stopwords.json", "r")))
mi = ModelInference('vectorizerV2.bin', 'modelV2.bin')

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
def get_make_features_data_type():
    actual_dtype = mi.make_features(data).dtypes.to_dict()
    return actual_dtype

def test_make_feature_data_type(get_make_features_data_type):
    expected_dtype = {
        'id':np.dtype('int64'), 'text':np.dtype('O'), 'char_count':np.dtype('int64'),
        'word_count':np.dtype('int64'), 'uniq_word_count':np.dtype('int64'), 'htag_count':np.dtype('int64'), 
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
    "cause me and @user get to live together for a whole week!   #cantwaittocook ðð",
    "half way through the website now and #allgoingwell very  "]
    })
    actual_columns = mi.make_features(data).columns.tolist()
    return actual_columns

def test_make_feature_schema(get_make_features_columns):
    expected_schema = ['id', 'text','char_count', 'word_count', 'uniq_word_count', 'htag_count', 
    'stopword_count', 'sent_count', 'avg_word_len','avg_sent_len', 'uniq_vs_words', 'stopwords_vs_words',
    'title_word_count', 'uppercase_count','noun_count', 'verb_count', 'adj_count', 
    'adv_count', 'pron_count', 'cleaned_text']
    assert get_make_features_columns == expected_schema

@pytest.fixture
def dummy_data():
    data = pd.DataFrame({
    "id":[0, 1, 2],
    "text":["Cause cause because YOU","cause me and @user get to live together for a whole week!   #cantwaittocook ðð", "half way through the website now and #allgoingwell very  "]
    })
    return data

@pytest.mark.skip
def test_make_features(dummy_data):
    expect_df = pd.DataFrame([])
    expect_df['id'] = [0, 1, 2]
    expect_df['text'] = ["Cause cause because YOU","cause me and @user get to live together for a whole week!   #cantwaittocook ðð", "half way through the website now and #allgoingwell very  "]
    expect_df['char_count'] = [23, 84, 57]
    expect_df['word_count'] = [4, 14, 9]
    expect_df['uniq_word_count'] = [4, 14, 9]
    expect_df['htag_count'] = [0, 1, 1]
    expect_df['stopword_count'] = [1, 5, 5]
    expect_df['sent_count'] = [1, 2, 1]
    expect_df['avg_word_len'] = [4.6, 5.6, 5.7]
    expect_df['avg_sent_len'] = [2.0, 4.7, 4.5]
    expect_df['uniq_vs_words'] = [1.0, 1.0, 1.0]
    expect_df['stopwords_vs_words'] = [0.3, 0.4, 0.6]
    expect_df['title_word_count'] = [1, 0, 0]
    expect_df['uppercase_count'] = [1, 0, 0]
    expect_df['noun_count'] = [0, 4, 3]
    expect_df['verb_count'] = [0, 2, 0]
    expect_df['adj_count'] = [0, 2, 0]
    expect_df['adv_count'] = [1, 1, 3]
    expect_df['pron_count'] = [1, 1, 0]
    expect_df['cleaned_text'] = [" ","user live week   canotwaittocook", "half website  allgoingwell"]

    feats_df = mi.make_features(dummy_data)
    # pd.testing.assert_frame_equal() 
    assert feats_df == expect_df

def test_vector_process_shape(dummy_data):
    expect_shape = 1000
    vec_shape = mi.vector_process(dummy_data).shape[1]
    assert vec_shape == expect_shape

def test_merged_shape(dummy_data):
    expect_shape = 1019
    merge_shape = mi.merge(dummy_data).shape[1]
    assert merge_shape == expect_shape
    

data = pd.DataFrame({
    "id":[0, 1, 2],
    "text":["Cause cause because YOU","cause me and @user get to live together for a whole week!   #cantwaittocook ðð", "half way through the website now and #allgoingwell very  "]
    })
print(mi.merge(data).columns)

