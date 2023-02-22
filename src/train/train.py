#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import json
import contractions
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score,f1_score
from sklearn.metrics import auc, average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from string import punctuation
import re
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
import sys
# Path to the module (ModelInference) and config
sys.path.append('/home/daniel/programming/NLP/sentiment-hate-system/src')
from pipeline.modelinference import ModelInference
from config import config


mi = ModelInference()

def merge(tfidf, data_to_merge):
    data = tfidf.merge(data_to_merge, left_index=True, right_index=True)
    data = data.drop(['cleaned_text'], axis=1)
    return data

def count_pos_tag(text, flags):
    
    """Function to check and count the respective parts of speech tags"""
    
    pos_group = {
    'noun':['NN','NNS','NNP','NNPS'],
    'pron':['PRP','PRP$','WP','WP$'],
    'verb':['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj':['JJ','JJR','JJS'],
    'adv':['RB','RBR','RBS','WRB']
    }

    count=0
    tokens = [contractions.fix(i.lower()) for i in word_tokenize(text)]
    tags = pos_tag(tokens)

    for (token, tag) in tags:
        token = re.sub(r"([0-9]+|[-_@./&+]+|``)", '', token)
        token = re.sub(r"(@[A-Za-z0-9_]+)|[^\w\s]|#|http\S+", '', token)
        token = token.encode("ascii", "ignore")
        token = token.decode()
        if tag in pos_group[flags]:
            count+=1
    return count

def make_features(data):

    stopwords = set(json.load(open(os.path.join(os.getcwd(),"src/stopWords/custome_nltk_stopwords.json"), "r")))
    
    data['noun_count'] = data.tweet.apply(lambda x: count_pos_tag(x, 'noun'))
    data['verb_count'] = data.tweet.apply(lambda x: count_pos_tag(x, 'verb'))
    data['adj_count'] = data.tweet.apply(lambda x: count_pos_tag(x, 'adj'))
    data['adv_count'] = data.tweet.apply(lambda x: count_pos_tag(x, 'adv'))
    data['pron_count'] = data.tweet.apply(lambda x: count_pos_tag(x, 'pron'))

    data['char_count'] = data.tweet.apply(len)
    data['word_count'] = data.tweet.apply(lambda x: len(x.split()))
    data['uniq_word_count'] = data.tweet.apply(lambda x: len(set(x.split())))
    data['htag_count'] = data.tweet.apply(lambda x: len(re.findall(r'#[\w\-]+', x)))
    data['stopword_count'] = data.tweet.apply(lambda x: len([wrd for wrd in word_tokenize(x) if wrd in stopwords]))
    data['sent_count'] = data.tweet.apply(lambda x: len(sent_tokenize(x)))
    data['avg_word_len'] = data['char_count']/(data['word_count']+1)
    data['avg_sent_len'] = data['word_count']/(data['sent_count']+1)
    data['uniq_vs_words'] = data.uniq_word_count/(data.word_count+1) # Ratio of unique words to the total number of words
    data['stopwords_vs_words'] = data.stopword_count/(data.word_count+1)
    data['title_word_count'] = data.tweet.apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    data['uppercase_count'] = data.tweet.apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    data = data.drop(['id', 'tweet'], axis=1)
    return data

def basic_prep(data):
    data['mention_count'] = data.tweet.apply(lambda x: len(re.findall(r"@[\w\-]+", x)))
    data['tweet'] = data.tweet.replace(regex=re.compile(r"@([A-Za-z0-9_]+)"), value='')
    data['tweet'] = data.tweet.replace(regex=re.compile(r"RT([\s:]+)"), value='')
    return data

def transform(data_cleaned, unlabelled_cleaned):
    tfVectorizer = TfidfVectorizer(sublinear_tf=True, 
                               min_df=5, norm='l2', encoding='latin-1', max_features=1000,
                               ngram_range=(1, 2))
    
    tfVectorizer.fit(data_cleaned.cleaned_text)
    train_tfidf_feat = tfVectorizer.transform(data_cleaned.cleaned_text).toarray()
    unlabelled_tfidf_feat = tfVectorizer.transform(unlabelled_cleaned.cleaned_text).toarray()
    train_tfidf = pd.DataFrame(train_tfidf_feat, columns=tfVectorizer.get_feature_names_out())
    unlabelled_tfidf = pd.DataFrame(unlabelled_tfidf_feat, columns=tfVectorizer.get_feature_names_out())
    return tfVectorizer, train_tfidf, unlabelled_tfidf

def prepare_and_train(train_data='labelled.parquet.gzip',
                        unlabelled_data='unlabelled.parquet.gzip',
                        alpha=0.1,
                        lr=0.1,
                        estimators=400,obj='binary',bal={0:1,1:1}):
    
    fixed_params = {
        'objective': 'binary',
        'metric':'auc',
        'is_unbalance':True,
        'force_col_wise':True,
        'feature_pre_filter':False,
        'boosting':'dart',
        'num_boost_round':1200,
        'early_stopping_rounds':50,
        'num_threads':4
    }
                    
    params = {
        'feature_fraction':0.8,
        'lambda_l1':0.3970341857380282,
        'lambda_l2':0.20506586286248293,
        'learning_rate':0.02624259084185803,
        'max_depth':61,
        'min_data_in_leaf':31,
        'num_leaves':138,
        'seed':43
    }

    unlabelled_data = pd.read_parquet(os.path.join(os.getcwd(),'src/train',unlabelled_data))
    labelled_data = pd.read_parquet(os.path.join(os.getcwd(),'src/train',train_data))
    labelled_data = labelled_data.drop('id', axis=1)
    labelled_data = labelled_data.rename(columns={'index':'id'})

    data_cleaned = labelled_data.copy()
    data_cleaned = basic_prep(data_cleaned)
    data_cleaned['cleaned_text'] = data_cleaned.tweet.apply(mi.preprocess_text)

    unlabelled_cleaned = unlabelled_data.copy()
    unlabelled_cleaned = basic_prep(unlabelled_cleaned)
    unlabelled_cleaned['cleaned_text'] = unlabelled_cleaned.tweet.apply(mi.preprocess_text)
    _, data_train_tfidf, unlabelled_tfidf = transform(data_cleaned, unlabelled_cleaned)
    
    data_cleaned = make_features(data_cleaned)
    unlabelled_cleaned = make_features(unlabelled_cleaned)
    
    data_cleaned = merge(data_train_tfidf, data_cleaned)
    unlabelled_cleaned = merge(unlabelled_tfidf, unlabelled_cleaned)
    
    target_labelled_data = data_cleaned.label
    data_cleaned = data_cleaned.drop(['label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data_cleaned.values, target_labelled_data.values,
                                                        test_size=0.2, random_state=43)
    X_train_bls, y_train_bls = BorderlineSMOTE(random_state = 154).fit_resample(X_train, y_train)
    semi_sup_model = make_pipeline(StandardScaler(), lgb.LGBMClassifier(n_estimators=estimators, 
                                                                        reg_alpha=alpha, 
                                                                        learning_rate=lr,
                                                                        objective=obj,
                                                                        class_weight=bal))
    
    semi_sup_model.fit(X_train_bls, y_train_bls)
    pred = semi_sup_model.predict_proba(X_test)[:, 1]
    y_pred = semi_sup_model.predict(X_test)
    semi_auc = roc_auc_score(y_test, pred)
    semi_misclass = np.mean(y_pred != y_test)*100
    
    unlabelled_cleaned = unlabelled_cleaned.values
    probs = semi_sup_model.predict_proba(unlabelled_cleaned)
    preds = semi_sup_model.predict(unlabelled_cleaned)
    
    df_pseudo = pd.DataFrame(probs, columns = ['C1Prob', 'C2Prob']) 
    df_pseudo['lab']=preds
    df_pseudo['max']=df_pseudo[["C1Prob", "C2Prob"]].max(axis=1)
    
    conf_ind=df_pseudo["max"]>0.979
    X_train_new = np.append(X_train, unlabelled_cleaned[conf_ind,:],axis=0)
    y_train_new = np.append(y_train, df_pseudo.loc[conf_ind, ['lab']])
    
    lgb_train = lgb.Dataset(X_train_new, label=y_train_new)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    
    lgb_model = lgb.train({**params,**fixed_params}, lgb_train,
                          valid_sets=[lgb_train, lgb_test],
                          valid_names=['train','valid'],
                          num_boost_round=fixed_params["num_boost_round"],
                          early_stopping_rounds=fixed_params["early_stopping_rounds"])
                          
    train_prob = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
    test_prob = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_pred = mi.binary_predict(test_prob)

    metrics = {
        "semi_auc":round(semi_auc, 5),
        "semi_misclass":round(semi_misclass, 5),
        "train_auc":round(roc_auc_score(y_train, train_prob), 5),
        "test_auc":round(roc_auc_score(y_test, test_prob), 5),
        "misclass":round(np.mean(y_pred != y_test)*100, 5)
    }
    
    return metrics

# if __name__ == "__main__":
#     print(prepare_and_train())

