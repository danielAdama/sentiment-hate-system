from src.config import config
import pandas as pd
import numpy as np
import os
import json
import contractions
import pickle
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from string import punctuation
import re
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)


racialWords = [
    'native', 'white trash', 'nigger', 'paki', 'chink', 
    'white boy', 'paki', 'whitey', 'pikey', 'nigga', 'spic', 'crow', 'squinty', 'wigga',
    'wetback', 'spick', 'gook'
    ]

vectorizer = pickle.load(open("src/vectorizers/vectorizer_label", "rb"))
scaler = pickle.load(open("src/scalers/data_label_scaler", "rb"))
model = pickle.load(open("src/models/model_labeller", "rb"))
stopwords_json = set(json.load(open("src/stopWords/stopwords_json.json", "r")))


data = pd.read_csv(os.path.join(config.DATAPATH, 'test.csv'))
data = data.iloc[:500]
data = data.rename(columns={'tweet':'text'})
print(data.shape)

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
# Remove stopwords to capture negativity in n-grams
stopwords.remove('no')
stopwords.remove('not')
stopwords.remove('but')
stopwords_json.remove('no')
stopwords_json.remove('not')
stopwords_json.remove('but')
# Combine the punctuation with both stopwords
stopwords_punctuation = set.union(stopwords, stopwords_json, punctuation)


def preprocess_text(text):
    
    """Function to clean text from irrelevant words and symbols"""
    
    if type(text) == float:
        print('Entry not valid')
        return ""
    sentence = []

    # Tokenize and lowercase all alphabet
    tokens = [contractions.fix(i.lower()) for i in word_tokenize(str(text))]
    
    # Part of speech
    tags = pos_tag(tokens)
    
    for (token, tag) in tags:
        # Remove all irrelevant symbols from token
        token = re.sub(r"([0-9]+|[-_@./&+]+|``)", '', token)
        token = re.sub(r"(@[A-Za-z0-9_]+)|[^\w\s]|#|http\S+", '', token)
        token = token.encode("ascii", "ignore")
        token = token.decode()
        
        # Grab the positions of the nouns(NN), verbs(VB), adverb(RB), and adjective(JJ)
        if tag.startswith('NN'):
            position = 'n'
        elif tag.startswith('VB'):
            position = 'v'
        elif tag.startswith('RB'):
            position = 'r'
        else:
            position = 'a'

        lemmatized_word = lemmatizer.lemmatize(token, position)
        if lemmatized_word not in stopwords_punctuation:
            sentence.append(lemmatized_word)
    final_sent = ' '.join(sentence)
    final_sent = final_sent.replace("n't", 'not').replace('ii', '').replace('iii', '')
    final_sent = final_sent.replace("'s", "").replace("''", "").replace("nt", "not")
    return final_sent

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

def feature(data):
    data['char_count'] = data.text.apply(len)
    data['word_count'] = data.text.apply(lambda x: len(x.split()))
    data['uniq_word_count'] = data.text.apply(lambda x: len(set(x.split())))
    data['htag_count'] = data.text.apply(lambda x: len(re.findall(r'(#w[A-Za-z0-9]*)', x)))
    data['stopword_count'] = data.text.apply(lambda x: len([wrd for wrd in word_tokenize(x) if wrd in stopwords]))
    data['sent_count'] = data.text.apply(lambda x: len(sent_tokenize(x)))
    data['avg_word_len'] = data['char_count']/(data['word_count']+1)
    data['avg_sent_len'] = data['word_count']/(data['sent_count']+1)
    data['uniq_vs_words'] = data.uniq_word_count/data.word_count # Ratio of unique words to the total number of words
    data['stopwords_vs_words'] = data.stopword_count/data.word_count
    data['title_word_count'] = data.text.apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    data['uppercase_count'] = data.text.apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    data['noun_count'] = data.text.apply(lambda x: count_pos_tag(x, 'noun'))
    data['verb_count'] = data.text.apply(lambda x: count_pos_tag(x, 'verb'))
    data['adj_count'] = data.text.apply(lambda x: count_pos_tag(x, 'adj'))
    data['adv_count'] = data.text.apply(lambda x: count_pos_tag(x, 'adv'))
    data['pron_count'] = data.text.apply(lambda x: count_pos_tag(x, 'pron'))
    data['cleaned_text'] = data.text.apply(preprocess_text)
    raw_text = data['text']
    data = data.drop(['id', 'text'], axis=1)

    tfidf_feats = vectorizer.transform(data.cleaned_text).toarray()
    tfidfDF = pd.DataFrame(tfidf_feats, columns=vectorizer.get_feature_names())
    x = tfidfDF.merge(data, left_index=True, right_index=True)
    x = x.drop(['cleaned_text'], axis=1)
    scaled_data = scaler.transform(x)
    return raw_text, scaled_data, x

def predicted_probability(scaled_data):
    if (scaled_data is not None):
        prob = model.predict_proba(scaled_data)[:,1]
        return prob

def predicted_output_category(scaled_data):
    if (scaled_data is not None):
        # To ensure we make predictions even when only a single user enters data
        preds = model.predict(scaled_data).reshape(-1, 1)
        return preds

def predicted_output(scaled_data):
    if (scaled_data is not None):
        raw_text, prepared_data, x = feature(scaled_data)
        data = pd.DataFrame(prepared_data, columns=x.columns)
        data['raw_text'] = raw_text
        data['predictions'] = predicted_output_category(prepared_data)
        data['probability'] = predicted_probability(prepared_data)
        data = data[['raw_text', 'predictions', 'probability']]
        print(data.head())
        data.to_csv('test_with_predictions.csv')

if __name__=="__main__":
    predicted_output(data)