import pandas as pd
import numpy as np
import os
import json
import contractions
import pickle
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from string import punctuation
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ModelInference():

    """Model Inference class for making predictions on unseen dataset with the model.
    
    Attributes:
        lemmatizer (WordNet) : Lemmatize using WordNet's built-in morphy function.
        vectorizer (TFIDF matrix) : Vectorizer transforms a collection of raw documents to the 
        TF-IDF features matrix during training.
        scaler (float) : The scaled matrix of features during training.
        model (float) : Model to perform inference on new data.
        stopwords_json (set) : Loads the custome Stopwords with exception of "no", "not" & "but". We 
        remove stopwords before feeding it to the model for inference.
        stopwords (set) : Loads the custome Stopwords from NLTK with exception of "no", "not" & "but".
        We remove stopwords before feeding it to the model for inference.
        stopwords_punctuation (set) : A variable with the compined stopwords and punctions.
    """
    
    def __init__(self):

        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = pickle.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/vectorizers/vectorizer_label", "rb"))
        self.scaler = pickle.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/scalers/data_label_scaler", "rb"))
        self.model = pickle.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/models/model_labeller", "rb"))
        self.stopwords = set(json.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/stopWords/custome_nltk_stopwords.json", "r")))
        self.stopwords_json = set(json.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/stopWords/custome_json_stopwords.json", "r")))
        self.stopwords_punctuation = set.union(self.stopwords, self.stopwords_json, punctuation)

    def get_pos_tag(self, text):
        
        """Function to group words according to their Parts of Speech
        
        Args: 
            text (object) : Represents tweets
        Returns:
            tags (tuples) : Classified words and parts of speech
        """

        if type(text) == float or type(text) == int:
            print('Entry not valid')
            return ""

        tokens = [contractions.fix(i.lower()) for i in word_tokenize(str(text))]
        tags = pos_tag(tokens)

        return tags
    

    def preprocess_text(self, text):

        """Function to clean text for model inference.

        Args:
            text (object) : Represents tweets
        Returns:
            text (string) : Pre-processed text

        """

        sentence = []
        tags = self.get_pos_tag(text)
        for (token, tag) in tags:
            # Remove irrelevant symbols from token
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

            lemmatized_word = self.lemmatizer.lemmatize(token, position)
            if lemmatized_word not in self.stopwords_punctuation:
                sentence.append(lemmatized_word)
        text = ' '.join(sentence)
        text = text.replace("n't", 'not').replace('ii', '').replace('iii', '')
        text = text.replace("'s", "").replace("''", "").replace("nt", "not")
        
        return text

    def count_pos_tag(self, text, flags):
    
        """Function to check and count the respective parts of speech tags
        
        Args:
            text (object) : Represents tweets
            flags (string) : Representing Part of Speech (noun, verb, adj, pronoun)
        Returns:
            count (int) : Frequency of part of speech
            
        """
        
        pos_group = {
            'noun':['NN','NNS','NNP','NNPS'],
            'pron':['PRP','PRP$','WP','WP$'],
            'verb':['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj':['JJ','JJR','JJS'],
            'adv':['RB','RBR','RBS','WRB']
            }

        count=0
        tags = self.get_pos_tag(text)

        for (token, tag) in tags:
            token = re.sub(r"([0-9]+|[-_@./&+]+|``)", '', token)
            token = re.sub(r"(@[A-Za-z0-9_]+)|[^\w\s]|#|http\S+", '', token)
            token = token.encode("ascii", "ignore")
            token = token.decode()
            if tag in pos_group[flags]:
                count+=1
        
        return count
        
    def make_features(self, data):

        """Function that generates features for modelling.

        Args:
            data (DataFrame) : Input DataFrame that contains tweet
        Returns:
            data (DataFrame) : DataFrame with new features
        
        """
        
        data['char_count'] = data.text.apply(len)
        data['word_count'] = data.text.apply(lambda x: len(x.split()))
        data['uniq_word_count'] = data.text.apply(lambda x: len(set(x.split())))
        data['htag_count'] = data.text.apply(lambda x: len(re.findall(r'(#w[A-Za-z0-9]*)', x)))
        data['stopword_count'] = data.text.apply(lambda x: len([wrd for wrd in word_tokenize(x) if wrd in self.stopwords]))
        data['sent_count'] = data.text.apply(lambda x: len(sent_tokenize(x)))
        data['avg_word_len'] = data['char_count']/(data['word_count']+1)
        data['avg_sent_len'] = data['word_count']/(data['sent_count']+1)
        data['uniq_vs_words'] = data.uniq_word_count/data.word_count # Ratio of unique words to the total number of words
        data['stopwords_vs_words'] = data.stopword_count/data.word_count
        data['title_word_count'] = data.text.apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
        data['uppercase_count'] = data.text.apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

        data['noun_count'] = data.text.apply(lambda x: self.count_pos_tag(x, 'noun'))
        data['verb_count'] = data.text.apply(lambda x: self.count_pos_tag(x, 'verb'))
        data['adj_count'] = data.text.apply(lambda x: self.count_pos_tag(x, 'adj'))
        data['adv_count'] = data.text.apply(lambda x: self.count_pos_tag(x, 'adv'))
        data['pron_count'] = data.text.apply(lambda x: self.count_pos_tag(x, 'pron'))
        data['cleaned_text'] = data.text.apply(self.preprocess_text)

        return data

    # def raw_tweet(self, data):
    #     data['raw_text'] = data['text']
    #     return data['raw_text']

    def vector_process(self, data):

        """Function to convert preprocessed text to vector.
        
        Args:
            data (DataFrame) : Input DataFrame that contains tweet
        Returns:
            x (DataFrame) : Vectorized features
        
        """

        data = self.make_features(data)
        data = data.drop(['id', 'text'], axis=1)
        tfidf_feats = self.vectorizer.transform(data.cleaned_text).toarray()
        tfidfDF = pd.DataFrame(tfidf_feats, columns=self.vectorizer.get_feature_names())
        x = tfidfDF.merge(data, left_index=True, right_index=True)
        x = x.drop(['cleaned_text'], axis=1)
        
        return x

    def scale_process(self, data):

        """Function to scale vectorized text.

        Args:
            data (DataFrame) : Input DataFrame that contains tweet
        Returns:
            scaled_data (float) : scaled features
        
        """

        x = self.vector_process(data)
        scaled_data = self.scaler.transform(x)
        
        return scaled_data

    def predicted_probability(self, data):

        """Function that outputs model probability.
        
        Args:
            data (DataFrame) : Input DataFrame that contains tweet
        Returns:
            prob (float) : Models probability on data.

        """

        scaled_data = self.scale_process(data)
        if (scaled_data is not None):
            prob = self.model.predict_proba(scaled_data)[:,1]
            return prob

    def predicted_output_category(self, data):

        """Function that ouputs model category (0 for Non-hate or 1 for hate).
        
        Args:
            data (DataFrame) : Input DataFrame that contains tweet
        Returns:
            preds (float) : Models prediction on data.

        """

        scaled_data = self.scale_process(data)
        if (scaled_data is not None):
            # To ensure we make predictions even when a single user enters data
            preds = self.model.predict(scaled_data).reshape(-1, 1)
            return preds