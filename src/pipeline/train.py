#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
import json
import datetime
import contractions
import pickle
import mysql.connector
from mysql.connector import Error
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score,f1_score
from sklearn.metrics import auc, average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from string import punctuation
import re
import seaborn as sns
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
import sys
# Path to the module (ModelInference) and config
# sys.path.append('/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src')
from pipeline.modelinference import ModelInference
from config import config

mi = ModelInference('vectorizerV3.bin', 'modelV3.bin')

class ToPandasDF():
    def __init__(self, password, host, database, user):

        self.password = password
        self.host = host
        self.database = database
        self.user = user
        
    
    def MySQLconnect(self, query):
        
        try:
            connection = mysql.connector.connect(host=self.host, 
                                                 database=self.database, 
                                                 password=self.password,
                                                 user=self.user)

            if connection.is_connected():

                print("Successfully connected to the database\n")

                cursor = connection.cursor()
                query = query
                cursor.execute(query)

                data = cursor.fetchall()

                df = pd.DataFrame(data, columns = ['id', 'date', 'tweet'])
        except Error as e:
            print(e)
            
        cursor.close()
        connection.close()
        
        return df
    
    def check_if_valid_data(self, data):
        
        # Create a timestamp of the date(Day, Month & Year)
        data['timestamp'] = data['date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
        
        if data.empty:
            print("No tweets downloaded. Finishing execution")
            
        if data['id'].unique().all():
            pass
        else:
            print(f"Primary Key check is violated, Number of duplicate values: {data.duplicated().sum()}")
            
        if data.isnull().values.any():
            print(f"\nNull values detected, Number of null: \n{data.isnull().sum()}")
        
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        timestamps = data['timestamp'].tolist()
        for timestamp in timestamps:
            if datetime.datetime.strptime(timestamp, '%Y-%m-%d') != yesterday:
                print("Atleast one of the returned tweet does not come from within the last 24 hours")
        
    def basic_processing(self, data):
        data = data.drop(['date', 'timestamp'], axis=1)
        print(f'\nNumber of duplicate entry of unlabelled data: {data.tweet.duplicated().sum()}')
        # Remove duplicates
        data = data[~data.tweet.duplicated()]
        print(f'Duplicate entry removed: {data.tweet.duplicated().sum()}')
        # We will remove the usernames and RT(retweet) in the tweet column
        # data['tweet'] = data.tweet.replace(regex=re.compile(r"@([A-Za-z0-9_]+)"), value='')
        # data['tweet'] = data.tweet.replace(regex=re.compile(r"RT([\s:]+)"), value='')
        return data

    def load_train_data(self):
        data = pd.read_csv(os.path.join(config.DATAPATH, 'train.csv'))
        # Remove all records with no label
        data = data[data.label != '']
        # data['tweet'] = data.tweet.replace(regex=re.compile(r"@([A-Za-z0-9_]+)"), value='')
        return data

if __name__ == '__main__':
    
    t = ToPandasDF(config.PASSWORD, config.HOST, config.DATABASE, config.USER)
    stored_data = t.MySQLconnect("SELECT id, created_at, tweet FROM `twitterdb`.`twitter_table`;")
    t.check_if_valid_data(stored_data)
    unlabelled_data = t.basic_processing(stored_data)
    labelled_data = t.load_train_data()


# In[3]:


# print(unlabelled_data.shape)
# unlabelled_data.info()
# unlabelled_data.head(10)


# In[4]:


# Initialize the TfidfVectorizer, Lemmatizer and stopwords
# tfVectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, max_features=800, ngram_range=(1, 1), use_idf=True)tfVectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, max_features=800, ngram_range=(1, 1), use_idf=True)
# tfVectorizer = TfidfVectorizer(min_df=5, max_df=0.75, max_features=1000, ngram_range=(1, 2))
# lemmatizer = WordNetLemmatizer()
# stopwords = set(json.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/stopWords/custome_nltk_stopwords.json", "r")))
# stopwords_json = set(json.load(open("/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/stopWords/custome_json_stopwords.json", "r")))
# stopwords_punctuation = set.union(stopwords, stopwords_json, punctuation)

# print(labelled_data.shape)
# labelled_data.info()
# labelled_data.head(10)


# # Target Exploration (label)

# In[5]:


# labelled_data.label.value_counts()


# **The dataset is imbalanced based on hate speech**

# In[6]:


# size=labelled_data.label.value_counts()
# labels='Non Hate', 'Hate'
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.pie(size, explode=(0, 0.2), labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'red'])
# ax.axis('equal')
# plt.title('Proportion of Twitter Users', size=15)
# ax.legend(labels, bbox_to_anchor=(1, 0), loc='lower left', title='Speech')
# plt.show()


# It is given that 7.0% of twitter users might Hate. So the baseline model could be to predict that 7.0% of the users will Hate. Given 7.0% is a small number, we need to ensure that the chosen model does predict with great accuracy this 7.0% as it is of interest to the company to identify these users as opposed to accurately predicting the users that are non haters.

# In[7]:


def basic_prep(data):
    data['mention_count'] = data.tweet.apply(lambda x: len(re.findall(r"@[\w\-]+", x)))
    data['tweet'] = data.tweet.replace(regex=re.compile(r"@([A-Za-z0-9_]+)"), value='')
    data['tweet'] = data.tweet.replace(regex=re.compile(r"RT([\s:]+)"), value='')
    return data

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
    final_sent = final_sent.replace("n't", 'not').replace("nt", "not")
    return final_sent

data_cleaned = labelled_data.copy()
data_cleaned = basic_prep(data_cleaned)
data_cleaned['cleaned_text'] = data_cleaned.tweet.apply(preprocess_text)
print(f'The longest for labelled tweet is: {max(data_cleaned.cleaned_text.str.len())}')
print(f'The shortest for labelled tweet is: {min(data_cleaned.cleaned_text.str.len())}')

unlabelled_cleaned = unlabelled_data.copy()
unlabelled_cleaned = basic_prep(unlabelled_cleaned)
unlabelled_cleaned['cleaned_text'] = unlabelled_cleaned.tweet.apply(preprocess_text)
print(f'The longest for unlabelled tweet is: {max(unlabelled_cleaned.cleaned_text.str.len())}')
print(f'The shortest for unlabelled tweet is: {min(unlabelled_cleaned.cleaned_text.str.len())}')


# # Visualize Word frequency

# In[8]:


# Let's split the dataset into non hate(0) and hate(1) so as to visualize the frequency of the words
no_hate = data_cleaned[data_cleaned['label']==0]
hate = data_cleaned[data_cleaned['label']==1]

wordcloud = WordCloud(background_color='black').generate(' '.join(no_hate.cleaned_text))
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('No Hatred Text')
plt.axis('off')
plt.show()


# **Above, we can see that those are `non-hate` related words**

# In[9]:


wordcloud = WordCloud(background_color='black').generate(' '.join(hate.cleaned_text))
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Hate Text')
plt.axis('off')
plt.show()


# In[10]:


def transform(data, column):
    # We will encode text data using TF-IDF
    tfidf_feats = tfVectorizer.fit(data[column])
    tfidf_feat_arr = tfidf_feats.transform(data[column]).toarray()
    tfidf = pd.DataFrame(tfidf_feat_arr, columns=tfVectorizer.get_feature_names_out())
    return tfidf, tfidf_feats

def merge(tfidf, data_to_merge):
    # Join both DataFrames
    data = tfidf.merge(data_to_merge, left_index=True, right_index=True)
    data = data.drop(['cleaned_text'], axis=1)
    return data

data_tfidf, tfVectorized = transform(data_cleaned, 'cleaned_text')
unlabelled_tfidf, _ = transform(unlabelled_cleaned, 'cleaned_text')


# # Feature Engineering

# ## Frequency distribution of Part of Speech Tags

# In[11]:


get_ipython().run_cell_magic('time', '', 'pos_group = {\n    \'noun\':[\'NN\',\'NNS\',\'NNP\',\'NNPS\'],\n    \'pron\':[\'PRP\',\'PRP$\',\'WP\',\'WP$\'],\n    \'verb\':[\'VB\',\'VBD\',\'VBG\',\'VBN\',\'VBP\',\'VBZ\'],\n    \'adj\':[\'JJ\',\'JJR\',\'JJS\'],\n    \'adv\':[\'RB\',\'RBR\',\'RBS\',\'WRB\']\n}\n\n        \ndef count_pos_tag(text, flags):\n    \n    """Function to check and count the respective parts of speech tags"""\n    \n    count=0\n    tokens = [contractions.fix(i.lower()) for i in word_tokenize(text)]\n    tags = pos_tag(tokens)\n\n    for (token, tag) in tags:\n        token = re.sub(r"([0-9]+|[-_@./&+]+|``)", \'\', token)\n        token = re.sub(r"(@[A-Za-z0-9_]+)|[^\\w\\s]|#|http\\S+", \'\', token)\n        token = token.encode("ascii", "ignore")\n        token = token.decode()\n        if tag in pos_group[flags]:\n            count+=1\n    return count\n\ndef make_features(data):\n\n    data[\'noun_count\'] = data.tweet.apply(lambda x: count_pos_tag(x, \'noun\'))\n    data[\'verb_count\'] = data.tweet.apply(lambda x: count_pos_tag(x, \'verb\'))\n    data[\'adj_count\'] = data.tweet.apply(lambda x: count_pos_tag(x, \'adj\'))\n    data[\'adv_count\'] = data.tweet.apply(lambda x: count_pos_tag(x, \'adv\'))\n    data[\'pron_count\'] = data.tweet.apply(lambda x: count_pos_tag(x, \'pron\'))\n\n    data[\'char_count\'] = data.tweet.apply(len)\n    data[\'word_count\'] = data.tweet.apply(lambda x: len(x.split()))\n    data[\'uniq_word_count\'] = data.tweet.apply(lambda x: len(set(x.split())))\n    data[\'htag_count\'] = data.tweet.apply(lambda x: len(re.findall(r\'#[\\w\\-]+\', x)))\n    data[\'stopword_count\'] = data.tweet.apply(lambda x: len([wrd for wrd in word_tokenize(x) if wrd in stopwords]))\n    data[\'sent_count\'] = data.tweet.apply(lambda x: len(sent_tokenize(x)))\n    data[\'avg_word_len\'] = data[\'char_count\']/(data[\'word_count\']+1)\n    data[\'avg_sent_len\'] = data[\'word_count\']/(data[\'sent_count\']+1)\n    data[\'uniq_vs_words\'] = data.uniq_word_count/(data.word_count+1) # Ratio of unique words to the total number of words\n    data[\'stopwords_vs_words\'] = data.stopword_count/(data.word_count+1)\n    data[\'title_word_count\'] = data.tweet.apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))\n    data[\'uppercase_count\'] = data.tweet.apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))\n    data = data.drop([\'id\', \'tweet\'], axis=1)\n    return data\n\n\ndata_cleaned = make_features(data_cleaned)\nunlabelled_cleaned = make_features(unlabelled_cleaned)\n\ndata_cleaned = merge(data_tfidf, data_cleaned)\nunlabelled_cleaned = merge(unlabelled_tfidf, unlabelled_cleaned)\n')


# In[12]:


print(f"Number of columns of the data_cleaned: {data_cleaned.shape[1]}")
print(f"Number of columns of the unlabelled_cleaned: {unlabelled_cleaned.shape[1]}")


# # Check for Missing Values

# In[13]:


print(data_cleaned.isnull().sum().sort_values(ascending=False))


# In[14]:


print(unlabelled_cleaned.isnull().sum().sort_values(ascending=False))


# In[15]:


target_labelled_data = data_cleaned.label
data_cleaned = data_cleaned.drop(['label'], axis=1)
# unlabelled_cleaned = unlabelled_cleaned.drop('id', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_cleaned, target_labelled_data, test_size=0.2, random_state=43)
print(f"Train Size: {(X_train.shape[0]/data_cleaned.shape[0]):.2f}%")
print(f"Test Size: {(X_test.shape[0]/data_cleaned.shape[0]):.2f}%")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, unlabelled_cleaned.shape)

def confusion_matrix_plot(cm, normalized= True, cmap= 'bone'):
    norm_cm = cm
    if normalized:
        plt.figure(figsize=(6,4))
        norm_cm = (cm.astype('float')/ cm.sum(axis= 1)[:, np.newaxis])
        return sns.heatmap(norm_cm, annot= cm, fmt='g', 
                           xticklabels= ['Predicted: No Hate', 'Predicted: Yes Hate'],
                           yticklabels=['Actual: No Hate', 'Actual: Yes Hate'])

def roc_auc_curve(y_test, pred):
    plt.figure(figsize=(14, 8))
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    plt.plot(fpr, tpr, label= f"Validation AUC-ROC={str(auc)}")
    x = np.linspace(0, 1, 1000)
    plt.plot(x, x, linestyle='-')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)

def train(model, scaler, X_train, y_train, X_test):
    
    """Sklearn training interface
    """
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train.values, y_train)
    pred = pipeline.predict_proba(X_test.values)[:, 1]
    y_pred = pipeline.predict(X_test.values)
    return pred, y_pred


# # Model Building & Evaluation

# ### Get the best Model to perform Pseudo-Labelling on the unlabelled_data

# ## MultinomialNB & Parameter tuning

# In[16]:


for a in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
    pred, y_pred = train(MultinomialNB(alpha=a), MinMaxScaler(), X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{a}->{auc:.4f}")


# In[17]:


pred, y_pred = train(MultinomialNB(alpha=0.1), MinMaxScaler(), X_train, y_train, X_test)
print(f"ROC AUC Naive Bayes Score: {roc_auc_score(y_test, pred):.4f}")
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test, pred)
plt.show()


# The orange line seen here represents the random selection. What it says is that if i get 50% of the False Positives in my random selection, I also get 50% of the True Positives or True Users that will hate.

# ## Logistic Regression & Parameter tuning

# In[18]:


for c in [0.0001, 0.001, 0.01, 0.1, 1]:
    pred, y_pred = train(LogisticRegression(solver='liblinear', C=c), StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{c}->{auc:.4f}")


# In[19]:


for m in [0.0001, 0.01, 0.1]:
    print(f"Inverse of regularization strength {m}")
    
    for tol in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
        pred, y_pred = train(LogisticRegression(solver='liblinear', C=m, tol=tol), StandardScaler(),
                             X_train, y_train, X_test)
        auc = roc_auc_score(y_test, pred)
        print(f"{tol}->{auc:.4f}")
    print()


# In[20]:


for mClass in ["auto", "ovr"]:
    pred, y_pred = train(LogisticRegression(solver='liblinear', C=0.0001, tol=10, multi_class=mClass),
                         StandardScaler(),
                         X_train, y_train, X_test)
    
    auc = roc_auc_score(y_test, pred)
    print(f"{mClass}->{auc:.4f}")


# In[21]:


pred, y_pred = train(LogisticRegression(solver='liblinear', C=0.0001, tol=10, multi_class='auto'),
                     StandardScaler(),
                     X_train, y_train, X_test)
print(f"ROC AUC Logistic Regression Score: {roc_auc_score(y_test, pred):.4f}")
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
plt.show()


# #### Apply Cost-Sensitive Logistic Regression for Imbalanced Classification

# In[22]:


balance = [{0:1,1:1}, {0:1,1:10}, {0:1,1:100}]

for weight in balance:
    pred, y_pred = train(LogisticRegression(solver='liblinear', C=0.0001, tol=10, multi_class='auto',
                                            class_weight=weight),
                         StandardScaler(),
                         X_train, y_train, X_test)
    print(f"ROC AUC Logistic Regression Score: {roc_auc_score(y_test, pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_plot(cm)
    plt.show()


# In[23]:


pred, y_pred = train(LogisticRegression(solver='liblinear', C=0.0001, tol=10, multi_class='auto',
                                        class_weight={0:1,1:10}),
                     StandardScaler(),
                     X_train, y_train, X_test)
print(f"ROC AUC Logistic Regression Score: {roc_auc_score(y_test, pred):.4f}")
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test, pred)
plt.show()

# compare the actual class and predicted class
out = pd.DataFrame(y_test[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# ## Random Forest & Parameter tuning

# In[24]:


for n in [250, 300, 350]:
    pred, y_pred = train(RandomForestClassifier(n_estimators=n, random_state=42),
                         StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{n}->{auc:.4f}")


# In[25]:


for m in [300, 350]:
    print(f"Number of estimators {m}")
    
    for c in ["gini", "entropy"]:
        pred, y_pred = train(RandomForestClassifier(n_estimators=m, criterion=c, random_state=42),
                             StandardScaler(),
                             X_train, y_train, X_test)
        auc = roc_auc_score(y_test, pred)
        print(f"{c}->{auc:.4f}")
    print()


# In[26]:


for samp in [2, 3, 4]:
    pred, y_pred = train(RandomForestClassifier(n_estimators=350, criterion='entropy', 
                                                max_depth=None, min_samples_split=samp),
                         StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{samp}->{auc:.4f}")


# In[27]:


pred, y_pred = train(RandomForestClassifier(n_estimators=350, criterion='entropy',
                                            max_depth=None, class_weight={0:1,1:100}, random_state=42),
                     StandardScaler(),
                     X_train, y_train, X_test)
auc = roc_auc_score(y_test, pred)
print(f"ROC AUC Random Forest Classifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test)*100)
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
plt.show()

# compare the actual class and predicted class
out = pd.DataFrame(y_test[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# In[28]:


pred, y_pred = train(RandomForestClassifier(n_estimators=350, criterion='entropy',
                                            max_depth=None, random_state=42),
                     StandardScaler(),
                     X_train, y_train, X_test)
auc = roc_auc_score(y_test, pred)
print(f"ROC AUC Random Forest Classifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test)*100)
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test, pred)
plt.show()

# compare the actual class and predicted class
out = pd.DataFrame(y_test[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# ## LGMClassifier & Parameter tuning

# In[29]:


for n in [25, 50, 100, 150, 200, 250, 300, 350, 400]:
    pred, y_pred = train(LGBMClassifier(n_estimators=n, objective='binary'),
                         StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{n}->{auc:.4f}")


# In[30]:


for lr in [0.001, 0.01, 0.1, 1, 1.5, 2, 2.5]:
    pred, y_pred = train(LGBMClassifier(n_estimators=250, objective='binary', learning_rate=lr),
                         StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{lr}->{auc:.4f}")


# In[31]:


pred, y_pred = train(LGBMClassifier(n_estimators=250, learning_rate=0.1, objective='binary'),
                     StandardScaler(),
                     X_train, y_train, X_test)
auc = roc_auc_score(y_test, pred)
print(f"ROC AUC LGBM Classifier Score: {auc:.4f}")
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
plt.show()


# In[32]:


balance = [{0:1,1:1}, {0:1,1:10}, {0:1,1:100}]

for weight in balance:
    pred, y_pred = train(LGBMClassifier(n_estimators=250, learning_rate=0.1, objective='binary',
                                        class_weight=weight),
                         StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"ROC AUC LGBM Classifier Score: {auc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_plot(cm)
    plt.show()


# In[33]:


pred, y_pred = train(LGBMClassifier(n_estimators=250, learning_rate=0.1, objective='binary',
                                    class_weight={0:1,1:10}),
                     StandardScaler(),
                     X_train, y_train, X_test)
auc = roc_auc_score(y_test, pred)
print(f"ROC AUC LGBM Classifier Score: {auc:.4f}")
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test, pred)
plt.show()


# ## XGBoost & Parameter tuning

# In[34]:


for n in [200, 300, 350]:
    pred, y_pred = train(XGBClassifier(n_estimators=n, eval_metric="auc", 
                                       objective='binary:logistic'),
                         StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{n}->{auc:.4f}")


# In[35]:


for lr in [0.0001, 0.001, 0.1, 1]:
    pred, y_pred = train(XGBClassifier(n_estimators=350, eval_metric="auc", 
                                       objective='binary:logistic', learning_rate=lr),
                         StandardScaler(),
                         X_train, y_train, X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"{lr}->{auc:.4f}")


# In[22]:


params = {
    'objective':'binary:logistic',
    'n_estimators':350,
    'eval_metric':'auc'
}
pred, y_pred = trainXGB(StandardScaler(), 
                        X_train, y_train, X_test, params)


# In[36]:


pred, y_pred = train(XGBClassifier(n_estimators=350, eval_metric="auc", 
                                    objective='binary:logistic'),
                     StandardScaler(),
                     X_train, y_train, X_test)
auc = roc_auc_score(y_test, pred)
print(f"ROC AUC XGBoost Classifier Score: {auc:.4f}")
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test, pred)
plt.show()


# The Random Forest model has the best performance. We will use cross validation to prevent overfitting and check so we know the actual scores of individual model.

# # Model Evaluation with Cross Validation

# In[17]:


def cross_validation_score(ml_model, scaler, thres = 0.5, random_st=42, cols = data_cleaned.columns):
    
    """
    Function to calculate the k fold cross validation stratified on the basis of target
    and prints the ROC, Recall and Precision Scores.
    
    Args:
        ml_model (numpy array) : ml_model for predictions
        thres (float) : threshold for the probabilities of the model predictions
        random_st (int) : random_st is the random state for Kfold
        cols (string) : cols are the column names
    
    Returns:
        cv_scores (float) : cross validation scores
    """
    
    i= 1
    x1 = data_cleaned.copy()
    x1 = data_cleaned[cols]
    cv_scores = []
    
    sKf = StratifiedKFold(n_splits= 5, shuffle= True, random_state= random_st)
    
    for train_index, test_index in sKf.split(x1, y):
        print(f"\n{i} of KFold {sKf.n_splits}")
        xtrain, xval = x1.iloc[train_index], x1.iloc[test_index]
        ytrain, yval = y.iloc[train_index], y.iloc[test_index]
        
        model = ml_model
        pipeline = make_pipeline(scaler, model)
        pipeline.fit(xtrain, ytrain)
        y_pred = pipeline.predict(xval)
        pred_probs = pipeline.predict_proba(xval)
        pp = []
        
        # Use threshold to define the classes based on probability values
        for j in pred_probs[:,1]:
            if j > thres:
                pp.append(1)
            else:
                pp.append(0)
        # Calculate scores for each fold
        pred_val = pp
        roc_score = roc_auc_score(yval, pred_probs[:,1])
        recall = recall_score(yval, pred_val)
        precision = precision_score(yval, pred_val)
        msg = ""
        msg += f"ROC AUC Score: {roc_score:.4f}, Recall Score: {recall:.4f}, Precision Score: {precision:.4f}"
        print(f"{msg}")
        cv_scores.append(roc_score)
        i+=1
        
    return cv_scores


# ## Multinomial NB

# In[19]:


y=target_labelled_data
nb_cv_score = cross_validation_score(MultinomialNB(alpha=0.1), MinMaxScaler())


# ## Logistic Regression

# In[39]:


log_cv_score = cross_validation_score(LogisticRegression(solver='liblinear', 
                                       C=0.0001, tol=10, multi_class='auto',
                                       class_weight={0:1,1:10}), StandardScaler())


# ## RandomForestClassifier

# In[40]:


rf_cv_score = cross_validation_score(RandomForestClassifier(n_estimators=350,
                                                            criterion='entropy', max_depth=None,
                                                            class_weight={0:1,1:100},
                                                            random_state=42),
                                     StandardScaler())


# ## LGBMClassifier

# In[41]:


lgm_cv_score = cross_validation_score(LGBMClassifier(n_estimators=250, 
                                                     learning_rate=0.1, objective='binary',
                                                     class_weight={0:1,1:10}),
                                      StandardScaler())


# ## XGBClassifier

# In[42]:


xgb_cv_score = cross_validation_score(XGBClassifier(n_estimators=350, 
                                                    eval_metric="auc",
                                                    objective='binary:logistic'),
                                      StandardScaler())


# # Comparison of Model Fold wise

# In[43]:


compare_score = pd.DataFrame({'nb_cv_score':nb_cv_score,
                              'log_cv_score':log_cv_score,
                              'rf_cv_score':rf_cv_score,
                              'lgm_cv_score':lgm_cv_score,
                              'xgb_cv_score':xgb_cv_score
                             })

compare_score.plot(y = ['nb_cv_score','log_cv_score','rf_cv_score',
                        'lgm_cv_score','xgb_cv_score'], 
                   kind = 'bar')

plt.title('Model Comparison Fold Wise')
plt.xlabel('Features')
plt.ylabel('ROC AUC');


# **The `LGBM Model` & `XGBoost Model` has the best performance across 5-fold. Therefore, we will work on improving them and select the one that generalizes best on unseen data**

# In[44]:


xgb_model = XGBClassifier(n_estimators=350, eval_metric="auc", objective='binary:logistic')
xgb_model.fit(X_train.values, y_train)

step_factor = 0.02
threshold_value = 0.1
roc_score = 0
proba = xgb_model.predict_proba(X_test.values)

# Continue to check for optimal value when threshold is
# less than 0.8
while threshold_value <= 0.8:
    temp_thresh = threshold_value
    predicted = (proba[:,1] >= temp_thresh).astype('int')
    print(f"Threshold: {temp_thresh}->{roc_auc_score(y_test, predicted)}")
    #store the threshold for best classification
    if roc_score < roc_auc_score(y_test, predicted):
        roc_score = roc_auc_score(y_test, predicted)
        threshold_score = threshold_value
    threshold_value = threshold_value + step_factor
print(f'\n---Optimum Threshold: {threshold_score}->ROC: {roc_score}')


# In[52]:


cross_validation_score(XGBClassifier(n_estimators=350, eval_metric="auc", 
                                    objective='binary:logistic'),
                       StandardScaler(), thres=0.1)


# In[45]:


lgb_model = LGBMClassifier(n_estimators=250, 
                         learning_rate=0.1, objective='binary',
                         class_weight={0:1,1:10})

lgb_model.fit(X_train.values, y_train)

step_factor = 0.02
threshold_value = 0.1
roc_score = 0
proba = lgb_model.predict_proba(X_test.values)

while threshold_value <= 0.8:
    temp_thresh = threshold_value
    predicted = (proba[:,1] >= temp_thresh).astype('int')
    print(f"Threshold: {temp_thresh}->{roc_auc_score(y_test, predicted)}")
    #store the threshold for best classification
    if roc_score < roc_auc_score(y_test, predicted):
        roc_score = roc_auc_score(y_test, predicted)
        threshold_score = threshold_value
    threshold_value = threshold_value + step_factor
print(f'\n---Optimum Threshold: {threshold_score}->ROC: {roc_score}')


# In[20]:


cross_validation_score(LGBMClassifier(n_estimators=250, 
                                      learning_rate=0.1, objective='binary',
                                      class_weight={0:1,1:10}),
                       StandardScaler(), thres=0.32)


# There is no improvement in the recall score
# 
# **Ways to improve this model**
# 1. Add more training data
# 2. Try Over/Undersampling techniques like SMOTE

# # Addressing Imbalanced Class with SMOTE

# In[21]:


print(data_cleaned.shape)
print(target_labelled_data.shape)


# In[22]:


print(target_labelled_data.value_counts())
print(target_labelled_data.value_counts(normalize=True)*100)


# In[23]:


smote = SMOTE(sampling_strategy='minority')
x_sm, y_sm = smote.fit_resample(data_cleaned, target_labelled_data)
print(y_sm.value_counts())
print(y_sm.value_counts(normalize=True)*100)


# In[24]:


X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(x_sm, y_sm, stratify=y_sm, 
                                                                random_state= 42, test_size= 0.2)


# In[25]:


pred, y_pred = train(XGBClassifier(n_estimators=350, eval_metric="auc", 
                                    objective='binary:logistic'),
                     StandardScaler(),
                     X_train_sm, y_train_sm, X_test_sm)

auc = roc_auc_score(y_test_sm, pred)
print(f"ROC AUC XGBoost Classifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test_sm)*100)
cm = confusion_matrix(y_test_sm, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test_sm, pred)
plt.show()

# compare the actual class and predicted class
out = pd.DataFrame(y_test_sm[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# In[26]:


pred, y_pred = train(LGBMClassifier(n_estimators=250, 
                                    learning_rate=0.1, objective='binary',
                                    class_weight={0:1,1:10}),
                     StandardScaler(),
                     X_train_sm, y_train_sm, X_test_sm)

auc = roc_auc_score(y_test_sm, pred)
print(f"ROC AUC LGBMClassifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test_sm)*100)
cm = confusion_matrix(y_test_sm, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test_sm, pred)
plt.show()

out = pd.DataFrame(y_test_sm[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# **We will use the XGBoost Classifier because of it's low '%' of Misclassified labels**

# In[29]:


xgb_pipeline = make_pipeline(StandardScaler(),
                             XGBClassifier(n_estimators=350, eval_metric="auc", 
                                           objective='binary:logistic'))

xgb_pipeline.fit(X_train_sm.values, y_train_sm)
pred = xgb_pipeline.predict_proba(X_test_sm.values)[:, 1]
y_pred = xgb_pipeline.predict(X_test_sm.values)
auc = roc_auc_score(y_test_sm, pred)
print(f"ROC AUC XGBoost Classifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test_sm)*100)
cm = confusion_matrix(y_test_sm, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test_sm, pred)
plt.show()

# compare the actual class and predicted class
out = pd.DataFrame(y_test_sm[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# # Pseudo-Labelling based on Confidence

# In[30]:


print("Predicting labels for unlabelled data"+".."*2)
preds_probs = xgb_pipeline.predict_proba(unlabelled_cleaned.values)
preds = xgb_pipeline.predict(unlabelled_cleaned.values)

prob_0 = preds_probs[:,0]
prob_1 = preds_probs[:,1]
# Store the predictions and probabilities in a dataframe
pseudo_df = pd.DataFrame([])
pseudo_df['pseudo_prediction'] = preds
pseudo_df['prob_0'] = prob_0
pseudo_df['prob_1'] = prob_1
pseudo_df.index = unlabelled_cleaned.index
pseudo_df['max'] = pseudo_df[['prob_0','prob_1']].max(axis=1)

pseudo_df.head()


# ## Plotting the Confidence

# The below graph gives the distribution of confidence as expressed by the probability of the class which is most probable.

# In[31]:


sns.histplot(data = pseudo_df, x = 'max', bins=10)
plt.show()


# We will consider probability of 0.8 because of has high probability count which means the predicted labels is near accurate.

# In[32]:


conf_ind=pseudo_df["max"]>0.8
X_new = np.append(X_train, unlabelled_cleaned.loc[conf_ind,:], axis=0)
y_new = np.append(y_train, pseudo_df.loc[conf_ind, ['pseudo_prediction']])
X_new = pd.DataFrame(X_new, columns = X_train.columns)
y_new = pd.Series(y_new, name = 'label')

print(f"Old Train Data shape: {X_train.shape} \nOld Train label shape: {y_train.shape}\n \nNew Train Data shape: {X_new.shape} \nNew Train label shape: {y_new.shape}\n")

print(X_new.isnull().sum())
print(y_new.isnull().sum())


# In[35]:


xgb_pipeline = make_pipeline(StandardScaler(),
                             XGBClassifier(n_estimators=350, eval_metric="auc", 
                                           objective='binary:logistic'))

xgb_pipeline.fit(X_new.values, y_new)
pred = xgb_pipeline.predict_proba(X_test.values)[:, 1]
y_pred = xgb_pipeline.predict(X_test.values)
auc = roc_auc_score(y_test, pred)
print('Performance after Semi-Supervised Learning')
print(f"ROC AUC XGBoost Classifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test)*100)
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test, pred)
plt.show()

# compare the actual class and predicted class
out = pd.DataFrame(y_test_sm[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# # Apply SMOTE

# In[36]:


print(X_new.shape)
print(y_new.shape)


# In[37]:


print(y_new.value_counts())
print(y_new.value_counts(normalize=True)*100)

smote = SMOTE(sampling_strategy='minority')
x_sm, y_sm = smote.fit_resample(X_new, y_new)
print()
print('After applying SMOTE')
print(y_sm.value_counts())
print(y_sm.value_counts(normalize=True)*100)


# In[38]:


X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(x_sm, y_sm, stratify=y_sm, 
                                                                random_state= 42, test_size= 0.2)

pred, y_pred = train(XGBClassifier(n_estimators=350, eval_metric="auc", 
                                    objective='binary:logistic'),
                     StandardScaler(),
                     X_train_sm, y_train_sm, X_test_sm)

auc = roc_auc_score(y_test_sm, pred)
print(f"ROC AUC XGBoost Classifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test_sm)*100)
cm = confusion_matrix(y_test_sm, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test_sm, pred)
plt.show()

out = pd.DataFrame(y_test_sm[0:30])
out['Predicted_class'] = y_pred[0:30]
out


# We can see a little improvement in the ROC AUC score and also a reduction in the percentage of Misclassified class. Therefore, we will store this as our final model.

# In[39]:


final_model = make_pipeline(StandardScaler(),
                            XGBClassifier(n_estimators=350, eval_metric="auc", 
                                           objective='binary:logistic'))

final_model.fit(X_train_sm.values, y_train_sm)
pred = final_model.predict_proba(X_test_sm.values)[:, 1]
y_pred = final_model.predict(X_test_sm.values)
auc = roc_auc_score(y_test_sm, pred)
print(f"ROC AUC XGBoost Classifier Score: {auc:.4f}")
print("'%' of Misclassified class:", np.mean(y_pred != y_test_sm)*100)
cm = confusion_matrix(y_test_sm, y_pred)
confusion_matrix_plot(cm)
roc_auc_curve(y_test_sm, pred)
plt.show()


# # Serialize Tfidf Vectorizer & the Final Model pipeline

# In[40]:


# with open('/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/vectorizers/vectorizerV3.bin', 'wb') as f:
#     joblib.dump(tfVectorized, f)
    
# with open('/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src/models/modelV3.bin', 'wb') as f:
#     joblib.dump(final_model, f)


# In[ ]:




