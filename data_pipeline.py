from reprlib import aRepr
import numpy
import pandas
import mysql.connector
import tweepy as tw
from src.config import config


consumer_key = config.APIKEY
consumer_secret = config.APIKEYSECRET
access_token = config.ACCESSTOKEN
access_token_secret = config.ACCESSTOKENSECRET

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# According to research, Canadian cities with the highest black population that experience 
# racism the most are Toronto, Montreal, Edmonton, Calgary, Winnipeg, and Vancouver.

hashtags = [
    '#toronto', '#edmonton', '#calgary', '#winnipeg', '#montreal', '#vancouver', '#studiolife', 
    '#aislife', '#requires', '#passions', '#white', '#supremacists', '#inthefeels', 
    '#deep', '#politics', '#blm', '#brexit', '#trump', '#music'
    ]

for hashtag in hashtags:
    query = tw.Cursor(api.search_tweets, q=hashtag).items(1)
    for tweet in query:
        date = tweet.created_at
        text = tweet.text

        print(text)