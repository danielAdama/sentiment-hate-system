from reprlib import aRepr
import numpy
import pandas
import mysql.connector
from mysql.connector import Error
import json
import tweepy as tw
from src.config import config


consumer_key = config.APIKEY
consumer_secret = config.APIKEYSECRET
access_token = config.ACCESSTOKEN
access_token_secret = config.ACCESSTOKENSECRET

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
client = tw.Client(bearer_token=config.BEARERTOKEN)


if __name__ == '__main__':

    # According to research, Canadian cities with the highest black population that experience 
    # racism the most are Toronto, Montreal, Edmonton, Calgary, Winnipeg, and Vancouver.
    # lang:en is asking for the tweets to be in english

    hashtags = [
        '#toronto lang:en', '#edmonton lang:en', '#calgary lang:en', '#winnipeg lang:en', '#montreal lang:en', 
        '#vancouver lang:en', '#studiolife lang:en', '#aislife lang:en', '#requires lang:en', '#passions lang:en',
        '#white lang:en', '#supremacists lang:en', '#inthefeels lang:en', '#deep lang:en', '#politics lang:en',
        '#blm lang:en', '#brexit lang:en', '#trump lang:en', '#music lang:en'
        ]

    for hashtag in hashtags:
        tweets = client.search_recent_tweets(query=hashtag, tweet_fields=['context_annotations', 'created_at'], max_results=100)

        for tweet in tweets.data:
            date = tweet.created_at
            text = tweet.text

    # for hashtag in hashtags:
    #     query = tw.Cursor(api.search_tweets, q=hashtag).items(100)

    #     for tweet in query:
    #         date = tweet.created_at
    #         text = tweet.text
        

        try:
            connection = mysql.connector.connect(
                host=config.HOST, 
                database=config.DATABASE,
                user=config.USER,
                password=config.PASSWORD
                )
            
            if connection.is_connected():

                cursor = connection.cursor()
                insertQuery = "INSERT INTO twitter_table (created_at, tweet) VALUES (%s, %s)"
                cursor.execute(insertQuery, (date, text))
                connection.commit()

        except Error as e:
            print(e)
            
        cursor.close()
        connection.close()
        print(f"Tweet collected at: {date}")
    print()
    print("Data is successfully entered.")