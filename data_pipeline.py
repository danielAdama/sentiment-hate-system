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


if __name__ == '__main__':

    # According to research, Canadian cities with the highest black population that experience 
    # racism the most are Toronto, Montreal, Edmonton, Calgary, Winnipeg, and Vancouver.

    hashtags = [
        '#toronto', '#edmonton', '#calgary', '#winnipeg', '#montreal', '#vancouver', '#studiolife', 
        '#aislife', '#requires', '#passions', '#white', '#supremacists', '#inthefeels', 
        '#deep', '#politics', '#blm', '#brexit', '#trump', '#music'
        ]

    for hashtag in hashtags:
        query = tw.Cursor(api.search_tweets, q=hashtag).items(200)

        for tweet in query:
            date = tweet.created_at
            text = tweet.text
            retweet_count = tweet.retweet_count
        

        try:
            connection = mysql.connector.connect(
                host=config.HOST, 
                database=config.DATABASE,
                user=config.USER,
                password=config.PASSWORD
                )
            
            # if connection.is_connected():
            cursor = connection.cursor()
            # cursor.execute("CREATE DATABASE twitterdb")
            # print("Database Created")
            # cursor.execute("CREATE TABLE twitter_table (created_at VARCHAR(45), tweet TEXT, retweet_count INT(11))")
            # print("Table Created")
            # cursor.execute("SHOW databases")
            # cursor.execute("SHOW tables")
            # for x in cursor:
            #     print(x)
            
            # cursor.execute("DROP TABLE IF EXISTS twitter_table")
            # print("Database Dropped")
            # cursor.execute("SELECT * FROM twitter_table")
            # myresult = cursor.fetchall()

            # for x in myresult:
            #     print(x)

            
            insertQuery = "INSERT INTO twitter_table (created_at, tweet, retweet_count) VALUES (%s, %s, %s)"
            cursor.execute(insertQuery, (date, text, retweet_count))
            connection.commit()

        except Error as e:
            print(e)
            
        cursor.close()
        connection.close()
        print(f"Tweet collected at: {date}")
    print()
    print("Data is successfully entered.")