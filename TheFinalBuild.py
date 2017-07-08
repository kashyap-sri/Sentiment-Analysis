import tweepy
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from time import sleep
from threading import Thread
import json
import re
import fileinput
import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from collections import Counter
import operator
from collections import OrderedDict
from twython import Twython


def get_sentiment(hashtag):
    try:
        #consumer key, consumer secret, access token, access secret.
        handle = open("tweets","w")

        TWITTER_APP_KEY = 'VAWM9T0iNsTRAJ6WLVp2YKLdK' #supply the appropriate value
        TWITTER_APP_KEY_SECRET = 'hoZLIPeiz2kS0auGfGoAgyasQUfgTIr6VlIhU8i2jwQxiuiCFH'
        TWITTER_ACCESS_TOKEN = '1683434132-hr1LHwxGf5lZC0Wv8GqxLI8Bezyu1b3JAjZYuhk'
        TWITTER_ACCESS_TOKEN_SECRET = 'MY5oaDvTRUHhMtUatynIMcUlmU0qgda4SID5iJiKXcn5t'

        t = Twython(app_key=TWITTER_APP_KEY,
        app_secret=TWITTER_APP_KEY_SECRET,
        oauth_token=TWITTER_ACCESS_TOKEN,
        oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

        search = t.search(q= '#'+hashtag,   #**supply whatever query you want here**
        count=1000)

        tweets = search['statuses']
        lis = []
        for tweet in tweets:
            lis.append(tweet['text'])
            print(tweet['text'])


        handle1 = open("parsed_tweets","w")


        for parsed_tweets in lis:
            parsed_tweets = re.sub(r'[A-Z]* @[A-Za-z0-9_]*: ','',parsed_tweets)
            parsed_tweets = re.sub(r'https://.*','',parsed_tweets)
            parsed_tweets = re.sub(r'@[a-zA-Z0-9]* ','',parsed_tweets)
            parsed_tweets = re.sub(r'"  "*','',parsed_tweets)
            parsed_tweets = re.sub(r'["*,:\'#.!@_?/\\()-]','',parsed_tweets)
            emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
            parsed_tweets = emoji_pattern.sub(r'', parsed_tweets)
            handle1.write(parsed_tweets)
            handle1.write("\n")

        handle1.close()


        reviews = pd.read_csv('reviews.txt',header = None)
        labels = pd.read_csv('labels.txt',header = None)

        total_counts = Counter()
        for _, row in reviews.iterrows():
            total_counts.update(row[0].split(' '))

        vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]

        word2idx = {word: i for i, word in enumerate(vocab)}

        def text_to_vector(text):
            word_vector = np.zeros(len(vocab), dtype=np.int_)
            for word in text.split(' '):
                idx = word2idx.get(word, None)
                if idx is None:
                    continue
                else:
                    word_vector[idx] += 1
            return np.array(word_vector)

        word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
        for ii, (_, text) in enumerate(reviews.iterrows()):
            word_vectors[ii] = text_to_vector(text[0])

        Y = (labels=='positive').astype(np.int_)
        records = len(labels)

        shuffle = np.arange(records)
        np.random.shuffle(shuffle)
        test_fraction = 0.9

        train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
        trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
        testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)

        def build_model():
            # This resets all parameters and variables, leave this here
            tf.reset_default_graph()

            # Inputs
            net = tflearn.input_data([None, 10000])

            # Hidden layer(s)
            net = tflearn.fully_connected(net, 200, activation='ReLU')
            net = tflearn.fully_connected(net, 25, activation='ReLU')

            # Output layer
            net = tflearn.fully_connected(net, 2, activation='softmax')
            net = tflearn.regression(net, optimizer='sgd',
                                     learning_rate=0.1,
                                     loss='categorical_crossentropy')

            model = tflearn.DNN(net)
            return model

        model = build_model()

        model.load("Twitter_Analysis.tfl")

        predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
        test_accuracy = np.mean(predictions == testY[:,0], axis=0)

        tweets_handle = open("parsed_tweets","r")
        tweet_file = tweets_handle.read()
        tweet_list = tweet_file.splitlines()
        predictionlist = {}
        for tweet in tweet_list:
            if len(tweet) > 30:
                 positive_prob = model.predict([text_to_vector(tweet.lower())])[0][1]
                 predictionlist[tweet] = positive_prob

        sorted_x = sorted(predictionlist.items(), key=operator.itemgetter(1))

        od = OrderedDict(sorted_x)
        ordered_list = list(od.items())

        posilist = []
        count = 3;
        for pos in reversed(ordered_list):
            if count >= 0:
                 posilist.append(pos[0])
                 count = count - 1

        negalist = []
        count = 3;
        for neg in (ordered_list):
            if count >= 0:
                 negalist.append(neg[0])
                 count = count - 1

        aggregate = sum(predictionlist.values())/len(predictionlist)

        print(posilist)
        print(negalist)
        print(aggregate*100)

        return {"pos": posilist, "neg": negalist, "sentiment": aggregate*100}

    except ZeroDivisionError:
        posilist = ["Not enough data available","Not enough data available","Not enough data available"]
        negalist = ["Not enough data available","Not enough data available","Not enough data available"]
        aggregate = 0
        return {"pos": posilist, "neg": negalist, "sentiment": aggregate*100}
