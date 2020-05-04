import tweepy
import pandas as pd
import glob
import csv

# Authenticate to Twitter
auth = tweepy.OAuthHandler("INSERT_YOUR_API_KEY", 
    "INSERT_YOUR_API_SECRET_KEY")
auth.set_access_token("INSERT_YOUR_ACCESS_TOKEN", 
    "INSERT_YOUR_ACCESS_TOKEN_SECRET")

# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True,
    wait_on_rate_limit_notify=True)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

hashtags = pd.read_csv('sorted_hashtag_0_dic.csv', usecols=[0], header=None)
hastags_list = hashtags.values.tolist()
flat_list = [item for sublist in hastags_list for item in sublist]
print(flat_list[0:300])
tweet_list = []
id = 10874
keyword = ''
target = 0
search_words = flat_list[0:300]
for item in search_words:
    print(item)
    tweets = tweepy.Cursor(api.search, q=item, since="2016-10-01", lang="en").items(100)
    for tweet in tweets:
        # print(tweet.text)
        outtweets = [id, 
                    keyword, 
                    tweet.user.location, 
                    tweet.text.encode("utf-8").decode("utf-8"),
                    target]
        tweet_list.append(outtweets)
        print(len(tweet_list))
        id += 1

header = ['id', 'keyword', 'location', 'text', 'target']
with open('tweets_0_300.csv', 'w') as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(i for i in header)
    for tweet in tweet_list:
        writer.writerow(tweet)
file.close()

twt = pd.read_csv('tweets_0_300.csv')
print(twt)