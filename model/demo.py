print('Booting up...')
import time
start = time.time()
import pandas as pd
import numpy as np
from math import isnan

import torch
import preprocess
import dan
import gensim
import random


end = time.time()
print('Imports finished in {} seconds'.format(end-start))

start = time.time()
word2vec = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
end = time.time()
print('Embeddings loaded in {} seconds'.format(end-start))
'''
start = time.time()
twt = pd.read_csv('train.csv')
twt = twt.set_index('id')

text = twt['text'].to_list()
prep_text = [preprocess.processing(i) for i in text]
twt['prepped'] = prep_text

twt['vec'] = pd.Series([preprocess.tweet_vec(tweet, word2vec) for tweet in twt['prepped']], index = twt.index)

devtrain_idx = twt.loc[~twt['vec'].isna()].index.tolist()
random.shuffle(devtrain_idx)

train_p = 0.70
train_idx = devtrain_idx[:round(train_p*len(devtrain_idx))]
dev_idx = devtrain_idx[round(train_p*len(devtrain_idx)):]

train = [(vec, targ) for targ, vec in  zip(twt['target'][train_idx], twt['vec'][train_idx])]
dev = [(vec, targ) for targ, vec in  zip(twt['target'][dev_idx], twt['vec'][dev_idx])]

end = time.time()
print('Preprocessing finished in {} seconds'.format(end-start))

net = dan.Net(hiddenDim = 52)
net.train(train, dev, verbose = False)

torch.save(net.state_dict(), 'C:/Users/jack1/Documents/CS5304-Final-Project-Data-Science')
'''
net = dan.Net(hiddenDim = 52)
net.load_state_dict(torch.load( 'C:/Users/jack1/Documents/CS5304-Final-Project-Data-Science/net_dict.pt'))

print('\n\t REAL OR FAKE: a proof-of-concept\n\tby Irene, Aayushi and Jack\n\n')
print('Enter "exit" at any time to shut down the application.')
response = 'xxx'
while not response == 'exit':
    response = input('Please enter a tweet: ')
    text = [response]
    proc_text = [preprocess.processing(i) for i in text]
    vec = [preprocess.tweet_vec(tweet, word2vec) for tweet in proc_text]
    
    _, y_stars = net.get_eval_data(vec, mode = 'test')
    if y_stars[0] == 0:
        print('non-Disaster')
    elif y_stars[0] == 1:
        print('Disaster') 

print('Thank you for using REAL OR FAKE!')