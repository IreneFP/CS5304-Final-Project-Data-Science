import pandas as pd
import numpy as np

import re
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def processing (sentence, lower = True, stop_words = False, stem = False):
    """
    given: a sentence in the form of a string
    returns: a list of constituent   words after removing numbers, special characters,
        stop words, etc.
    """
    stop_words = stopwords.words('english')
    porter = PorterStemmer()

    result = sentence
    if lower:
        result = result.lower() #Lower case 
    result = re.sub(r'\d+', '', result) #Removing numbers
    result = result.translate(str.maketrans('', '', string.punctuation)) #Remove weird characters
    result = result.strip() #Eliminate blanks from begining and end of setences
    result = result.split() #Separate into words
    if not stop_words:
        result = [w for w in result if not w in stop_words] #Eliminate stop_words
    if stem:
        result = [porter.stem(word) for word in result] #Stem Words
    return (result)

def tweet_vec (tweet, word2vec):
    """
    given: a list of words and an embedding file
    returns: a list of the corresponding embeddings 
    """
    word_vecs = [word2vec.get_vector(w) for w in tweet if w in word2vec.vocab]
    if len(word_vecs) >= 1:
        return np.stack(word_vecs).mean(0)
    else:
        return None

if __name__ == '__main__':
    """
    testing optional arguments to preprocess
    """
    import pandas as pd
    import random
    twt = pd.read_csv('train.csv')
    twt = twt.set_index('id')
    text = twt['text'].to_list()
    prep_text = [processing(i, lower = False) for i in text]
    print(random.choice(prep_text))
    print(random.choice(prep_text)) 
    print(random.choice(prep_text))



