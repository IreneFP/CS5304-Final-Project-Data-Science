import pandas as pd
import numpy as np

import re
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def processing (sentence):
    """
    given: a sentence in the form of a string
    returns: a list of constituent words after removing numbers, special characters,
        stop words, etc.
    """
    stop_words = stopwords.words('english')
    porter = PorterStemmer()

    result = sentence.lower() #Lower case 
    result = re.sub(r'\d+', '', result) #Removing numbers
    result = result.translate(str.maketrans('', '', string.punctuation)) #Remove weird characters
    result = result.strip() #Eliminate blanks from begining and end of setences
    result = result.split() #Separate into words
    result = [w for w in result if not w in stop_words] #Eliminate stop_words
    # result = [porter.stem(word) for word in result] #Stem Words
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







