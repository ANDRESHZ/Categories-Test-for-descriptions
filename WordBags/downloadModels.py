import numpy as np
import nltk
from nltk import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
print("...")
print(api.load("word2vec-google-news-300",return_path=True))
print("...")
print(api.load("word2vec-google-news-300",return_path=True))
print("...")
print(api.load("glove-twitter-200",return_path=True))
print("...")
print(api.load("fasttext-wiki-news-subwords-300",return_path=True))
print("...")
print(api.load('conceptnet-numberbatch-17-06-300',return_path=True))
print("...")
print(api.load('glove-wiki-gigaword-300',return_path=True))
print("...")
nltk.download('punkt_tab')