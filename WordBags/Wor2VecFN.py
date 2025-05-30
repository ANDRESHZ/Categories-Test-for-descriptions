import numpy as np
import nltk
from nltk import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
# print(api.load("word2vec-google-news-300",return_path=True))
print(api.load("glove-twitter-200",return_path=True))
print(api.load("fasttext-wiki-news-subwords-300",return_path=True))
print(api.load('conceptnet-numberbatch-17-06-300',return_path=True))
print(api.load('glove-wiki-gigaword-300',return_path=True))
nltk.download('punkt_tab')

fine_tune_sentences = ["I like to eat broccoli and bananas.",
                       "I ate a banana and spinach smoothie for breakfast.",
                       "Chinchillas and kittens are cute."]
tokenized_sentences = [word_tokenize(sentence)
                       for sentence in fine_tune_sentences]
# Output: tokenized_sentences[0] = ['I', 'like', 'to', 'eat', 'broccoli', 'and', 'bananas', '.']
print(tokenized_sentences)
# Load pretrained model for finetuning
# pretrained_model_path = 'GoogleNews-vectors-negative300'
pretrained_model_path = api.load("glove-twitter-25",return_path=True).replace("\\","/")
pretrained_model_path = "Y:/bins/GoogleNews-vectors-negative300.bin"
# Load the model
pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path,binary=True,)
# model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=False)
pretrained_vocab = list(pretrained_model.index_to_key)

# Create new model
model = Word2Vec(vector_size=pretrained_model.vector_size,min_count=4)
model.build_vocab(tokenized_sentences)
total_examples = model.corpus_count
model_vocab = list(model.wv.index_to_key)
print(total_examples,np.shape(model_vocab))
# Load pretrained model's vocabulary.
model.build_vocab([pretrained_vocab])

# vectors_lockf property is initialize in __init__ method of Word2Vec class.
# We are using build_vocab method to update vocabulary of model, so we need initialize vectors_lockf property manually.
model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)

# load pretrained model's embeddings into model
model.wv.intersect_word2vec_format(pretrained_model_path, binary=True)

model.train(tokenized_sentences,total_examples=total_examples, epochs=model.epochs)