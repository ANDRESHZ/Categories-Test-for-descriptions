import numpy as np
import nltk
from nltk import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
# model = api.load("word2vec-google-news-300")
model = api.load("glove-twitter-50")


nltk.download('punkt_tab')

fine_tune_sentences = ["I like to eat broccoli and bananas.",
                       "I ate a banana and spinach smoothie for breakfast.",
                       "Chinchillas and kittens are cute."]
tokenized_sentences = [word_tokenize(sentence)
                       for sentence in fine_tune_sentences]
# Output: tokenized_sentences[0] = ['I', 'like', 'to', 'eat', 'broccoli', 'and', 'bananas', '.']

# Load pretrained model for finetuning
pretrained_model_path = 'GoogleNews-vectors-negative300'
pretrained_model = KeyedVectors.load(
    pretrained_model_path, binary=True)
pretrained_vocab = list(pretrained_model.index_to_key)

# Create new model
model = Word2Vec(vector_size=pretrained_model.vector_size)
model.build_vocab(tokenized_sentences)
total_examples = model.corpus_count
model_vocab = list(model.wv.index_to_key)

# Load pretrained model's vocabulary.
model.build_vocab([pretrained_vocab], update=True)

# vectors_lockf property is initialize in __init__ method of Word2Vec class.
# We are using build_vocab method to update vocabulary of model, so we need initialize vectors_lockf property manually.
model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)

# load pretrained model's embeddings into model
model.wv.intersect_word2vec_format(pretrained_model_path, binary=False)

model.train(tokenized_sentences,
            total_examples=total_examples, epochs=model.epochs)