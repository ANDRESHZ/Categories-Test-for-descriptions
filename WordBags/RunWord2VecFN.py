import argparse
import os
import shutil
import sys
from os.path import join as Une
from os.path import split as Sep
from os.path import exists as Ex
import pandas as pd
from tqdm import tqdm
import json
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from sys import platform
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
import re
from tqdm import tqdm
import sys
if platform == "linux" or platform == "linux2":
   # linux
   addLetter=""
elif platform == "darwin":
    # OS X
    addLetter=""
elif platform == "win32":
    # Windows...
    addLetter="X:"
RUTABASE=addLetter+"/LILO/LILO-Categories-Test/"
FILE_CATEG=Une(RUTABASE,"Resources/categories.csv")
FILE_PRODU=Une(RUTABASE,"Resources/products.csv")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text,cat=False):
    """Clean and preprocess text"""
    if isinstance(text, str):
        if cat:
            text=text.replace('--',' to ').replace('-',' ').replace('/',' for ')
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)
    return ''

def chunk_sentence(sentence, max_length=50):
    """Split sentence into chunks of maximum length while preserving words"""
    words = sentence.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(current_chunk) < max_length:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def prepare_sentences(categories_file, products_file):
    """Prepare and clean sentences from both CSV files"""
    # Read CSV files
    categories_df = pd.read_csv(categories_file,sep=";",header=0)
    products_df = pd.read_csv(products_file,sep=";",header=0,lineterminator='\r',usecols=["uniqueIdentifier","metaTitle","metaDescription","metaSpecs","vendorCategory","vendor"])
    
    sentences = []
    
    # Process categories
    print("Processing categories...")
    for handle in tqdm(categories_df['cleanHandle']):
        clean_handle = clean_text(str(handle),cat=True)
        if clean_handle:
            sentences.extend(chunk_sentence(clean_handle))
    
    # Process products
    print("Processing products...")
    for _, row in tqdm(products_df.iterrows()):
        combined_text = ' '.join([
            str(row['metaTitle']),
            str(row['metaDescription']),
            str(row['vendorCategory'])
        ])
        clean_text_result = clean_text(combined_text)
        if clean_text_result:
            sentences.extend(chunk_sentence(clean_text_result))
    
    return [word_tokenize(sent) for sent in sentences]

def finetune_word2vec(pretrained_path, sentences, output_path):
    """Finetune the Word2Vec model"""
    print("Loading pretrained model...")
    if pretrained_path.lower().find(".txt"):
        pretrained_model = Word2Vec.load(pretrained_path)
    else:
        pretrained_model = KeyedVectors.load_word2vec_format(
            pretrained_path,
            binary=True if pretrained_path.lower().find(".bin") else False
        )
    
    print("Initializing new model...")
    model = Word2Vec(
        vector_size=pretrained_model.vector_size,
        min_count=2
    )
    
    print("Building vocabulary...")
    model.build_vocab(sentences)
    total_examples = model.corpus_count
    
    print("Loading pretrained vocabulary...")
    model.build_vocab([list(pretrained_model.key_to_index.keys())], update=True)
    
    # Initialize vectors_lockf
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
    print("Loading pretrained embeddings...")
    model.wv.intersect_word2vec_format(pretrained_path, binary=True, lockf=1.0)
    print("Training model...")
    model.train(
        sentences,
        total_examples=total_examples,
        epochs=7
    )
    # Save the model
    print("Saving model...")
    model.save(f"{output_path}{Sep((pretrained_path)[1]).replace(".bin","").replace(".txt","")}.model")
    # model.wv.save_word2vec_format(f"{output_path}_word2vec.bin", binary=True)
    
    return model
def texts_to_vectors(model, texts, vector_size=300):
    """
    Convert a list of texts into their vector representations
    
    Parameters:
    - model: trained Word2Vec model
    - texts: list of strings to convert
    - vector_size: dimension of word vectors (default 300 for Google News vectors)
    
    Returns:
    - List of document vectors
    - List of words that were processed
    """
    def text_to_vector(text,cat=False):
        # Clean and tokenize the text
        cleaned_text = clean_text(text,cat=cat)
        words = word_tokenize(cleaned_text)
        
        # Get vectors for each word and average them
        word_vectors = []
        existing_words = []
        
        for word in words:
            try:
                vector = model.wv[word]
                word_vectors.append(vector)
                existing_words.append(word)
            except KeyError:
                continue  # Skip words not in vocabulary
                
        if word_vectors:
            # Return average vector if we found any words
            return np.mean(word_vectors, axis=0), existing_words
        else:
            # Return zero vector if no words were found
            return np.zeros(vector_size), existing_words

    # Process all texts
    vectors = []
    processed_words = []
    
    for text in tqdm(texts, desc="Converting texts to vectors"):
        vector, words = text_to_vector(text)
        vectors.append(vector)
        processed_words.append(words)
        
    return np.array(vectors), processed_words

def main():
    # File paths
    # PRETRAINED_PATH = "Y:/bins/GoogleNews-vectors-negative300.bin"
    PRETRAINED_PATH = "Y:/bins/glove.twitter.27B.200d.txt"
    OUTPUT_PATH = "WordBags/FN_"
    
    # Prepare sentences
    print("Preparing sentences...")
    sentences = prepare_sentences(FILE_CATEG, FILE_PRODU)
    
    # Finetune model
    print("Finetuning model...")
    model = finetune_word2vec(PRETRAINED_PATH, sentences, OUTPUT_PATH)
    
    print("Process completed successfully!")
    
    model = Word2Vec.load(f"{OUTPUT_PATH}{Sep((PRETRAINED_PATH)[1]).replace(".bin","").replace(".txt","")}.model")

    # Your texts
    texts = [
        "Your first text here",
        "Your second text here",
        "Your third text here"
    ]
    vectors, processed_words = texts_to_vectors(model, texts)
    print(np.shape(vectors), processed_words)

    return model

if __name__ == "__main__":
    main()
