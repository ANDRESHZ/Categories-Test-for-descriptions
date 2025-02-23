import argparse
import os
import shutil
import sys
from os.path import join as Une
from os.path import split as Sep
from os.path import exists as Ex
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
import json
from langchain_core.documents import Document
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec, KeyedVectors
import re
from tqdm import tqdm
from sys import platform
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
RUTACHROMA=RUTABASE+"WordBags/VectorDB_W2V/"
RUTACHROMA_PRODUCTS=RUTABASE+"WordBags/VectorDB_Products_W2V/"
COL_NAME_PROD="prTest"
local_llm="FN_GoogleNews-vectors-negative300" if os.getenv('LLM_MODEL')==None else os.getenv('LLM_MODEL')
WordEmbModels= ["FN_conceptnet-numberbatch-300-17-06","FN_fasttext-wiki-news-subwords-300"]

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

def chunk_sentence(sentence, max_length=2000):
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
    RealSenten=[]
    sentences2 = []
    idSenten2=[]
    # Process categories
    print("Processing categories...")
    for handle in tqdm(categories_df['cleanHandle']):
        clean_handle = clean_text(str(handle),cat=True)
        if clean_handle:
            sentences.extend(chunk_sentence(clean_handle))
            RealSenten.append(handle)
    
    # Process products
    print("Processing products...")
    for _, row in tqdm(products_df.iterrows()):
        idact=str(row["uniqueIdentifier"])
        combined_text = ' '.join([
            str(row['metaTitle']),
            # str(row['metaDescription']),
            str(row['vendorCategory']).lower().replace("none","").replace("nan","")
        ])
        clean_text_result = clean_text(combined_text,cat=False)
        if clean_text_result:
            sentences2.extend(chunk_sentence(clean_text_result))
            idSenten2.append(idact)
            
    
    return [sentences,RealSenten,sentences2,idSenten2]

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
    local_llm2=local_llm if args.model is None else args.model
    colectionName=local_llm2.replace(":","_").replace("/","-")
    # File paths
    # PRETRAINED_PATH = "Y:/bins/GoogleNews-vectors-negative300.bin"
    sentencesALL = prepare_sentences(FILE_CATEG, FILE_PRODU)
    
    
   
    
    doclist=[]
    indi=0
    ORDTa=sentencesALL[1]
    for row in sentencesALL[0]:
        orgdtALL=ORDTa[indi]
        orgdt=str(orgdtALL).split("/")
        doclist.append(Document(page_content=row,
                                metadata={"source": Sep(FILE_CATEG)[-1],
                                "clase":str(orgdtALL),"nivel":len(orgdt),"index":indi,"catType":"ALL"}))
        doclist.append(Document(page_content=orgdtALL.split(' for ')[-1],
                                metadata={"source": Sep(FILE_CATEG)[-1],
                                "clase":str(orgdtALL),"nivel":len(orgdt),"index":indi,"catType":"LAST"}))
        indi=indi+1
    add_to_chroma_Multiple(chunks=doclist,Nsim=False,embfunc=None,MODEL=local_llm2,embModels=WordEmbModels)
    # model2=  Word2Vec.load
    # model3=
    # Your texts
    model1 = Word2Vec.load(f"{local_llm}.model")
    texts = [
        "Your first text here",
        "Your second text here",
        "Your third text here"
    ]
    vectors, processed_words = texts_to_vectors(model, texts)
    print(np.shape(vectors), processed_words)

    return True
def add_to_chroma_Multiple(chunks: list[Document],Nsim=True,embfunc=None,MODEL=local_llm,pathChroma="",embModels=["FN_conceptnet-numberbatch-300-17-06","FN_fasttext-wiki-news-subwords-300"]):
    """Add Documents to Chroma Vector store multiples Embedings

    Args:
        chunks (list[Document]): _description_
        Nsim (bool, optional): shows icons on print. Defaults to True.
        embfunc (OllamaEmbeddings|OpenAIEmbeddings): embedind function.
        MODEL (str, optional): llm name to use. Defaults to Global variable local_llm.
        pathChroma (str, optional): ruta de almacenamiento. empty str use global variable RUTACHROMA
        embModels (list, optional): List of word embedings models. Defaults to 
        openai (bool, optional): use openia o Local. Defaults to False.
        port (str, optional): port of API. Defaults to "11434".
    """
    def useEMBMODEL(embName:str="",chunks_ids:list[Document]=None,ChormaPath=pathChroma):
        #Cargar con diferentes nombres
        dbEmbeds = Chroma(persist_directory=ChormaPath, embedding_function=None,create_collection_if_not_exists=True,collection_name=embName.replace(":","_").replace("/","-"))
        existing_ids = set((dbEmbeds.get(include=[]))["ids"])
        new_chunks2 = []
        lastId=len(chunks_ids)-1
        flag=-1
        print("Check Chunks for: "+str(len(chunks_ids))+" with word embed model >> "+embName.replace(COL_NAME_PROD,""))
        for idchunk in tqdm(range(len(chunks_ids))):
            chunk=chunks_ids[idchunk]
            if chunk.metadata["id"] not in existing_ids:
                new_chunks2.append(chunk)
            if idchunk!=0 and ((idchunk+1)%5000==0 or idchunk>=lastId):
                if len(new_chunks2):
                    flag=10
                    print(("👉🔧 " if Nsim==True else "")+"adding documents "+str(len(new_chunks2))+" from "+str(idchunk+1-len(new_chunks2))+" to "+str(idchunk)+">> "+embName.replace(COL_NAME_PROD,""))
                    new_chunk_ids = [chunk.metadata["id"] for chunk in (new_chunks)]
                    new_chunk_metadata = [chunk.metadata for chunk in (new_chunks)]
                    if pathChroma=="" or pathChroma==RUTACHROMA:
                        new_chunk_clases = [chunk.metadata["clase"] for chunk in (new_chunks2)]
                    else:
                        new_chunk_clases = ["product_"+str(ichunk) for ichunk in range(len(new_chunks2))]
                    model = Word2Vec.load(f"{RUTABASE}WordBags/{embName.replace(COL_NAME_PROD,"")}.model")
                    vectors, processed_words = texts_to_vectors(model, listtext)
                    # db.add_documents(new_chunks, ids=new_chunk_ids,clases=new_chunk_clases)
                    db._collection.add(ids=new_chunk_ids,embeddings=vectors,metadatas=new_chunk_metadata,documents=new_chunks)
                    new_chunks=[]
                    listtext=[]
                    
                    
                    new_chunks2=[]
                else:
                    print(("✅ " if Nsim==True else "")+ "No new documents to add "+str(idchunk+1-len(new_chunks2))+" to "+str(idchunk)+">> "+embName.replace(COL_NAME_PROD,""))
        if flag>=1:
            print(("✨ " if Nsim else "") +"Updated database >> "+embName.replace(COL_NAME_PROD,""))


    # Load the existing database.
    ChormaPath=pathChroma if pathChroma!="" else RUTACHROMA 
    os.makedirs(os.path.dirname(ChormaPath), exist_ok=True)
    chunks_with_ids = calculate_chunk_ids(chunks, True if (pathChroma!="" or pathChroma==RUTACHROMA_PRODUCTS) else False)
    # Calculate Page IDs.
    for embMod in embModels:
        useEMBMODEL(embName=("" if MODEL.find(COL_NAME_PROD)==-1 else COL_NAME_PROD)+embMod,chunks_ids=chunks_with_ids,ChormaPath=ChormaPath)
    db = Chroma(persist_directory=ChormaPath, embedding_function=None,create_collection_if_not_exists=True,collection_name=MODEL.replace(":","_").replace("/","-"))
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of document parts in the DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    lastId=len(chunks_with_ids)-1
    flag=-1
    print("Check Chunks for: "+str(len(chunks_with_ids))+" with word Principal model >> "+MODEL.replace(COL_NAME_PROD,""))
    listIds=[]
    listtext=[]
    for idchunk in tqdm(range(len(chunks_with_ids))):
        chunk=chunks_with_ids[idchunk]
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            listIds.append(chunk.metadata["id"])
            listtext.append(chunk.page_content)
            
        if idchunk!=0 and ((idchunk+1)%5000==0 or idchunk>=lastId):
            if len(new_chunks):
                flag=10
                print(("👉🔧 " if Nsim==True else "")+"aggregating documents "+str(len(new_chunks))+" from "+str(idchunk+1-len(new_chunks))+" to "+str(idchunk)+" of "+str(lastId+1)+" chunks")
                new_chunk_ids = [chunk.metadata["id"] for chunk in (new_chunks)]
                new_chunk_metadata = [chunk.metadata for chunk in (new_chunks)]
                if pathChroma=="" or pathChroma==RUTACHROMA:
                    new_chunk_clases = [chunk.metadata["clase"] for chunk in (new_chunks)]
                else:
                    new_chunk_clases = ["product_"+str(ichunk) for ichunk in range(len(new_chunks))]
                model = Word2Vec.load(f"{RUTABASE}WordBags/{MODEL.replace(COL_NAME_PROD,"")}.model")
                vectors, processed_words = texts_to_vectors(model, listtext)
                # db.add_documents(new_chunks, ids=new_chunk_ids,clases=new_chunk_clases)
                db._collection.add(ids=new_chunk_ids,embeddings=vectors,metadatas=new_chunk_metadata,documents=new_chunks)
                new_chunks=[]
                listIds=[]
                listtext=[]
            else:
                print(("✅ " if Nsim==True else "")+ "No new documents to add "+str(idchunk+1-len(new_chunks))+" to "+str(idchunk))
    if  flag>=1:
        print(("✨ " if Nsim else "") +"Updated database >> Principal Model"+MODEL.replace(COL_NAME_PROD,""))

def calculate_chunk_ids(chunks,IdSimple:bool=False):
    """create a list of chunk and ids, add metadata

    Args:
        chunks (List[Documents]): parts of text
        IdSimple (bool, optional): _description_. Defaults to False.

    Returns:
        List[Documents]: all chunks with metadata
    """
    #identificador en DB "VectorDB/title.pdf:index:niveles"
    last_page_id = None
    current_chunk_index = 0
    print("Calculating Chunks IDs")
    for chunk in tqdm(chunks):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("index")
        nivel=chunk.metadata.get("nivel")
        if IdSimple:
            current_page_id = f"{source.split('/')[-1]}{page}"
        else:
            current_page_id = f"{source.split('/')[-1]}:{page}:{nivel}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        if IdSimple:
            chunk_id=current_page_id
        else: 
            chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
    return chunks


if __name__ == "__main__":
    main()
