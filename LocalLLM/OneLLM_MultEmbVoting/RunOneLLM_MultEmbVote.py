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
RUTACHROMA=RUTABASE+"LocalLLM/VectorDB2/"
RUTACHROMA_PRODUCTS=RUTABASE+"LocalLLM/VectorDB_Products/"
COL_NAME_PROD="prodTest"
local_llm="deepseek-r1:8b" if os.getenv('LLM_MODEL')==None else os.getenv('LLM_MODEL')
WordEmbModels=["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"]
#import the module from other folder
sys.path.insert(1, RUTABASE+"LocalLLM")
import get_embedding_function as emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=10, help="puerto local")
    parser.add_argument("--nosymbols", action="store_true", help="usar simbolos en el cdm")
    parser.add_argument("--openai", action="store_true", help="usar open ai embeddings Falso si es Local")
    parser.add_argument("--model",type=str, default=None, help="modelo a usar en Embedings")
    parser.add_argument("--port", type=str, default="11434", help="puerto local")
    parser.add_argument("--text", type=str, default="", help="texto a comvertir")
    parser.add_argument("--reset", action="store_true", help="limpiar DB")
    parser.add_argument("--delcolection", action="store_true", help="limpiar coleccion de Chroma")
    parser.add_argument("--delproducts", action="store_true", help="limpiar coleccion de Chroma de porductos")
    args = parser.parse_args()
    symbols=(args.nosymbols)==False
    
    
    local_llm2=local_llm if args.model is None else args.model
    colectionName=local_llm2.replace(":","_").replace("/","-")
    
    if args.openai:
        print("Using OpenAI embeddings")
        embedings = emb.get_OPENIA_embedding_function(local_llm2)
    else:
        print("Using local embeddings")
        embedings = emb.get_embedding_function_local(local_llm2,port=args.port)
        
    if args.reset:#==False:
        print(("🔥💣💥 " if symbols else "") +"Cleaning Data Base")
        clear_database_chroma(RUTACHROMA)
        TODO=1
    elif args.delcolection:#==False:
        print(("🔥 " if symbols else "") +"Delete Collection")
        dbdel = Chroma(persist_directory=RUTACHROMA, embedding_function=embedings,collection_name=colectionName)
        dbdel.delete_collection()
        del dbdel
    if args.delproducts:#:==False:
        print(("🔥💽 " if symbols else "") +"Deleting PRODUCT Collection")
        clear_database_chroma(RUTACHROMA_PRODUCTS)
    
    print("Work with >>>>>> 🏷",local_llm2)
    
    dataCategOrg=pd.read_csv(FILE_CATEG,sep=";",header=0)
    dataCateg=dataCategOrg.copy()
    colname=list(dataCateg)[0]
    dataCateg[colname]=dataCategOrg[colname].str.replace('--',' to ').str.replace('-',' ').str.replace('/',' for ')
    
    doclist=[]
    indi=0
    for row in dataCateg[colname]:
        orgdtALL=dataCategOrg[colname].iloc[indi]
        orgdt=str(orgdtALL).split("/")
        doclist.append(Document(page_content=row,
                                metadata={"source": Sep(FILE_CATEG)[-1],
                                "clase":str(orgdtALL),"nivel":len(orgdt),"index":indi,"catType":"ALL"}))
        doclist.append(Document(page_content=row.split(' for ')[-1],
                                metadata={"source": Sep(FILE_CATEG)[-1],
                                "clase":str(orgdtALL),"nivel":len(orgdt),"index":indi,"catType":"LAST"}))
        indi=indi+1
    WordEmbModels=["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"]
    add_to_chroma_Multiple(chunks=doclist,Nsim=symbols,embfunc=embedings,MODEL=local_llm2,embModels=WordEmbModels)
    textListALL=[]
    ListUniqueIdentifier=[]
    if args.text!="":#ENTRADA DE TEXTO MANUAL
        print(("💻 " if symbols else "") +"text embeddings from CMD")
        textListALL = [args.text]
    else:
        print(("📝 " if symbols else "") +"Loading Products")
        dataProd=pd.read_csv(FILE_PRODU,sep=";",header=0,lineterminator='\r',usecols=["uniqueIdentifier","metaTitle","metaDescription","metaSpecs","vendorCategory","vendor"])
        for idAct,query_text in (dataProd[["uniqueIdentifier","metaTitle","vendorCategory"]].iterrows()):
            textsList=str(query_text["metaTitle"]).split(",")
            data2=("['"+str(query_text["vendorCategory"])+"']").replace("['[","[").replace("]']","]").replace("''","'")
            dataEval=eval("[None]" if data2=="['nan']" else data2)
            if len(textsList[0])<=2:
                text1= " ".join(textsList[:min(len(textsList),2)])
            else:
                text1=textsList[0]
            textsList=text1+((" categorize as "+" or ".join(dataEval)) if str(dataEval[0])!="None" else "")
            textListALL.append(textsList)
            ListUniqueIdentifier.append(str(query_text["uniqueIdentifier"]))
    
    
    print(("💽" if symbols else "") +"obtaining product embedinds")
    doclist=[]
    for iPr in tqdm(range(len(textListALL))):
        textAct=textListALL[iPr]
        doclist.append(Document(page_content=textAct,metadata={"source": Sep(FILE_PRODU)[-1],"index":iPr}))
    
    # doclist=doclist[0:201]
    
    add_to_chroma_Multiple(chunks=doclist,Nsim=symbols,embfunc=embedings,MODEL=(COL_NAME_PROD+colectionName),pathChroma=RUTACHROMA_PRODUCTS,embModels=WordEmbModels)
    
    print(("💻 📝" if symbols else "") +"obtaining product similarities")
    db = Chroma(persist_directory=RUTACHROMA,collection_name=colectionName,create_collection_if_not_exists=False)
    dbProd = Chroma(persist_directory=RUTACHROMA_PRODUCTS,collection_name=(COL_NAME_PROD+colectionName),create_collection_if_not_exists=False)
    DBs=[]
    DBsProd=[]
    for wrodMod in WordEmbModels:
        DBs.append(Chroma(persist_directory=RUTACHROMA,collection_name=(wrodMod).replace(":","_").replace("/","-"),create_collection_if_not_exists=False))
        DBsProd.append(Chroma(persist_directory=RUTACHROMA_PRODUCTS,collection_name=(COL_NAME_PROD+wrodMod).replace(":","_").replace("/","-"),create_collection_if_not_exists=False))

    nameFileProd=Sep(FILE_PRODU)[-1]
    st=time.time()
    ListCategories=[]
    for iPr in tqdm(range(len(doclist))):
        try:
            del results,results2,results3,results4
        except:
            pass
        # Search the DB.
        Ksamples=5
        Clases=[]
        voteBox=None
        for iDb in range(len(WordEmbModels)):
            vectorEmb=DBsProd[iDb].get(ids=nameFileProd+f'{iPr}',include=['embeddings'])
            results=DBs[iDb].similarity_search_by_vector_with_relevance_scores(vectorEmb["embeddings"], k=Ksamples,filter={"catType":"ALL"})
            results2 = DBs[iDb].similarity_search_by_vector_with_relevance_scores(vectorEmb["embeddings"], k=Ksamples,filter={"catType":"LAST"})
            results3=DBs[iDb].max_marginal_relevance_search_by_vector(vectorEmb["embeddings"], k=Ksamples,filter={"catType":"ALL"})
            results4=DBs[iDb].max_marginal_relevance_search_by_vector(vectorEmb["embeddings"], k=Ksamples,filter={"catType":"LAST"})
            for res in [results,results2,results3,results4]:
                votes=Ksamples
                for r in res:
                    try:
                        if np.shape(r)[0]==2:
                            Clases.append([r[0].metadata["clase"],votes])
                    except:
                        Clases.append([r.metadata["clase"],votes])
                    votes=votes-0.45
            del results,results2,results3,results4
        
        vector=dbProd.get(ids=nameFileProd+f'{iPr}',include=['embeddings'])
        results = db.similarity_search_by_vector_with_relevance_scores(vector["embeddings"], k=Ksamples,filter={"catType":"ALL"})
        results2 = db.similarity_search_by_vector_with_relevance_scores(vector["embeddings"], k=Ksamples,filter={"catType":"LAST"})
        results3=db.max_marginal_relevance_search_by_vector(vector["embeddings"], k=Ksamples,filter={"catType":"ALL"})
        results4=db.max_marginal_relevance_search_by_vector(vector["embeddings"], k=Ksamples,filter={"catType":"LAST"})
        
        for res in [results,results2,results3,results4]:
            votes=Ksamples
            for r in res:
                try:
                    if np.shape(r)[0]==2:
                        Clases.append([r[0].metadata["clase"],votes])
                except:
                    Clases.append([r.metadata["clase"],votes])
                votes=votes-0.45           
            
        voteBox=dict.fromkeys(set(np.array(Clases)[:,0]))

        for Clase in Clases:
            if voteBox[Clase[0]]==None:
                voteBox[Clase[0]]=Clase[1]
            else:
                voteBox[Clase[0]]+=Clase[1]
        ListCategories.append([kv[0] for kv in voteBox.items() if kv[1] == max(voteBox.values())][0])
        if iPr%4020==0:
            # print(vector["embeddings"],"|",results2[0][1],"|",results2[0][0])
            print(textListALL[iPr],"|",results2[0][0].metadata["clase"],"|",results2[0][0].page_content)
            if iPr==0:
                print(">>>>>",voteBox,"<<<<<")
            # break
        
    with open('LocalLLM/OneLLM_MultEmbVoting/answer_template.json', 'w') as file:
        json.dump([{"_id": ids, "category": categorie} for ids, categorie in zip(ListUniqueIdentifier, ListCategories)], file)
    print("time=",time.time()-st)
    return True

def extract_types(text,RGEX=r' type: (.*?) for '):
    """etract type text usin regex

    Args:
        text (str): text to proccess
        RGEX (regexp, optional): _description_. Defaults to r' type: (.*?) for '.

    Returns:
        str: union
    """
    import re
    matches = re.findall(RGEX, text)
    return '  '.join(matches) if matches else ''

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

def clear_database_chroma(path=""):
    """Delete all data from Chroma database.
    """
    path=RUTACHROMA if path=="" else path
    print(">>>>>",path,"<<<")
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print(f"Path {path} does not exist")

def add_to_chroma_Multiple(chunks: list[Document],Nsim=True,embfunc=None,MODEL=local_llm,pathChroma="",embModels=["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"],openai:bool=False,port:str="11434"):
    """Add Documents to Chroma Vector store multiples Embedings

    Args:
        chunks (list[Document]): _description_
        Nsim (bool, optional): shows icons on print. Defaults to True.
        embfunc (OllamaEmbeddings|OpenAIEmbeddings): embedind function.
        MODEL (str, optional): llm name to use. Defaults to Global variable local_llm.
        pathChroma (str, optional): ruta de almacenamiento. empty str use global variable RUTACHROMA
        embModels (list, optional): List of word embedings models. Defaults to ["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"].
        openai (bool, optional): use openia o Local. Defaults to False.
        port (str, optional): port of API. Defaults to "11434".
    """
    def useEMBMODEL(embName:str="",chunks_ids:list[Document]=None,ChormaPath=pathChroma):
        #Cargar con diferentes nombres
        dbEmbeds = Chroma(persist_directory=ChormaPath, embedding_function=emb.get_OPENIA_embedding_function(embName.replace(COL_NAME_PROD,"")) if openai else emb.get_embedding_function_local(embName.replace(COL_NAME_PROD,""),port=port),create_collection_if_not_exists=True,collection_name=embName.replace(":","_").replace("/","-"))
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
                    new_chunk_ids = [chunk.metadata["id"] for chunk in (new_chunks2)]
                    if pathChroma=="" or pathChroma==RUTACHROMA:
                        new_chunk_clases = [chunk.metadata["clase"] for chunk in (new_chunks2)]
                    else:
                        new_chunk_clases = ["product_"+str(ichunk) for ichunk in range(len(new_chunks2))]
                    dbEmbeds.add_documents(new_chunks2, ids=new_chunk_ids,clases=new_chunk_clases)
                    
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
    db = Chroma(persist_directory=ChormaPath, embedding_function=embfunc,create_collection_if_not_exists=True,collection_name=MODEL.replace(":","_").replace("/","-"))
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of document parts in the DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    lastId=len(chunks_with_ids)-1
    flag=-1
    print("Check Chunks for: "+str(len(chunks_with_ids))+" with word Principal model >> "+MODEL.replace(COL_NAME_PROD,""))
    for idchunk in tqdm(range(len(chunks_with_ids))):
        chunk=chunks_with_ids[idchunk]
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
        if idchunk!=0 and ((idchunk+1)%5000==0 or idchunk>=lastId):
            if len(new_chunks):
                flag=10
                print(("👉🔧 " if Nsim==True else "")+"aggregating documents "+str(len(new_chunks))+" from "+str(idchunk+1-len(new_chunks))+" to "+str(idchunk)+" of "+str(lastId+1)+" chunks")
                new_chunk_ids = [chunk.metadata["id"] for chunk in (new_chunks)]
                if pathChroma=="" or pathChroma==RUTACHROMA:
                    new_chunk_clases = [chunk.metadata["clase"] for chunk in (new_chunks)]
                else:
                    new_chunk_clases = ["product_"+str(ichunk) for ichunk in range(len(new_chunks))]
                db.add_documents(new_chunks, ids=new_chunk_ids,clases=new_chunk_clases)
                new_chunks=[]
            else:
                print(("✅ " if Nsim==True else "")+ "No new documents to add "+str(idchunk+1-len(new_chunks))+" to "+str(idchunk))
    if  flag>=1:
        print(("✨ " if Nsim else "") +"Updated database >> Principal Model"+MODEL.replace(COL_NAME_PROD,""))

def add_to_chroma(chunks: list[Document],Nsim=True,embfunc=None,MODEL=local_llm,pathChroma=""):
    """Add Documents to Chroma Vector store

    Args:
        chunks (list[Document]): _description_
        Nsim (bool, optional): shows icons on print. Defaults to True.
        embfunc (OllamaEmbeddings|OpenAIEmbeddings): embedind function.
        MODEL (str, optional): llm name to use. Defaults to Global variable local_llm.
        pathChroma (str, optional): ruta de almacenamiento. empty str use global variable RUTACHROMA
    """
    # Load the existing database.
    ChormaPath=pathChroma if pathChroma!="" else RUTACHROMA 
    os.makedirs(os.path.dirname(ChormaPath), exist_ok=True)
    db = Chroma(persist_directory=ChormaPath, embedding_function=embfunc,create_collection_if_not_exists=True,collection_name=MODEL.replace(":","_").replace("/","-"))
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks, True if (pathChroma!="" or pathChroma==RUTACHROMA_PRODUCTS) else False)
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of document parts in the DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    lastId=len(chunks_with_ids)-1
    print("Check Chunks for: "+str(len(chunks_with_ids)))
    for idchunk in tqdm(range(len(chunks_with_ids))):
        chunk=chunks_with_ids[idchunk]
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
        if idchunk!=0 and ((idchunk+1)%5000==0 or idchunk>=lastId):
            if len(new_chunks):
                print(("👉🔧 " if Nsim==True else "")+"aggregating documents "+str(len(new_chunks))+" from "+str(idchunk+1-len(new_chunks))+" to "+str(idchunk)+" of "+str(lastId+1)+" chunks")
                new_chunk_ids = [chunk.metadata["id"] for chunk in (new_chunks)]
                if pathChroma=="" or pathChroma==RUTACHROMA:
                    new_chunk_clases = [chunk.metadata["clase"] for chunk in (new_chunks)]
                else:
                    new_chunk_clases = ["product_"+str(ichunk) for ichunk in range(len(new_chunks))]
                db.add_documents(new_chunks, ids=new_chunk_ids,clases=new_chunk_clases)
                print(("✨ " if Nsim else "") +"Updated database")
                new_chunks=[]
            else:
                print(("✅ " if Nsim==True else "")+ "No new documents to add "+str(idchunk+1-len(new_chunks))+" to "+str(idchunk))

if __name__ == "__main__":
    main()
    