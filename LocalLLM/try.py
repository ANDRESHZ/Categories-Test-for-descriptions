import argparse
import os
import shutil
import get_embedding_function as emb
from os.path import join as Une
from os.path import split as Sep
from os.path import exists as Ex
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document

RUTABASE="X:\\LILO\\LILO-Categories-Test\\"
FILE_CATEG=Une(RUTABASE,"Resources\\categories.csv")
FILE_PRODU=Une(RUTABASE,"Resources\\products.csv")
RUTACHROMA=RUTABASE+"\\LocalLLM\\VectorDB"
local_llm="llama3.1" if os.getenv('LLM_MODEL')==None else os.getenv('LLM_MODEL') 
def main():
    print(local_llm)
    parser = argparse.ArgumentParser()
    parser.add_argument("--nosymbols", action="store_true", help="usar simbolos en el cdm")
    parser.add_argument("--openai", action="store_true", help="usar open ai embeddings Falso si es Local")
    parser.add_argument("--model",type=str, default=None, help="modelo a usar en Embedings")
    parser.add_argument("--port", type=str, default="11434", help="puerto local")
    parser.add_argument("--text", type=str, default="", help="texto a comvertir")
    parser.add_argument("--reset", action="store_true", help="limpiar DB")
    parser.add_argument("--delcolection", action="store_true", help="limpiar DB")
    args = parser.parse_args()
    
    Texto=args.text
    symbols=(args.nosymbols)==False
   
    
    colectionName=local_llm.replace(":","_").replace("/","-")       
    local_llm2=local_llm if args.model is None else args.model
    if args.openai:
        print("Usando embeddings de OpenAI")
        embedings = emb.get_OPENIA_embedding_function(local_llm2)
    else:
        print("Usando embeddings locales")
        embedings = emb.get_embedding_function_local(local_llm2,port=args.port)
        
    if args.reset:#==False:
        print(("✨ " if symbols else "") +"Limpiando Base de datos")
        clear_database_chroma()
        TODO=1
    elif args.delcolection:#==False:
        print(("✨ " if symbols else "") +"Borrando Coleccion")
        dbdel = Chroma(persist_directory=RUTACHROMA, embedding_function=embedings,collection_name=colectionName)
        dbdel.delete_collection()
        del dbdel
        
    print(">>>>>>",local_llm2)
    result=embedings.embed_query(Texto)
    print(Texto, str(result)[:100])
    print(FILE_CATEG)
    dataCategOrg=pd.read_csv(FILE_CATEG,sep=";",header=0)
    dataCateg=dataCategOrg.copy()
    colname=list(dataCateg)[0]
    dataCateg[colname]=dataCategOrg[colname].str.replace('--',' type: ').str.replace('-',' ').str.replace('/',' for ')
    dataCateg["extracted"] = dataCateg[colname].str.extract(r'type: (.*?) for ')
    dataCateg["extracted2"] = dataCateg[colname].str.extract(r' for (.*?) type: ')
    # dataCateg["depurado"]=" ".join([vals[:vals.find(" for ")] for vals in dataCateg[colname].str.split(" type: ")])
    dataCategOrg=dataCateg.copy()
    dataCategOrg["depurado"]=dataCategOrg[colname].copy()
    # dataCategOrg["depurado"]=dataCategOrg[colname].apply(extract_types)
    # dataCategOrg["depurado2"]=dataCategOrg[colname].apply(extract_types)
    dataCategOrg["depurado"]=dataCategOrg[colname].apply(lambda x: extract_types(x,r' type: (.*?) for '))
    dataCategOrg["depurado2"]=dataCategOrg[colname].apply(lambda x: extract_types(x,r' for (.*?) type: '))
    print(dataCategOrg["depurado"][:10])
    print(dataCategOrg["depurado2"][:10])
    doclist=[]
    indi=0
    for row in dataCateg[colname]:
        orgdt=dataCategOrg.iloc[indi].str.split("/")
        doclist.append(Document(page_content=row,
                                metadata={"source": FILE_CATEG,"clase":(orgdt[-1])[-1],"nivel":len(orgdt),"index":indi,"catType":"ALL"}))
        doclist.append(Document(page_content=row.split(' for ')[-1],
                                metadata={"source": FILE_CATEG,"clase":(orgdt[-1])[-1],"nivel":len(orgdt),"index":indi,"catType":"LAST"}))
        indi=+1
    print(doclist[-3:])
    print(dataCateg[:5])
    add_to_chroma_Multiple(chunks=doclist,embfunc=embedings,Nsim=symbols,MODEL=local_llm2,
                           embModels=["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"])
    dataProd=pd.read_csv(FILE_PRODU,sep=";",header=0,lineterminator='\r',usecols=["metaTitle","metaDescription","metaSpecs","vendorCategory","vendor"])
    print(repr(dataProd[["metaTitle","vendorCategory"]][2:6]))
    
    for idAct,query_text in (dataProd[["metaTitle","vendorCategory"]][3:11].iterrows()):
        textsList=query_text["metaTitle"].split(",")
        data2=("["+str(query_text["vendorCategory"])+"]").replace("[[","[").replace("]]","]")
        dataEval=eval("[None]" if data2=="[nan]" else data2)
        if len(textsList[0])<=2:
            text1= " ".join(textsList[:min(len(textsList),2)])
        else:
            text1=textsList[0]
        # print(data2,"|",dataEval)
        textsList=text1+((" "+" or ".join(dataEval)) if str(dataEval[0])!="None" else "")
        print(textsList)
        embedding_function = embedings
        db = Chroma(persist_directory=RUTACHROMA, embedding_function=embedding_function,collection_name=colectionName)
        # Search the DB.
        results = db.similarity_search_with_score(textsList, k=10,filter={"catType":"ALL"})
        results2 = db.similarity_search_with_score(textsList, k=10,filter={"catType":"LAST"})
        results3=db.max_marginal_relevance_search(textsList, k=5,filter={"catType":"ALL"})
        results4=db.max_marginal_relevance_search(textsList, k=5,filter={"catType":"LAST"})
        

        
        # results = db.similarity_search_with_relevance_scores(" ".join(textsList[:1]), k=4)
        print("--------------POSIBLES-------------\n","\n".join([doc.page_content+" | "+str(_score) for doc, _score in results]),"\n-LAST CAT-\n","\n".join([doc.page_content+" | "+str(_score) for doc, _score in results2]))
        print("--------------POSIBLES BY RELEVANCE-------------\n","\n".join([doc.page_content for doc in results3]),"\n-LAST CAT-\n","\n".join([doc.page_content for doc in results4]),"\n.......")
        
        dta=db.get(where_document={"$contains": results3[0].page_content})
        print(">>>>>",repr(dta))
        getdata=db._collection.get(include=['embeddings'],where_document={"$contains": results3[0].page_content})
        # print(">>",len(getdata["embeddings"][0]),getdata["embeddings"])
        print(dta["metadatas"])
        print(dta["metadatas"][0]["index"])

    return "END"

def extract_types(text,RGEX=r' type: (.*?) for '):
    import re
    matches = re.findall(RGEX, text)
    return '  '.join(matches) if matches else ''

def add_to_chroma(chunks: list[Document],Nsim=True,embfunc=None,MODEL=local_llm):
    # Load the existing database.
    CHROMA_PATH=RUTACHROMA
    os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embfunc,create_collection_if_not_exists=True,collection_name=MODEL.replace(":","_").replace("/","-"))
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Numero de partes de documentos en la DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    print("Verificar Chunks")
    for chunk in tqdm(chunks_with_ids):
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(("👉 " if Nsim==True else "")+ f"Agregando nuevos documentos: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in tqdm(new_chunks)]
        new_chunk_clases = [chunk.metadata["clase"] for chunk in tqdm(new_chunks)]
        db.add_documents(new_chunks, ids=new_chunk_ids,clases=new_chunk_clases)
        print(("✨ " if Nsim else "") +"Base de datos actualizada")
        # db.persist()
    else:
        print(("✅ " if Nsim==True else "")+ "No hay nuevos documentos para agregar")
        
def add_to_chroma_Multiple(chunks: list[Document],Nsim=True,embfunc=None,MODEL=local_llm,embModels=["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"],openai:bool=False,port:str="11434"):
    def useEMBMODEL(embName:str="",chunks_ids:list[Document]=None):
        #Cargar con diferentes nombres
        new_chunks2 = []
        dbEmbeds = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb.get_OPENIA_embedding_function(embName) if openai else emb.get_embedding_function_local(embName,port=port),create_collection_if_not_exists=True,collection_name=embName.replace(":","_").replace("/","-"))
        existing_ids = set((dbEmbeds.get(include=[]))["ids"])
        for ch in tqdm(chunks_ids):
            if ch.metadata["id"] not in existing_ids:
                new_chunks2.append(ch)
        
        if len(new_chunks2):
            print(("👉 " if Nsim==True else "")+ f"Agregando nuevos documentos: {len(new_chunks2)} a Coleccion>> {str(embName)}")
            dbEmbeds.add_documents(new_chunks2, ids=[chunk.metadata["id"] for chunk in tqdm(new_chunks2)],clases=[chunk.metadata["clase"] for chunk in tqdm(new_chunks2)])
            print(("✨ " if Nsim else "") +"Base de datos actualizada parea Coleccion>> "+embName)
        else:
            print(("✅ " if Nsim==True else "")+ "No hay nuevos documentos para agregar a Coleccion>> "+ embName)


    # Load the existing database.
    CHROMA_PATH=RUTACHROMA
    os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    for embMod in embModels:
        useEMBMODEL(embName=embMod,chunks_ids=chunks_with_ids)
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embfunc,create_collection_if_not_exists=True,collection_name=MODEL.replace(":","_").replace("/","-"))
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Numero de partes de documentos en la DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    print("Verificar Chunks")
    for chunk in tqdm(chunks_with_ids):
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(("👉 " if Nsim==True else "")+ f"Agregando nuevos documentos: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in tqdm(new_chunks)]
        new_chunk_clases = [chunk.metadata["clase"] for chunk in tqdm(new_chunks)]
        db.add_documents(new_chunks, ids=new_chunk_ids,clases=new_chunk_clases)
        print(("✨ " if Nsim else "") +"Base de datos actualizada")
        # db.persist()
    else:
        print(("✅ " if Nsim==True else "")+ "No hay nuevos documentos para agregar")
        
def calculate_chunk_ids(chunks):
    #identificador en DB "VectorDB/title.pdf:index:niveles"
    last_page_id = None
    current_chunk_index = 0
    print("Calculando IDs de los Chunks")
    for chunk in tqdm(chunks):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("index")
        nivel=chunk.metadata.get("nivel")
        current_page_id = f"{source.split("/")[-1]}:{page}:{nivel}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
    return chunks
def clear_database_chroma():
    if os.path.exists(RUTACHROMA):
        shutil.rmtree(RUTACHROMA)
if __name__ == "__main__":
    main()
