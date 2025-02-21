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
RUTACHROMA_PRODUCTS=RUTABASE+"\\LocalLLM\\VectorDB_Products"
COL_NAME_PROD="productsTest"
local_llm="llama3.1" if os.getenv('LLM_MODEL')==None else os.getenv('LLM_MODEL')


def main():
    parser = argparse.ArgumentParser()
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
        print("Usando embeddings de OpenAI")
        embedings = emb.get_OPENIA_embedding_function(local_llm2)
    else:
        print("Usando embeddings locales")
        embedings = emb.get_embedding_function_local(local_llm2,port=args.port)
        
    if args.reset:#==False:
        print(("🔥💣💥 " if symbols else "") +"Limpiando Base de datos")
        clear_database_chroma()
        TODO=1
    elif args.delcolection:#==False:
        print(("🔥 " if symbols else "") +"Borrando Coleccion")
        dbdel = Chroma(persist_directory=RUTACHROMA, embedding_function=embedings,collection_name=colectionName)
        dbdel.delete_collection()
        del dbdel
    elif args.delproducts:#==False:
        print(("🔥💽 " if symbols else "") +"Borrando Coleccion de PRODUCTOS")
        # dbdel = Chroma(persist_directory=RUTACHROMA_PRODUCTS, embedding_function=embedings,collection_name=COL_NAME_PROD)
        # dbdel.delete_collection()
        # del dbdel
        clear_database_chroma(RUTACHROMA_PRODUCTS)
    print("Work with >>>>>> 🏷",local_llm2)
    
    dataCategOrg=pd.read_csv(FILE_CATEG,sep=";",header=0)
    dataCateg=dataCategOrg.copy()
    colname=list(dataCateg)[0]
    dataCateg[colname]=dataCategOrg[colname].str.replace('--',' type: ').str.replace('-',' ').str.replace('/',' for ')
    dataCategOrg=dataCateg.copy()
    dataCategOrg["depurado"]=dataCategOrg[colname].apply(lambda x: extract_types(x,r' type: (.*?) for '))
    dataCategOrg["depurado2"]=dataCategOrg[colname].apply(lambda x: extract_types(x,r' for (.*?) type: '))
    
    doclist=[]
    indi=0
    for row in dataCateg[colname]:
        orgdt=dataCategOrg.iloc[indi].str.split("/")
        doclist.append(Document(page_content=row,
                                metadata={"source": FILE_CATEG,"clase":(orgdt[-1])[-1],"nivel":len(orgdt),"index":indi,"catType":"ALL"}))
        doclist.append(Document(page_content=row.split(' for ')[-1],
                                metadata={"source": FILE_CATEG,"clase":(orgdt[-1])[-1],"nivel":len(orgdt),"index":indi,"catType":"LAST"}))
        indi=+1
    add_to_chroma(chunks=doclist,Nsim=symbols,embfunc=embedings,MODEL=local_llm2)
    textListALL=[]
    if args.text!="":#ENTRADA DE TEXTO MANUAL
        print(("💻 " if symbols else "") +"texto a embeddings desde CMD")
        textListALL = [args.text]
    else:
        print(("📝 " if symbols else "") +"Cargando Productos")
        dataProd=pd.read_csv(FILE_PRODU,sep=";",header=0,lineterminator='\r',usecols=["metaTitle","metaDescription","metaSpecs","vendorCategory","vendor"])
        for idAct,query_text in (dataProd[["metaTitle","vendorCategory"]].iterrows()):
            textsList=str(query_text["metaTitle"]).split(",")
            data2=("['"+str(query_text["vendorCategory"])+"']").replace("['[","[").replace("]']","]").replace("''","'")
            # print(data2)
            dataEval=eval("[None]" if data2=="['nan']" else data2)
            if len(textsList[0])<=2:
                text1= " ".join(textsList[:min(len(textsList),2)])
            else:
                text1=textsList[0]
            textsList=text1+((" "+" or ".join(dataEval)) if str(dataEval[0])!="None" else "")
            textListALL.append(textsList)
    
    print(("💽" if symbols else "") +"obteniendo embedinds de porductos")
    doclist=[]
    for iPr in tqdm(range(len(textListALL))):
        textAct=textListALL[iPr]
        doclist.append(Document(page_content=textAct,metadata={"source": Sep(FILE_PRODU)[-1],"index":iPr}))
    add_to_chroma(chunks=doclist,Nsim=symbols,embfunc=embedings,MODEL=COL_NAME_PROD,pathChroma=RUTACHROMA_PRODUCTS)
    
    print(("💻 📝" if symbols else "") +"obteniendo similaridades de porductos")
    db = Chroma(persist_directory=RUTACHROMA, embedding_function=embedings,collection_name=colectionName)
    dbProd = Chroma(persist_directory=RUTACHROMA_PRODUCTS, embedding_function=embedings,collection_name=COL_NAME_PROD)
    for iPr in tqdm(range(len(textListALL))):  
        # Search the DB.
        # results = db.similarity_search_with_score(text, k=10,filter={"catType":"ALL"})
        # results2 = db.similarity_search_with_score(text, k=10,filter={"catType":"LAST"})
        # results3=db.max_marginal_relevance_search(text, k=10,filter={"catType":"ALL"})
        # results4=db.max_marginal_relevance_search(text, k=10,filter={"catType":"LAST"})
        # dta=db.get(where_document={"$contains": results3[0].page_content})
        # idxActual=dta["metadatas"][0]["index"]
        break
        
    # print(idxActual)
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

def calculate_chunk_ids(chunks):
    """create a list of chunk and ids, add metadata

    Args:
        chunks (List[Documents]): parts of text

    Returns:
        List[Documents]: all chunks with metadata
    """
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

def clear_database_chroma(path=RUTACHROMA):
    """Delete all data from Chroma database.
    """
    if os.path.exists(path):
        shutil.rmtree(path)

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
    chunks_with_ids = calculate_chunk_ids(chunks)
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Numero de partes de documentos en la DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    lastId=len(chunks_with_ids)-1
    print("Verificar Chunks para: "+str(len(chunks_with_ids)))
    for idchunk in tqdm(range(len(chunks_with_ids))):
        chunk=chunks_with_ids[idchunk]
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
        if idchunk!=0 and (idchunk%5000==0 or idchunk>=lastId):
            if len(new_chunks):
                print(("👉🔧 " if Nsim==True else "")+"agregando documentos "+str(len(new_chunks))+" desde "+str(idchunk+1-len(new_chunks))+" hasta "+str(idchunk)+" de "+str(lastId+1)+" chunks")
                # print(("👉 " if Nsim==True else "")+ f"Agregando nuevos documentos: {len(new_chunks)}")
                new_chunk_ids = [chunk.metadata["id"] for chunk in (new_chunks)]
                if pathChroma=="" or pathChroma==RUTACHROMA:
                    new_chunk_clases = [chunk.metadata["clase"] for chunk in (new_chunks)]
                else:
                    new_chunk_clases = ["product_"+str(ichunk) for ichunk in range(len(new_chunks))]
                db.add_documents(new_chunks, ids=new_chunk_ids,clases=new_chunk_clases)
                print(("✨ " if Nsim else "") +"Base de datos actualizada")
                # db.persist()
                new_chunks=[]
            else:
                print(("✅ " if Nsim==True else "")+ "No hay nuevos documentos para agregar"+str(idchunk+1-len(new_chunks))+" hasta "+str(idchunk))
    if len(new_chunks)<=0:
        print(("✅ " if Nsim==True else "")+ "No mas nuevos documentos para agregar")
if __name__ == "__main__":
    main()
    