# base de generación de embedings ["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"]

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
def get_embedding_function_local(local_llm="",port="11434"):
    local_llm=os.getenv('LLM_MODEL') if local_llm=="" else local_llm
    embeddings = OllamaEmbeddings(model=local_llm,base_url='http://localhost:'+port+'/')
    return embeddings

def get_OPENIA_embedding_function(llmmodel=""):
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_ORGANIZATION=os.getenv('OPENAI_ORGANIZATION')
    llmmodel=os.getenv('OPENAI_LLM_MODEL') if llmmodel=="" else "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openas_api_key=OPENAI_API_KEY, openai_organization=OPENAI_ORGANIZATION) #text-embedding-3-large  text-embedding-3-small
    return embeddings
