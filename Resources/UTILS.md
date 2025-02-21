# Initial commands
```
python -m venv /LILOTest
c:\LILOTest\Scripts\activate
```
# install dependences:
```
pip install --upgrade pip
```
## direct intsall
```
pip install -r requeriments.txt
```
## step install
```
pip install -U pip setuptools wheel
pip install -U spacy
pip install seaborn
pip install scikit-learn
pip install pandas
pip install gensim
pip install langchain chromadb gradio ollama pypdf
pip install openai
pip install -U langchain-community
pip install langchain_ollama
pip install langchain_openai
pip install langchain-chroma
```

# ollama
## install ollama linux:
```
curl -fsSL https://ollama.com/install.sh | sh 
```

## install ollama windows or MAC:
downloan and install this https://ollama.com/download/windows
https://ollama.com/download/mac

## obtain models, select one or two to download (you must change the .env file to use the model):
```
ollama list
ollama pull llama3.2
ollama pull llama3.1
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:7b
```

## obtain word embdings model 
embModels=["bge-m3","granite-embedding:278m","snowflake-arctic-embed2", "avr/sfr-embedding-mistral","mxbai-embed-large", "nomic-embed-text"] the models used depend of the task, you can use one or more models, you can use the following command to download the models
```
ollama pull bge-m3
ollama pull granite-embedding:278m
ollama pull snowflake-arctic-embed2
ollama pull avr/sfr-embedding-mistral
ollama pull mxbai-embed-large
ollama pull nomic-embed-text
```


## update models powershell
```
ollama list | Select-Object -Skip 1 | ForEach-Object {
    $model = ($_ -split '\s+')[0]
    if ($model -and $model -ne "NAME") {
        Write-Host "Updating model: $model"
        ollama pull $model
    }
}
```
