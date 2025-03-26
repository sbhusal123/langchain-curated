from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Loading and Splitting Text
raw_text = TextLoader('./readme.txt').load()
splitter  = CharacterTextSplitter(separator='.', chunk_size=100, chunk_overlap=10)
docs = splitter.split_documents(raw_text)


# construct embeddings
from langchain_chroma.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings


embeding = OllamaEmbeddings(model='nomic-embed-text:latest')
db = Chroma.from_documents(docs, embeding)

query = "How about refund policy?"

# query for similarity
result = db.similarity_search(query, k=1)

