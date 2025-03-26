from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# similarity search query
query = "contribution in phy?"

# vector store
vector_store = FAISS.from_texts(
    ["Ram was awarded with Genius Book Award for his contribution to phyics.",
     "Sita was awarded with Genius Book Award for her contribution to computer science.",
     "Laxman was awarded with Genius Book Award for his contribution to semiconductor science.",
    ],
    embedding=OllamaEmbeddings(model="llama3:latest")
)

# similarity search for 1 nearest document
docs = vector_store.similarity_search(query, k=1)
# print(docs[0].page_content)

# as a retriever in rag chain
retriver = vector_store.as_retriever()
# docs = retriver.invoke(query)
# print(docs[0].page_content)
