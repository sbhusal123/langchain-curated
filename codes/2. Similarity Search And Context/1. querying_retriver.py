from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """
Answer the question based on following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3:latest")

query = "contribution in phy?"

vector_store = FAISS.from_texts(
    ["Ram was awarded with Genius Book Award for his contribution to phyics.",
     "Sita was awarded with Genius Book Award for her contribution to computer science.",
     "Laxman was awarded with Genius Book Award for his contribution to semiconductor science.",
    ],
    embedding=OllamaEmbeddings(model="llama3:latest")
)

docs = vector_store.similarity_search(query, k=1)
# print(docs[0].page_content)

retriver = vector_store.as_retriever()
# docs = retriver.invoke(query)
# print(docs[0].page_content)

retrival_chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough() # contribution in phy?
    } 
    | prompt
    | model
    | StrOutputParser()
)

output = retrival_chain.invoke(query)
print(output)
