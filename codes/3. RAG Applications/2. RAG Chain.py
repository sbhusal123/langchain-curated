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

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ChatMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

prompt = """
You are a customer support specialist.
question: {question}

You assist user with general inquiries, based on {context}
and technical issues.
"""

system_message_prompt_template = SystemMessagePromptTemplate.from_template(prompt)

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model='llama3:latest')


retriver = db.as_retriever()

chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough()
    }
    | chat_prompt_template
    | model
    | StrOutputParser()
)

resp = chain.invoke(query)

print(resp)
